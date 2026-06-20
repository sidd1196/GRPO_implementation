"""GRPO as a *client* of CodeEnv.

This module knows about torch/transformers; the environment does not. The trainer's job:
  1. roll out G multi-turn trajectories on the same task (a GRPO group),
  2. ask the env for each trajectory's reward (it never computes reward itself),
  3. group-normalise rewards -> advantages, broadcast to every turn's tokens,
  4. take a clipped policy-gradient step.

Two configs share one loop:
  GRPO baseline     : KL to a frozen ref, symmetric clip eps=0.2, fixed temperature.
  MicroCoder-GRPO   : no KL + high upper clip (Fix 3), two-stage temperature (Fix 2),
                      truncation masking (Fix 1). (arxiv 2603.07777)

Rollout backend (the actor / learner split — see README):
  "hf"          : reference loop, one sequence at a time (slow, for debugging).
  "hf_batched"  : one batched model.generate() per turn across the live trajectories (~3-5x).
  "vllm"        : a vLLM engine samples all trajectories with continuous batching (~10x).
                  After each optimizer step the LoRA adapter is merged and the weights are
                  pushed into the running vLLM engine, so the actor stays on-policy.

The GRPO math is identical across backends — only generation speed changes.

Requires the `train` extra:  pip install -e ".[train]"   (+ ".[vllm]" for backend="vllm")
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from code_rl_env import CodeEnv, Rubric, TaskSpec
from code_rl_env.episode import Trajectory, Turn

# A backend generator turns a list of (system_prompt, user_text) prompts into a list of
# (prompt_ids[1,P], comp_ids[C], completion_text, truncated) — one per prompt, in order.
GenResult = Tuple[torch.Tensor, torch.Tensor, str, bool]
GenFn = Callable[[List[Tuple[str, str]], float], List[GenResult]]


# ── config ─────────────────────────────────────────────────────────────────────
@dataclass
class GRPOConfig:
    G: int = 4
    num_steps: int = 200
    lr: float = 1e-5
    max_turns: int = 3                 # multi-turn episode cap (1 = single-shot bandit)
    max_new_tokens: int = 256
    max_prompt_len: int = 1024
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2          # raised to 0.5 for MicroCoder (Fix 3)
    kl_coeff: float = 0.01             # 0 for MicroCoder (Fix 3)
    temperature: float = 0.8
    # MicroCoder fixes
    microcoder: bool = False
    temp_stage1: float = 0.7           # Fix 2
    temp_stage2: float = 1.0
    temp_switch_step: int = 100
    mask_prob: float = 0.3             # Fix 1: rho
    repeat_check_len: int = 128
    # credit: use the final-turn reward or the best across turns
    reward_mode: str = "final"         # "final" | "best"
    eval_every: int = 50
    seed: int = 42
    # rollout backend (actor)
    backend: str = "hf"                # "hf" | "hf_batched" | "vllm"
    vllm_gpu_util: float = 0.40        # fraction of GPU mem vLLM may grab (training model needs the rest)
    vllm_model: Optional[str] = None   # base model id for vLLM; inferred from the policy if None


# ── generation: shared helpers ───────────────────────────────────────────────────
def _chat_text(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(model, tokenizer, system_prompt: str, user_text: str,
                 temperature: float, max_new_tokens: int, max_prompt_len: int
                 ) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Generate ONE completion (used by backend='hf' and by evaluate()).
    Returns (prompt_ids[1,P], comp_ids[C], text)."""
    text = _chat_text(tokenizer, system_prompt, user_text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_prompt_len).to(model.device)
    plen = inputs.input_ids.shape[1]
    greedy = temperature == 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    comp_ids = out[0, plen:].detach().cpu()
    completion = tokenizer.decode(comp_ids, skip_special_tokens=True).strip()
    return inputs.input_ids.detach().cpu(), comp_ids, completion


def get_log_probs(model, prompt_ids: torch.Tensor, comp_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs of comp_ids given prompt_ids. Returns [C]."""
    dev = next(model.parameters()).device
    p_ids, c_ids = prompt_ids.to(dev), comp_ids.to(dev)
    if c_ids.numel() == 0:
        return torch.tensor([-10.0], device=dev)
    ids = torch.cat([p_ids, c_ids.unsqueeze(0)], dim=1)
    logits = model(input_ids=ids).logits[0]
    plen = p_ids.shape[1]
    log_probs = F.log_softmax(logits[plen - 1:-1], dim=-1)
    return log_probs[torch.arange(len(c_ids)), c_ids]


def _cut_completion(gen_ids: torch.Tensor, eos_id: int, max_new_tokens: int
                    ) -> Tuple[torch.Tensor, bool]:
    """Trim a generated id sequence at the first EOS (kept as a real action) and report
    whether it was truncated (hit the token budget without emitting EOS)."""
    eos_pos = (gen_ids == eos_id).nonzero()
    if eos_pos.numel() > 0:
        return gen_ids[: eos_pos[0].item() + 1], False
    return gen_ids, len(gen_ids) >= max_new_tokens


# ── backend: HuggingFace (single + batched) ──────────────────────────────────────
def _make_hf_gen(model, tokenizer, cfg: GRPOConfig, batched: bool) -> GenFn:
    def gen(prompts: List[Tuple[str, str]], temperature: float) -> List[GenResult]:
        if not batched:
            out: List[GenResult] = []
            for sys_p, user_p in prompts:
                p_ids, c_ids, text = generate_one(model, tokenizer, sys_p, user_p,
                                                  temperature, cfg.max_new_tokens, cfg.max_prompt_len)
                out.append((p_ids, c_ids, text, len(c_ids) >= cfg.max_new_tokens))
            return out

        # batched: one generate() call across all prompts (left-padded so completions align)
        chat_texts = [_chat_text(tokenizer, s, u) for s, u in prompts]
        enc = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=cfg.max_prompt_len).to(model.device)
        greedy = temperature == 0.0
        with torch.no_grad():
            gen_out = model.generate(
                **enc, max_new_tokens=cfg.max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        plen = enc.input_ids.shape[1]            # padded prompt length (same for all rows)
        results: List[GenResult] = []
        for i, text in enumerate(chat_texts):
            comp, trunc = _cut_completion(gen_out[i, plen:].detach().cpu(),
                                          tokenizer.eos_token_id, cfg.max_new_tokens)
            # store the *unpadded* prompt ids — get_log_probs concatenates prompt+comp
            p_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=cfg.max_prompt_len).input_ids
            decoded = tokenizer.decode(comp, skip_special_tokens=True).strip()
            results.append((p_ids, comp, decoded, trunc))
        return results

    return gen


# ── backend: vLLM (fast actor) ────────────────────────────────────────────────────
def _get_vllm_model(llm):
    """Reach the underlying model inside a (single-GPU) vLLM engine, across versions."""
    paths = (
        lambda e: e.model_executor.driver_worker.model_runner.model,
        lambda e: e.model_executor.driver_worker.worker.model_runner.model,
    )
    eng = llm.llm_engine
    for p in paths:
        try:
            return p(eng)
        except AttributeError:
            continue
    raise RuntimeError("could not locate the vLLM model for weight sync; check vLLM version")


def _merged_state_dict(policy):
    """Merge the LoRA adapter into the base weights, yield (name, tensor) pairs with clean
    HuggingFace names, then unmerge so training continues on the adapter."""
    policy.merge_adapter()
    try:
        base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        sd = base.state_dict()
        cleaned = []
        for name, w in sd.items():
            if "lora_" in name:                      # skip adapter params (already merged in)
                continue
            cleaned.append((name.replace(".base_layer", ""), w))
    finally:
        policy.unmerge_adapter()
    return cleaned


def _vllm_load_weights(worker, weights):
    """Runs *inside* the vLLM worker (via collective_rpc): load merged weights into the live
    model. With the engine in-process (colocated), `weights` are the trainer's GPU tensors —
    no cross-process copy."""
    worker.model_runner.model.load_weights(weights)


class VLLMRollout:
    """A vLLM engine used purely as the sampler, with per-step weight sync from the policy."""

    def __init__(self, model_name: str, tokenizer, cfg: GRPOConfig):
        import os
        # Keep the V1 engine in the SAME process (no worker subprocess) so weight sync can hand
        # the policy's GPU tensors straight to the model. V1's default multiprocessing moves the
        # model into a separate process, which both breaks the old model path and would force us
        # to copy ~3GB of weights across processes every step.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=cfg.vllm_gpu_util,
            max_model_len=cfg.max_prompt_len + cfg.max_new_tokens,
            enable_prefix_caching=False,             # weights change every step — no stale cache
            enforce_eager=True,                      # skip CUDA-graph capture (we hot-swap weights)
        )

    def sync_weights(self, policy) -> None:
        weights = _merged_state_dict(policy)
        # Preferred (V1): run load_weights inside the worker via collective_rpc.
        try:
            self.llm.collective_rpc(_vllm_load_weights, args=(weights,))
        except (AttributeError, TypeError):
            # Fallback for older (V0) engines that expose the model in-process.
            _get_vllm_model(self.llm).load_weights(weights)

    def close(self) -> None:
        """Release the engine's GPU memory so a second run can build a fresh one."""
        import gc
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment, destroy_model_parallel)
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        try:
            del self.llm.llm_engine.model_executor
        except Exception:
            pass
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()

    def gen_fn(self) -> GenFn:
        from vllm import SamplingParams

        def gen(prompts: List[Tuple[str, str]], temperature: float) -> List[GenResult]:
            chat_texts = [_chat_text(self.tokenizer, s, u) for s, u in prompts]
            prompt_ids = [self.tokenizer(t, truncation=True,
                                         max_length=self.cfg.max_prompt_len).input_ids
                          for t in chat_texts]
            sp = SamplingParams(
                temperature=temperature if temperature > 0 else 0.0,
                top_p=1.0,
                max_tokens=self.cfg.max_new_tokens,
            )
            # vLLM's token-prompt input API moved from a kwarg (<=0.5) to a dict/TokensPrompt
            # (>=0.6); try the newer form first, fall back to the old kwarg.
            try:
                outs = self.llm.generate(
                    [{"prompt_token_ids": ids} for ids in prompt_ids],
                    sampling_params=sp, use_tqdm=False)
            except (TypeError, ValueError):
                outs = self.llm.generate(
                    prompt_token_ids=prompt_ids, sampling_params=sp, use_tqdm=False)
            results: List[GenResult] = []
            for pids, o in zip(prompt_ids, outs):
                comp = list(o.outputs[0].token_ids)
                trunc = o.outputs[0].finish_reason == "length"
                c_ids = torch.tensor(comp, dtype=torch.long) if comp else torch.zeros(0, dtype=torch.long)
                text = self.tokenizer.decode(comp, skip_special_tokens=True).strip()
                results.append((torch.tensor([pids], dtype=torch.long), c_ids, text, trunc))
            return results

        return gen


def _infer_model_name(policy) -> str:
    base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    name = getattr(base.config, "_name_or_path", None)
    if not name:
        raise ValueError("could not infer base model id for vLLM; set GRPOConfig.vllm_model")
    return name


def make_gen_fn(cfg: GRPOConfig, policy, tokenizer,
                vllm_backend: Optional["VLLMRollout"]) -> GenFn:
    if cfg.backend == "vllm":
        assert vllm_backend is not None
        return vllm_backend.gen_fn()
    if cfg.backend == "hf_batched":
        return _make_hf_gen(policy, tokenizer, cfg, batched=True)
    if cfg.backend == "hf":
        return _make_hf_gen(policy, tokenizer, cfg, batched=False)
    raise ValueError(f"unknown backend {cfg.backend!r}")


# ── rollout ────────────────────────────────────────────────────────────────────
@dataclass
class TrajTokens:
    turns: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)  # (prompt_ids, comp_ids)
    truncated_flags: List[bool] = field(default_factory=list)


def rollout_group(gen_fn: GenFn, task: TaskSpec, rubric: Rubric, cfg: GRPOConfig,
                  temperature: float, rng: random.Random
                  ) -> Tuple[List[Trajectory], List[TrajTokens]]:
    """Roll out G multi-turn trajectories on the SAME task (one GRPO group), stepping all
    live trajectories *in lockstep* so each turn's generations are produced in one batched
    call (the backend decides how to batch)."""
    envs, obss = [], []
    trajs: List[Trajectory] = []
    tokens: List[TrajTokens] = []
    for _ in range(cfg.G):
        env = CodeEnv([task], rubric=rubric, max_turns=cfg.max_turns, rng=rng)
        obss.append(env.reset(task))
        envs.append(env)
        trajs.append(Trajectory(task_id=task.task_id))
        tokens.append(TrajTokens())

    system_prompt = envs[0].system_prompt
    active = list(range(cfg.G))
    while active:
        prompts = [(system_prompt, obss[i].prompt_text) for i in active]
        gens = gen_fn(prompts, temperature)
        still_active = []
        for idx, (p_ids, c_ids, text, trunc) in zip(active, gens):
            sr = envs[idx].step(text)
            trajs[idx].turns.append(Turn(obss[idx].prompt_text, text, sr.reward,
                                         sr.observation.feedback or "", sr.info["breakdown"], sr.info))
            tokens[idx].turns.append((p_ids, c_ids))
            tokens[idx].truncated_flags.append(trunc)
            obss[idx] = sr.observation
            if not sr.done:
                still_active.append(idx)
        active = still_active
    return trajs, tokens


def _is_repetitive(comp_ids: torch.Tensor, m: int) -> bool:
    if len(comp_ids) < 2 * m:
        return False
    return comp_ids[-m:].tolist() == comp_ids[-2 * m:-m].tolist()


# ── training loop ──────────────────────────────────────────────────────────────
def run_grpo(policy, tokenizer, train_tasks: List[TaskSpec], cfg: GRPOConfig,
             ref_model=None, rubric: Optional[Rubric] = None,
             eval_tasks: Optional[List[TaskSpec]] = None) -> Dict:
    from tqdm.auto import tqdm

    rubric = rubric or Rubric()
    rng = random.Random(cfg.seed)
    optimizer = AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.01)
    device = next(policy.parameters()).device
    mode = "MicroCoder-GRPO" if cfg.microcoder else "GRPO"

    # actor: build the rollout backend (vLLM loads its own engine; HF reuses the policy)
    vllm_backend = None
    if cfg.backend == "vllm":
        model_name = cfg.vllm_model or _infer_model_name(policy)
        vllm_backend = VLLMRollout(model_name, tokenizer, cfg)
        vllm_backend.sync_weights(policy)          # align the engine with the (LoRA) policy at step 0
    gen_fn = make_gen_fn(cfg, policy, tokenizer, vllm_backend)
    tqdm.write(f"{mode}: rollout backend = {cfg.backend!r}")

    log = {k: [] for k in ("steps", "losses", "rewards", "reward_stds", "entropies",
                           "grad_norms", "kl", "temperatures", "solve_rate", "mean_turns",
                           "eval_steps", "eval_accs")}

    for step in tqdm(range(cfg.num_steps), desc=f"{mode} training"):
        task = rng.choice(train_tasks)

        # Fix 2: two-stage temperature (MicroCoder only)
        if cfg.microcoder:
            temperature = cfg.temp_stage2 if step >= cfg.temp_switch_step else cfg.temp_stage1
        else:
            temperature = cfg.temperature

        trajs, tokens = rollout_group(gen_fn, task, rubric, cfg, temperature, rng)

        # reward per trajectory -> group-normalised advantage
        if cfg.reward_mode == "best":
            rewards = torch.tensor([t.best_reward for t in trajs], device=device)
        else:
            rewards = torch.tensor([t.final_reward for t in trajs], device=device)
        if rewards.std() > 1e-6:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)

        optimizer.zero_grad()
        losses, kl_terms, entropies = [], [], []

        for g in range(cfg.G):
            adv = advantages[g]
            for ti, (p_ids, c_ids) in enumerate(tokens[g].turns):
                # Fix 1: truncation masking — don't penalise long, non-repetitive (maybe-correct) outputs
                turn_adv = adv
                if (cfg.microcoder and tokens[g].truncated_flags[ti]
                        and not _is_repetitive(c_ids, cfg.repeat_check_len)
                        and rng.random() < cfg.mask_prob):
                    turn_adv = torch.zeros((), device=device)

                lp = get_log_probs(policy, p_ids, c_ids)
                with torch.no_grad():
                    old_lp = lp.detach()
                ratio = torch.exp(lp.mean() - old_lp.mean())
                unclipped = ratio * turn_adv
                clipped = torch.clamp(ratio, 1 - cfg.epsilon_low, 1 + cfg.epsilon_high) * turn_adv
                losses.append(-torch.minimum(unclipped, clipped))
                entropies.append(-lp.mean().detach())

                # KL to frozen ref (GRPO baseline only; Fix 3 removes it)
                if not cfg.microcoder and ref_model is not None and cfg.kl_coeff > 0:
                    with torch.no_grad():
                        ref_lp = get_log_probs(ref_model, p_ids, c_ids)
                    kl_terms.append((torch.exp(ref_lp - lp) - (ref_lp - lp) - 1).mean())

        pg_loss = torch.stack(losses).mean()
        kl_val = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=device)
        loss = pg_loss + cfg.kl_coeff * kl_val

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # actor/learner sync: push the freshly-updated (merged) weights into the vLLM engine
        if vllm_backend is not None:
            vllm_backend.sync_weights(policy)

        log["steps"].append(step)
        log["losses"].append(loss.item())
        log["rewards"].append(rewards.mean().item())
        log["reward_stds"].append(rewards.std().item())
        log["entropies"].append(torch.stack(entropies).mean().item())
        log["grad_norms"].append(grad_norm.item())
        log["kl"].append(float(kl_val.detach()))
        log["temperatures"].append(temperature)
        log["solve_rate"].append(sum(t.solved for t in trajs) / cfg.G)
        log["mean_turns"].append(sum(t.n_turns for t in trajs) / cfg.G)

        if eval_tasks and (step + 1) % cfg.eval_every == 0:
            acc = evaluate(policy, tokenizer, eval_tasks, rubric)["pass@1"]
            log["eval_steps"].append(step + 1)
            log["eval_accs"].append(acc)
            tqdm.write(f"  step {step+1:3d} | loss={loss.item():.4f} | reward={rewards.mean():.2f} "
                       f"| solve={log['solve_rate'][-1]:.2f} | turns={log['mean_turns'][-1]:.2f} "
                       f"| eval_pass@1={acc:.3f}")

    if vllm_backend is not None:                   # free the engine so the next config can build one
        vllm_backend.close()
    return log


# ── evaluation (single-turn greedy pass@1; works for MBPP in-dist + HumanEval transfer) ──
def evaluate(policy, tokenizer, tasks: List[TaskSpec], rubric: Optional[Rubric] = None,
             max_new_tokens: int = 512, max_prompt_len: int = 1024) -> Dict:
    from tqdm.auto import tqdm

    rubric = rubric or Rubric()
    policy.eval()
    passed, results = [], []
    for t in tqdm(tasks, desc="eval", leave=False):
        user = f"Solve this Python task. The function must be named `{t.entry_point}`.\n\n{t.prompt}"
        from code_rl_env.environment import SYSTEM_PROMPT
        _, _, text = generate_one(policy, tokenizer, SYSTEM_PROMPT, user,
                                  temperature=0.0, max_new_tokens=max_new_tokens,
                                  max_prompt_len=max_prompt_len)
        _, vr, _ = rubric.score(text, t)
        ok = vr.all_passed
        results.append({"task_id": t.task_id, "passed": ok, "completion": text})
        if ok:
            passed.append(t.task_id)
    policy.train()
    return {"pass@1": len(passed) / len(tasks), "passed": passed, "results": results}
