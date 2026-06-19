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

Requires the `train` extra:  pip install -e ".[train]"
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from code_rl_env import CodeEnv, Rubric, TaskSpec
from code_rl_env.episode import Trajectory, Turn


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


# ── generation ─────────────────────────────────────────────────────────────────
def _chat_text(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(model, tokenizer, system_prompt: str, user_text: str,
                 temperature: float, max_new_tokens: int, max_prompt_len: int
                 ) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Generate ONE completion. Returns (prompt_ids[1,P], comp_ids[C], text)."""
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


# ── rollout ────────────────────────────────────────────────────────────────────
@dataclass
class TrajTokens:
    turns: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)  # (prompt_ids, comp_ids)
    truncated_flags: List[bool] = field(default_factory=list)


def rollout_group(model, tokenizer, task: TaskSpec, rubric: Rubric, cfg: GRPOConfig,
                  temperature: float, rng: random.Random
                  ) -> Tuple[List[Trajectory], List[TrajTokens]]:
    """Roll out G independent multi-turn trajectories on the SAME task (one GRPO group)."""
    trajs: List[Trajectory] = []
    tokens: List[TrajTokens] = []
    for _ in range(cfg.G):
        env = CodeEnv([task], rubric=rubric, max_turns=cfg.max_turns, rng=rng)
        obs = env.reset(task)
        traj = Trajectory(task_id=task.task_id)
        tt = TrajTokens()
        done = False
        while not done:
            p_ids, c_ids, text = generate_one(
                model, tokenizer, env.system_prompt, obs.prompt_text,
                temperature, cfg.max_new_tokens, cfg.max_prompt_len)
            sr = env.step(text)
            traj.turns.append(Turn(obs.prompt_text, text, sr.reward,
                                   sr.observation.feedback or "", sr.info["breakdown"], sr.info))
            tt.turns.append((p_ids, c_ids))
            tt.truncated_flags.append(len(c_ids) >= cfg.max_new_tokens)
            obs, done = sr.observation, sr.done
        trajs.append(traj)
        tokens.append(tt)
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

        trajs, tokens = rollout_group(policy, tokenizer, task, rubric, cfg, temperature, rng)

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

        log["steps"].append(step)
        log["losses"].append(loss.item())
        log["rewards"].append(rewards.mean().item())
        log["reward_stds"].append(rewards.std().item())
        log["entropies"].append(torch.stack(entropies).mean().item())
        log["grad_norms"].append(grad_norm.item())
        log["kl"].append(float(kl_val))
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
