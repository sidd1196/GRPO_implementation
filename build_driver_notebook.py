"""Generates grpo_rlvr_dapo_code.ipynb — a teaching DRIVER notebook that imports and
runs the code_rl_env package, narrating each layer. Run: python3 build_driver_notebook.py
"""
import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md(r"""
# GRPO on a Decoupled, Multi-Turn Code-RL Environment

This notebook is a **driver**. The actual logic lives in the `code_rl_env/` Python package
(version-controlled, unit-tested, GPU-free) and in `train_grpo.py`. Here we *import and run*
those pieces and explain what each one does.

**The idea.** The earlier version of this experiment fused everything — task sampling,
unit-test reward, and the GRPO update — into one training loop. That works, but it isn't an
*environment*: you can't test the reward without a GPU, you can't swap the algorithm, and you
can't make it multi-turn. Here we refactor it into a proper RL environment:

| Layer | File | Responsibility |
|---|---|---|
| `TaskSpec` + loaders | `code_rl_env/tasks.py` | MBPP / HumanEval → one task shape |
| Sandbox | `code_rl_env/sandbox.py` | run code in a timed subprocess |
| Verifier | `code_rl_env/verifier.py` | per-test pass/fail + error (the reward *source*) |
| Rubric | `code_rl_env/rubric.py` | weighted blend of named reward functions |
| **`CodeEnv`** | `code_rl_env/environment.py` | **Gym-style `reset()`/`step()`, multi-turn** |
| GRPO client | `train_grpo.py` | rolls out trajectories, consumes the env's reward |

The environment is **model-agnostic** (it only deals in text and never imports torch). GRPO is
just *one* consumer of it — eval and best-of-n could be others.
"""))

cells.append(md(r"""
## 0 · Setup — clone the package and install it

On Colab this clones the repo and installs the package in editable mode. The `[train]`
extra pulls torch / transformers / peft (already present on a Colab GPU runtime).
"""))

cells.append(code(r"""
import os
REPO = "GRPO_implementation"
if not os.path.isdir(REPO) and not os.path.isdir("code_rl_env"):
    !git clone -q https://github.com/sidd1196/GRPO_implementation.git
if os.path.isdir(REPO):
    %cd {REPO}
!git pull -q 2>/dev/null
!pip install -q -e ".[train]"
# Colab's base image ships an old torchao that the installed peft rejects; we don't use it
# (plain bf16 LoRA), so remove it to avoid an ImportError when building the LoRA policy.
!pip uninstall -q -y torchao 2>/dev/null

print("package installed")
print("⚠️ If torch was already imported this session, RESTART the runtime once "
      "(Runtime → Restart session), then run from this cell.")
"""))

cells.append(md(r"""
## 1 · `TaskSpec` — the unit of work

A task is a problem statement, the required function name, and a list of executable test
snippets. MBPP and HumanEval are normalised into the **same** shape, so one environment and
one eval harness serve both in-distribution (MBPP) and transfer (HumanEval).
"""))

cells.append(code(r"""
from code_rl_env import load_mbpp

mbpp = load_mbpp(limit=180)
t = mbpp[0]
print("task_id   :", t.task_id)
print("entry pt  :", t.entry_point)
print("prompt    :", t.prompt)
print("tests     :", t.tests)
print(f"\nloaded {len(mbpp)} MBPP tasks")
"""))

cells.append(md(r"""
## 2 · Verifier — the reward *source* (no GPU)

`ExecutionVerifier.verify` runs each test independently in a sandboxed subprocess and returns
a structured result: which tests passed, and the error text for the ones that didn't. Two
design choices matter:

- **Partial credit** (fraction of tests passed) gives a dense gradient even when no completion
  fully solves a task — critical early in training with a small model.
- The captured **error** becomes the feedback the model sees on the next turn.

We demo on a tiny toy task so the behaviour is obvious and deterministic.
"""))

cells.append(code(r"""
from code_rl_env import TaskSpec, ExecutionVerifier

demo = TaskSpec(
    task_id="demo/double", prompt="Return x doubled.",
    tests=["assert f(2) == 4", "assert f(3) == 6", "assert f(0) == 0"],
    entry_point="f", source="demo",
)

v = ExecutionVerifier()
print("Correct solution:")
vr = v.verify("def f(x):\n    return x * 2", demo)
print(f"  passed {vr.n_passed}/{vr.n_total}  fraction={vr.fraction_passed:.2f}  all_passed={vr.all_passed}")

print("\nPartially-correct solution (x + 2):")
vr = v.verify("def f(x):\n    return x + 2", demo)
print(f"  passed {vr.n_passed}/{vr.n_total}  fraction={vr.fraction_passed:.2f}")
print("  feedback the model would see next turn:")
print("   ", vr.feedback().replace(chr(10), chr(10) + "    "))
"""))

cells.append(md(r"""
## 3 · Rubric — compose weighted reward functions

A reward needn't be a single hard-coded number. A `Rubric` is a weighted blend of named
reward functions (`tests`, `syntax`, `format`, …). Swapping in a denser rubric never touches
the environment or the trainer — that's the decoupling. (This mirrors the `verifiers`-style
`Rubric` abstraction used in modern LLM-RL stacks.)
"""))

cells.append(code(r"""
from code_rl_env import default_rubric, dense_rubric

# A fully-correct solution that is wrapped in markdown fences (a common chat-model habit):
fenced_correct = "```python\ndef f(x):\n    return x * 2\n```"

for name, rubric in [("default (tests only)", default_rubric()), ("dense (tests+syntax+format)", dense_rubric())]:
    reward, vr, breakdown = rubric.score(fenced_correct, demo)
    print(f"{name:32s} reward={reward:.3f}  breakdown={ {k: round(v,2) for k,v in breakdown.items()} }")
print("\nThe dense rubric docks the fenced output on `format` even though all tests pass.")
"""))

cells.append(md(r"""
## 4 · `CodeEnv` — the multi-turn environment

This is the core. The Gym-style API:

```python
obs  = env.reset(task)          # -> the instruction to give the model
step = env.step(completion)     # -> StepResult(observation, reward, done, info)
```

**Multi-turn dynamics:** turn 0 the model writes a function; if it fails, the next observation
carries *its previous attempt plus the failing test/traceback*, and it gets to revise — up to
`max_turns`, terminating early on a full pass. This write → run → read-error → revise loop is
what makes it an *environment* rather than a one-shot bandit.

We drive it here with **hand-written completions** (no model needed) to show the protocol.
"""))

cells.append(code(r"""
from code_rl_env import CodeEnv

env = CodeEnv([demo], max_turns=3)
obs = env.reset(demo)
print("── TURN 0 — the model sees:")
print(obs.prompt_text)

print("\n── It submits a WRONG function:  def f(x): return x + 2")
sr = env.step("def f(x):\n    return x + 2")
print(f"   reward={sr.reward:.2f}  done={sr.done}  solved={sr.info['solved']}")

print("\n── TURN 1 — the env now shows the failure and asks for a fix:")
print(sr.observation.prompt_text)

print("\n── It submits the CORRECTED function:  def f(x): return x * 2")
sr = env.step("def f(x):\n    return x * 2")
print(f"   reward={sr.reward:.2f}  done={sr.done}  solved={sr.info['solved']}")
"""))

cells.append(md(r"""
## 5 · Prove the environment is correct — no GPU

Because the env is decoupled from the model, its correctness is testable with plain `pytest`.
These run anywhere (CI, your laptop) in a couple of seconds — the verifier scores known code
correctly and the multi-turn protocol behaves. *This* is the payoff of treating it as infra.
"""))

cells.append(code(r"""
!pytest -q tests/
"""))

cells.append(md(r"""
## 6 · GRPO as a *client* of the environment

Now the only GPU part. `train_grpo.run_grpo`:

1. samples a task and rolls out **G multi-turn trajectories** (one GRPO group),
2. asks the env for each trajectory's reward — it never computes reward itself,
3. group-normalises rewards → advantages, broadcasts them to every turn's tokens,
4. takes a clipped policy-gradient step.

The same loop runs two configs via `GRPOConfig`:

- **GRPO baseline** — KL to a frozen reference, symmetric clip ε=0.2.
- **MicroCoder-GRPO** (arxiv 2603.07777) — the three code-specific fixes: no-KL + high upper
  clip (Fix 3), two-stage temperature (Fix 2), truncation masking (Fix 1).

**Rollout backend.** We use `backend="hf_batched"`: each turn, the `G` live trajectories are
generated in one batched `model.generate()` call on the policy itself. Because the sampler *is*
the policy, there is no weight sync — but sampling and training still run one after the other,
so the loop is synchronous and on-policy. `run_grpo` logs per-step generation / sandbox /
training / sync time; `timing_summary(log)` reports the breakdown. (`backend="vllm"` — a separate
sampler engine with per-step weight sync — is still in `train_grpo.py` but not used here.)
"""))

cells.append(code(r"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from code_rl_env import load_mbpp, load_humaneval, default_rubric
from train_grpo import GRPOConfig, run_grpo, evaluate, timing_summary

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

rubric        = default_rubric()
tasks         = load_mbpp(limit=180)
train_tasks   = tasks[:150]
eval_mbpp     = tasks[150:180]          # in-distribution held-out
humaneval     = load_humaneval()        # transfer

def fresh_policy():
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(m, lora)

print(f"train={len(train_tasks)}  eval_mbpp={len(eval_mbpp)}  humaneval={len(humaneval)}")

# Results go to Google Drive: Colab's own disk is wiped when the runtime disconnects.
import json, os
from google.colab import drive
drive.mount("/content/drive")
RESULTS_DIR = "/content/drive/MyDrive/GRPO_implementation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Write one run's training log, timing summary and full eval results (incl. which
# problems passed) to RESULTS_DIR/<name>.json — called right after each run.
def save_results(name, log=None, **evals):
    out = {"log": log, "timing": timing_summary(log) if log else None, "evals": evals}
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved", f"{RESULTS_DIR}/{name}.json")
"""))

cells.append(md(r"""
### 6a · Baseline measurement — report BOTH in-distribution and transfer

The original experiment only measured HumanEval (transfer), which made RL look like it *hurt*.
We now also measure held-out MBPP (in-distribution) — where RL is trained — so the
"where does RL actually help?" question gets an honest answer.
"""))

cells.append(code(r"""
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")
base_mbpp_res = evaluate(base, tokenizer, eval_mbpp, rubric)
base_he_res   = evaluate(base, tokenizer, humaneval, rubric)
base_mbpp, base_he = base_mbpp_res["pass@1"], base_he_res["pass@1"]
print(f"BASE   MBPP(in-dist)={base_mbpp:.3f}   HumanEval(transfer)={base_he:.3f}")
save_results("base", mbpp=base_mbpp_res, humaneval=base_he_res)
del base; torch.cuda.empty_cache()
"""))

cells.append(md("### 6b · GRPO baseline (KL + symmetric clip)"))

cells.append(code(r"""
policy = fresh_policy()
ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")
for p in ref.parameters():
    p.requires_grad_(False)

grpo_cfg = GRPOConfig(microcoder=False, num_steps=200, G=4, max_turns=3,
                      kl_coeff=0.01, epsilon_low=0.2, epsilon_high=0.2, temperature=0.8,
                      backend="hf_batched")      # one batched model.generate() per turn
grpo_log = run_grpo(policy, tokenizer, train_tasks, grpo_cfg,
                    ref_model=ref, rubric=rubric, eval_tasks=eval_mbpp)

grpo_mbpp_res = evaluate(policy, tokenizer, eval_mbpp, rubric)
grpo_he_res   = evaluate(policy, tokenizer, humaneval, rubric)
grpo_mbpp, grpo_he = grpo_mbpp_res["pass@1"], grpo_he_res["pass@1"]
print(f"GRPO   MBPP(in-dist)={grpo_mbpp:.3f}   HumanEval(transfer)={grpo_he:.3f}")
for k, v in timing_summary(grpo_log).items():
    print(f"  {k:15s} {v:.3f}")
save_results("grpo", grpo_log, mbpp=grpo_mbpp_res, humaneval=grpo_he_res)
del policy, ref; torch.cuda.empty_cache()
"""))

cells.append(md("### 6c · MicroCoder-GRPO (3 code-specific fixes, no KL)"))

cells.append(code(r"""
mc_policy = fresh_policy()
mc_cfg = GRPOConfig(microcoder=True, num_steps=200, G=4, max_turns=3,
                    kl_coeff=0.0, epsilon_low=0.2, epsilon_high=0.5,
                    temp_stage1=0.7, temp_stage2=1.0, temp_switch_step=100,
                    mask_prob=0.3, repeat_check_len=128,
                    backend="hf_batched")
mc_log = run_grpo(mc_policy, tokenizer, train_tasks, mc_cfg,
                  ref_model=None, rubric=rubric, eval_tasks=eval_mbpp)

mc_mbpp_res = evaluate(mc_policy, tokenizer, eval_mbpp, rubric)
mc_he_res   = evaluate(mc_policy, tokenizer, humaneval, rubric)
mc_mbpp, mc_he = mc_mbpp_res["pass@1"], mc_he_res["pass@1"]
print(f"MicroCoder-GRPO   MBPP(in-dist)={mc_mbpp:.3f}   HumanEval(transfer)={mc_he:.3f}")
for k, v in timing_summary(mc_log).items():
    print(f"  {k:15s} {v:.3f}")
save_results("micro", mc_log, mbpp=mc_mbpp_res, humaneval=mc_he_res)
"""))

cells.append(md("## 7 · Results — in-distribution vs transfer"))

cells.append(code(r"""
print(f"{'Model':<20}{'MBPP (in-dist)':>16}{'HumanEval (transfer)':>22}")
print("-" * 58)
for name, m, h in [("Base", base_mbpp, base_he),
                   ("GRPO", grpo_mbpp, grpo_he),
                   ("MicroCoder-GRPO", mc_mbpp, mc_he)]:
    print(f"{name:<20}{m:>16.3f}{h:>22.3f}")

# Paired: pass@1 differences are within noise on 30 problems, so look at the problems
# where the two methods DISAGREE — that is the sharper comparison.
lines = []
for split, g_res, m_res in [("MBPP", grpo_mbpp_res, mc_mbpp_res),
                            ("HumanEval", grpo_he_res, mc_he_res)]:
    g, m = set(g_res["passed"]), set(m_res["passed"])
    lines.append(f"{split:<10} only GRPO solved {len(g - m):>3} | only MicroCoder solved {len(m - g):>3}")
print("\n" + "\n".join(lines))

with open(f"{RESULTS_DIR}/summary.txt", "w") as f:
    for name, m, h in [("Base", base_mbpp, base_he), ("GRPO", grpo_mbpp, grpo_he),
                       ("MicroCoder-GRPO", mc_mbpp, mc_he)]:
        f.write(f"{name:<20} MBPP={m:.3f}  HumanEval={h:.3f}\n")
    f.write("\n" + "\n".join(lines) + "\n\nTiming (mean s/step):\n")
    for name, log in [("GRPO", grpo_log), ("MicroCoder", mc_log)]:
        f.write(f"{name}: " + "  ".join(f"{k}={v:.3f}" for k, v in timing_summary(log).items()) + "\n")
print("saved", f"{RESULTS_DIR}/summary.txt")
"""))

cells.append(code(r"""
import matplotlib.pyplot as plt

# Each step trains on ONE problem, so raw per-step curves are very noisy: plot a moving average.
def smooth(xs, w=10):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]

curves = [("rewards",       "mean group reward"),
          ("solve_rate",    "group solve rate (all tests passed)"),
          ("comp_len_mean", "mean answer length (tokens)"),
          ("trunc_rate",    "share of answers cut off at the token limit")]
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (key, title) in zip(axes.flat, curves):
    ax.plot(smooth(grpo_log[key]), label="GRPO")
    ax.plot(smooth(mc_log[key]), label="MicroCoder")
    ax.set_title(f"{title}  (10-step avg)"); ax.set_xlabel("step"); ax.legend()
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150); plt.show()

# Where does each step's time go? Phases run one after another — nothing overlaps.
phases = [("t_gen", "generate (GPU busy)"), ("t_env", "run tests (GPU idle)"),
          ("t_train", "train (GPU busy)"), ("t_sync", "weight sync (GPU idle)")]
tg, tm = timing_summary(grpo_log), timing_summary(mc_log)
fig, ax = plt.subplots(figsize=(8, 2.8))
left = [0.0, 0.0]
for key, label in phases:
    vals = [tg[key], tm[key]]
    if max(vals) == 0:                       # e.g. no weight sync with hf_batched
        continue
    ax.barh(["GRPO", "MicroCoder"], vals, left=left, label=label)
    left = [l + v for l, v in zip(left, vals)]
ax.set_xlabel("seconds per training step (mean)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/timing.png", dpi=150); plt.show()
"""))

cells.append(md(r"""
### 7b · Examples — the code each model actually wrote

For problems where GRPO and MicroCoder-GRPO **disagree** (one passed, the other failed), print
the task, then the base / GRPO / MicroCoder code side by side, with the first failing test and
its error. Reads the saved results from Drive, so after a restart just re-run the setup cells.
"""))

cells.append(code(r"""
res = {n: json.load(open(f"{RESULTS_DIR}/{n}.json"))["evals"] for n in ("base", "grpo", "micro")}
tasks_by_id = {t.task_id: t for t in eval_mbpp + humaneval}

def show_example(split, task_id):
    t = tasks_by_id[task_id]
    print("=" * 80 + f"\n{split} | {task_id}\n\nTASK:\n{t.prompt.strip()}\n")
    for name in ("base", "grpo", "micro"):
        r = next(x for x in res[name][split]["results"] if x["task_id"] == task_id)
        print(f"--- {name.upper()}  {'✅ passed' if r['passed'] else '❌ failed'}")
        print(r["completion"].strip())
        if not r["passed"]:
            _, vr, _ = rubric.score(r["completion"], t)   # re-run the tests to show why it failed
            print("\n" + vr.feedback())
        print()

N_EXAMPLES = 3
for split in ("mbpp", "humaneval"):
    grpo_ok  = set(res["grpo"][split]["passed"])
    micro_ok = set(res["micro"][split]["passed"])
    disagree = sorted(micro_ok - grpo_ok) + sorted(grpo_ok - micro_ok)   # Micro wins listed first
    print(f"\n##### {split}: {len(disagree)} problems where GRPO and MicroCoder disagree")
    # if they agree everywhere, still show one problem so there is code on screen
    for tid in (disagree or [res["micro"][split]["results"][0]["task_id"]])[:N_EXAMPLES]:
        show_example(split, tid)
"""))

cells.append(md(r"""
## 8 · What this refactor buys

**As an artifact** this is now a *code-RL environment*, not a training script:

- **Decoupled** — the env deals in text and never imports torch. GRPO is one client; eval is
  another. Any algorithm can consume the same `reset()/step()` surface.
- **Verifiable & testable** — the reward source is unit-tested without a GPU (Section 5). The
  environment's correctness is established independently of training.
- **Multi-turn** — write → run tests → read the traceback → revise. The `mean turns to
  terminate` curve shows how often the model needs a second attempt, and whether training
  teaches it to self-correct.
- **Honest measurement** — reporting in-distribution (MBPP) *and* transfer (HumanEval)
  separates "did RL learn the trained distribution?" from "did it generalise?". The single
  transfer number alone is what made the original result look like pure regression.

**Limitations carried over:** 1.5B policy, 200 steps, MBPP's 3-test reward is coarse, single
seed. The point of this notebook is the *environment abstraction and the honest evaluation*,
not a leaderboard number.

**Natural next steps:** DAPO-style dynamic sampling (resample zero-variance groups), a denser
rubric (`dense_rubric()` is ready), a tool-use turn (let the model call the interpreter
itself), and wrapping `CodeEnv` in a thin `verifiers`-compatible adapter.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("grpo_rlvr_dapo_code.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote grpo_rlvr_dapo_code.ipynb with {len(cells)} cells")
