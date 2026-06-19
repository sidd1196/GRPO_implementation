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
print("package installed")
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
"""))

cells.append(code(r"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from code_rl_env import load_mbpp, load_humaneval, default_rubric
from train_grpo import GRPOConfig, run_grpo, evaluate

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
"""))

cells.append(md(r"""
### 6a · Baseline measurement — report BOTH in-distribution and transfer

The original experiment only measured HumanEval (transfer), which made RL look like it *hurt*.
We now also measure held-out MBPP (in-distribution) — where RL is trained — so the
"where does RL actually help?" question gets an honest answer.
"""))

cells.append(code(r"""
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")
base_mbpp = evaluate(base, tokenizer, eval_mbpp, rubric)["pass@1"]
base_he   = evaluate(base, tokenizer, humaneval, rubric)["pass@1"]
print(f"BASE   MBPP(in-dist)={base_mbpp:.3f}   HumanEval(transfer)={base_he:.3f}")
del base; torch.cuda.empty_cache()
"""))

cells.append(md("### 6b · GRPO baseline (KL + symmetric clip)"))

cells.append(code(r"""
policy = fresh_policy()
ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")
for p in ref.parameters():
    p.requires_grad_(False)

grpo_cfg = GRPOConfig(microcoder=False, num_steps=200, G=4, max_turns=3,
                      kl_coeff=0.01, epsilon_low=0.2, epsilon_high=0.2, temperature=0.8)
grpo_log = run_grpo(policy, tokenizer, train_tasks, grpo_cfg,
                    ref_model=ref, rubric=rubric, eval_tasks=eval_mbpp)

grpo_mbpp = evaluate(policy, tokenizer, eval_mbpp, rubric)["pass@1"]
grpo_he   = evaluate(policy, tokenizer, humaneval, rubric)["pass@1"]
print(f"GRPO   MBPP(in-dist)={grpo_mbpp:.3f}   HumanEval(transfer)={grpo_he:.3f}")
del policy, ref; torch.cuda.empty_cache()
"""))

cells.append(md("### 6c · MicroCoder-GRPO (3 code-specific fixes, no KL)"))

cells.append(code(r"""
mc_policy = fresh_policy()
mc_cfg = GRPOConfig(microcoder=True, num_steps=200, G=4, max_turns=3,
                    kl_coeff=0.0, epsilon_low=0.2, epsilon_high=0.5,
                    temp_stage1=0.7, temp_stage2=1.0, temp_switch_step=100,
                    mask_prob=0.3, repeat_check_len=128)
mc_log = run_grpo(mc_policy, tokenizer, train_tasks, mc_cfg,
                  ref_model=None, rubric=rubric, eval_tasks=eval_mbpp)

mc_mbpp = evaluate(mc_policy, tokenizer, eval_mbpp, rubric)["pass@1"]
mc_he   = evaluate(mc_policy, tokenizer, humaneval, rubric)["pass@1"]
print(f"MicroCoder-GRPO   MBPP(in-dist)={mc_mbpp:.3f}   HumanEval(transfer)={mc_he:.3f}")
"""))

cells.append(md("## 7 · Results — in-distribution vs transfer"))

cells.append(code(r"""
print(f"{'Model':<20}{'MBPP (in-dist)':>16}{'HumanEval (transfer)':>22}")
print("-" * 58)
for name, m, h in [("Base", base_mbpp, base_he),
                   ("GRPO", grpo_mbpp, grpo_he),
                   ("MicroCoder-GRPO", mc_mbpp, mc_he)]:
    print(f"{name:<20}{m:>16.3f}{h:>22.3f}")
"""))

cells.append(code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(grpo_log["rewards"], label="GRPO", alpha=.8)
ax[0].plot(mc_log["rewards"], label="MicroCoder", alpha=.8)
ax[0].set_title("mean group reward"); ax[0].set_xlabel("step"); ax[0].legend()

ax[1].plot(grpo_log["solve_rate"], label="GRPO", alpha=.8)
ax[1].plot(mc_log["solve_rate"], label="MicroCoder", alpha=.8)
ax[1].set_title("group solve rate (any turn)"); ax[1].set_xlabel("step"); ax[1].legend()

ax[2].plot(grpo_log["mean_turns"], label="GRPO", alpha=.8)
ax[2].plot(mc_log["mean_turns"], label="MicroCoder", alpha=.8)
ax[2].set_title("mean turns to terminate"); ax[2].set_xlabel("step"); ax[2].legend()
plt.tight_layout(); plt.show()
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
