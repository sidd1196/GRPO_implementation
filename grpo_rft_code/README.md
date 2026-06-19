# GRPO / RFT for Code — Production-Stack Reinforcement Fine-Tuning

Reinforcement fine-tuning of a code model with **verifiable execution rewards**,
built on the standard post-training stack: **TRL `GRPOTrainer`**, **PEFT LoRA**,
**vLLM** rollouts, **wandb** tracking, run end-to-end from **YAML config**. Fits
a single A100.

It trains two policies from the same base checkpoint — a **standard GRPO
baseline** and a **MicroCoder-GRPO** variant with three code-specific stability
fixes — and evaluates both on two axes: in-distribution (MBPP held-out) and
transfer (HumanEval).

> Companion to a from-scratch GRPO implementation (`../grpo_rlvr_dapo_code.ipynb`)
> where the same algorithm and fixes are written by hand. This repo is the
> production-tooling counterpart: same ideas, real stack, reproducible.

## The reward is the interpreter

No reward model, no human labels. Each MBPP problem ships `assert` test cases;
the model's generated function is run against them in an isolated subprocess and
the reward is the **fraction of tests passed**. Partial credit creates reward
variance inside each group of `G` completions, which is what gives GRPO a usable
advantage signal early in training. See `src/rewards.py`.

## MicroCoder-GRPO fixes → production stack

| Fix | Mechanism in this repo |
|---|---|
| **No-KL + high clip** (Fix 3) | `beta=0.0`, `loss_type="dapo"`, `epsilon_high=0.5` (native TRL) |
| **Truncation masking** (Fix 1) | `mask_truncated_completions=true` (native TRL) |
| **Two-stage temperature** (Fix 2) | custom `TrainerCallback` (`src/callbacks.py`) flips temperature 0.7→1.0 at the switch step |

The baseline and the variant differ *only* in config fields
(`configs/grpo_baseline.yaml` vs `configs/microcoder_grpo.yaml`), so any
measured delta is attributable to the fixes.

## Why two eval axes

RLVR on a narrow MBPP slice can lift the **in-distribution** number while
**regressing on transfer** to a different benchmark. Reporting only the transfer
number turns a real, defensible finding into a misleading "RL made it worse"
headline. `src/eval.py` reports both and records per-problem pass/fail so the
analysis can show exactly which problems each method fixed or broke.

## Run it

```bash
pip install -r requirements.txt
wandb login

bash scripts/train.sh     # trains baseline, then MicroCoder-GRPO
bash scripts/eval.sh      # writes results.jsonl (base + both policies)
```

## Repo layout

```
configs/   grpo_baseline.yaml, microcoder_grpo.yaml   # the only thing that changes between runs
src/
  rewards.py     verifiable execution reward + function-name parsing
  data.py        MBPP pipeline + in-distribution held-out split
  callbacks.py   two-stage temperature schedule (Fix 2)
  train.py       config-driven TRL GRPOTrainer + LoRA + vLLM + wandb
  eval.py        in-distribution (MBPP) + transfer (HumanEval) pass@1
scripts/   train.sh, eval.sh
```

## Scaling note (single GPU → cluster)

This runs on one A100 via vLLM `colocate` mode. The same `train.py` scales to
multi-GPU with `accelerate launch` + DeepSpeed ZeRO-3 and vLLM `server` mode for
the rollout engine — no code changes, only launch config. (Not run here; stated
honestly rather than faked.)

## Results

Filled in after the run — table of base / GRPO / MicroCoder-GRPO on MBPP
held-out (in-distribution) and HumanEval (transfer) pass@1, plus the paired
per-problem breakdown.
```
Model            MBPP held-out pass@1    HumanEval transfer pass@1
base             ...                     ...
grpo             ...                     ...
microcoder       ...                     ...
```
```
