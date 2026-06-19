"""Config-driven GRPO / RFT training on the production stack.

    python src/train.py --config configs/grpo_baseline.yaml
    python src/train.py --config configs/microcoder_grpo.yaml

One entrypoint, two runs, selected entirely by YAML — the baseline and the
MicroCoder-GRPO variant differ only in config fields. Stack: TRL GRPOTrainer
(objective + clipping + KL), PEFT LoRA (parameter-efficient policy), vLLM
(fast rollouts), wandb (tracking). Designed to fit one A100 in colocate mode.
"""

from __future__ import annotations

import argparse

import yaml
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from data import load_mbpp_splits
from rewards import make_execution_reward
from callbacks import TwoStageTemperatureCallback


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_grpo_config(cfg: dict, run_name: str) -> GRPOConfig:
    # The YAML keys under `grpo:` are named to match GRPOConfig fields exactly,
    # so they pass straight through. Keep them in sync if you bump TRL.
    return GRPOConfig(run_name=run_name, **cfg["grpo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--push_to_hub", default=None,
                    help="HF repo id to push the adapter + model card to.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- data ---
    train_ds, heldout_ds = load_mbpp_splits(**cfg["data"])
    print(f"train={len(train_ds)}  heldout={len(heldout_ds)}")

    # --- LoRA policy ---
    lc = cfg["lora"]
    peft_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        task_type="CAUSAL_LM",
    )

    # --- trainer ---
    grpo_config = build_grpo_config(cfg, run_name=cfg["run_name"])
    reward_fn = make_execution_reward()

    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    # --- Fix 2: two-stage temperature schedule (MicroCoder-GRPO only) ---
    ts = cfg.get("two_stage_temperature", {})
    if ts.get("enabled"):
        cb = TwoStageTemperatureCallback(
            switch_step=ts["switch_step"],
            temp_stage1=ts["temp_stage1"],
            temp_stage2=ts["temp_stage2"],
        )
        cb.trainer = trainer
        trainer.add_callback(cb)

    trainer.train()
    trainer.save_model(cfg["grpo"]["output_dir"])

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
