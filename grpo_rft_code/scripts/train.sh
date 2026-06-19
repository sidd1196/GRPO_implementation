#!/usr/bin/env bash
# Train both runs from the same base checkpoint. Run from the repo root.
#   bash scripts/train.sh
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
export WANDB_PROJECT="grpo-rft-code"

python src/train.py --config configs/grpo_baseline.yaml
python src/train.py --config configs/microcoder_grpo.yaml
