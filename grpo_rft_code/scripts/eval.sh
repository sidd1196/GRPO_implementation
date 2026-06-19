#!/usr/bin/env bash
# Evaluate base + both tuned policies on in-distribution (MBPP held-out) and
# transfer (HumanEval). Appends one JSON row per model to results.jsonl.
#   bash scripts/eval.sh
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
BASE="Qwen/Qwen2.5-Coder-1.5B-Instruct"

rm -f results.jsonl

python src/eval.py --model "$BASE"                              --label base
python src/eval.py --model outputs/grpo-baseline   --base "$BASE" --label grpo
python src/eval.py --model outputs/microcoder-grpo --base "$BASE" --label microcoder
