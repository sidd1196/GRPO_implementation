"""Evaluate a base or RL-tuned policy on two axes:

  1. In-distribution  — MBPP held-out split (same distribution as training)
  2. Transfer         — HumanEval (a different code benchmark, never trained on)

Reporting both is the point. RLVR on a narrow MBPP slice can lift the
in-distribution number while *regressing* on transfer — and only showing the
transfer number (as the first version of this project did) turns a real,
defensible result into a misleading "RL made it worse" headline.

    python src/eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct          # base
    python src/eval.py --model outputs/microcoder-grpo --base Qwen/Qwen2.5-Coder-1.5B-Instruct

Greedy decoding, pass@1. Writes a JSON line per model to results.jsonl so the
analysis notebook can build the paired per-problem breakdown.
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_mbpp_splits, SYSTEM_PROMPT
from rewards import extract_code, run_tests, get_function_name


def load_policy(model_path: str, base: str | None):
    """Load a plain model, or a LoRA adapter merged onto its base."""
    base_id = base or model_path
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if base:  # model_path is a LoRA adapter dir
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, messages, max_new_tokens=1024) -> str:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_mbpp_heldout(model, tok, heldout) -> dict:
    passed = []
    for ex in heldout:
        text = generate(model, tok, ex["prompt"])
        score = run_tests(extract_code(text), ex["tests"], setup=ex["test_setup"])
        if score == 1.0:
            passed.append(ex["task_id"])
    return {"pass@1": len(passed) / len(heldout), "n": len(heldout), "passed": passed}


def eval_humaneval(model, tok) -> dict:
    """Transfer benchmark. Each problem ships a prompt (function signature +
    docstring), a canonical `test` function, and an `entry_point` name."""
    ds = load_dataset("openai_humaneval", split="test")
    passed = []
    for ex in ds:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]
                + f"\n\nThe function must be named `{ex['entry_point']}`."},
        ]
        code = extract_code(generate(model, tok, messages))
        # HumanEval's check() is invoked by appending its test block + a call.
        program = f"{code}\n{ex['test']}\ncheck({ex['entry_point']})\n"
        if run_tests("", [program]) == 1.0:  # run the whole assembled program once
            passed.append(ex["task_id"])
    return {"pass@1": len(passed) / len(ds), "n": len(ds), "passed": passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id or LoRA adapter dir")
    ap.add_argument("--base", default=None, help="base model id if --model is an adapter")
    ap.add_argument("--label", default=None, help="name for this row in results.jsonl")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    model, tok = load_policy(args.model, args.base)
    _, heldout = load_mbpp_splits()

    row = {
        "label": args.label or args.model,
        "mbpp_heldout": eval_mbpp_heldout(model, tok, heldout),
        "humaneval_transfer": eval_humaneval(model, tok),
    }
    print(f"{row['label']:<22} "
          f"MBPP held-out pass@1={row['mbpp_heldout']['pass@1']:.3f}  "
          f"HumanEval transfer pass@1={row['humaneval_transfer']['pass@1']:.3f}")

    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
