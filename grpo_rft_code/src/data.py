"""MBPP data pipeline for GRPO/RFT.

Produces a conversational prompt dataset (TRL applies the model's chat template
automatically) with the extra columns the reward function needs: `tests` and
`test_setup`. Also carves out an in-distribution held-out split so we can
measure whether RL improved the model on the task it actually trained on — the
honest counterpart to the HumanEval *transfer* number.
"""

from __future__ import annotations

from datasets import load_dataset, Dataset

from rewards import get_function_name

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a single, correct, self-contained "
    "Python function that solves the problem. Respond with only the function inside "
    "a ```python code block."
)


def _build_prompt(text: str, function_name: str | None) -> list[dict]:
    user = text.strip()
    if function_name:
        user += f"\n\nThe function must be named `{function_name}`."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _to_rows(split) -> list[dict]:
    rows = []
    for ex in split:
        tests = ex["test_list"]
        fn = get_function_name(tests)
        rows.append(
            {
                "prompt": _build_prompt(ex["text"], fn),
                "tests": tests,
                "test_setup": ex.get("test_setup_code", "") or "",
                "function_name": fn or "",
                "task_id": ex["task_id"],
            }
        )
    return rows


def load_mbpp_splits(n_train: int = 374, n_heldout: int = 100, seed: int = 0):
    """Return (train_ds, heldout_ds).

    `train` feeds GRPO; `heldout` is the in-distribution eval set (same MBPP
    distribution, never trained on). HumanEval is loaded separately in eval.py
    as the out-of-distribution transfer benchmark.
    """
    # MBPP "full" config: ~974 problems with text/code/test_list/test_setup_code.
    raw = load_dataset("google-research-datasets/mbpp", "full", split="train")
    raw = raw.shuffle(seed=seed)

    train_rows = _to_rows(raw.select(range(n_train)))
    heldout_rows = _to_rows(raw.select(range(n_train, n_train + n_heldout)))

    # Drop problems whose tests have no parseable function name — the reward
    # would be undefined for them.
    train_rows = [r for r in train_rows if r["function_name"]]
    heldout_rows = [r for r in heldout_rows if r["function_name"]]

    return Dataset.from_list(train_rows), Dataset.from_list(heldout_rows)
