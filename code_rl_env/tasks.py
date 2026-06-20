"""Task specifications and dataset loaders.

A `TaskSpec` is the unit of work the environment serves: a problem statement, the
required entry-point function name, and a list of executable test snippets. MBPP and
HumanEval are both normalised into this single shape so the *same* environment, verifier
and eval harness serve in-distribution (MBPP) and transfer (HumanEval) tasks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskSpec:
    task_id: str
    prompt: str              # NL problem (MBPP) or signature+docstring (HumanEval)
    tests: List[str]         # executable snippets; each is run as `code + "\n\n" + test`
    entry_point: str         # the function name the tests call
    source: str              # "mbpp" | "humaneval" | ...
    meta: dict = field(default_factory=dict)


def _entry_from_asserts(tests: List[str]) -> str:
    """MBPP tests look like `assert fibonacci(10) == 55` -> 'fibonacci'."""
    for t in tests:
        m = re.search(r"\bassert\s+(\w+)\s*\(", t)
        if m:
            return m.group(1)
    return ""


def load_mbpp(split: str = "train+validation+test", limit: Optional[int] = None) -> List[TaskSpec]:
    from datasets import load_dataset

    # The sanitized config's `train` split has only 120 problems — too few for a
    # 150-train / 30-eval split. Concatenating train+validation+test yields ~420
    # tasks (all carry the `prompt` field), so downstream slicing has room.
    raw = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    tasks: List[TaskSpec] = []
    for i in range(len(raw)):
        tests = list(raw[i]["test_list"])
        tasks.append(TaskSpec(
            task_id=f"mbpp/{raw[i]['task_id']}",
            prompt=raw[i]["prompt"],
            tests=tests,
            entry_point=_entry_from_asserts(tests),
            source="mbpp",
        ))
    return tasks[:limit] if limit else tasks


def load_humaneval(limit: Optional[int] = None) -> List[TaskSpec]:
    from datasets import load_dataset

    raw = load_dataset("openai/openai_humaneval", split="test")
    tasks: List[TaskSpec] = []
    for i in range(len(raw)):
        entry = raw[i]["entry_point"]
        # The model writes the FULL function (incl. signature), so the verifier runs the
        # completion standalone against the check harness — no fragile prompt+body splicing.
        test = raw[i]["test"] + f"\n\ncheck({entry})"
        tasks.append(TaskSpec(
            task_id=raw[i]["task_id"],
            prompt=raw[i]["prompt"],
            tests=[test],
            entry_point=entry,
            source="humaneval",
            meta={"signature_prompt": raw[i]["prompt"]},
        ))
    return tasks[:limit] if limit else tasks
