"""Verifiable execution reward for GRPO/RFT on code.

The reward signal is the Python interpreter — no reward model, no human labels.
Each MBPP problem ships a list of `assert` test cases; we run the model's code
against them in an isolated subprocess and return the *fraction* that pass.

Partial credit (0.33, 0.67, 1.0) is deliberate: it creates reward variance
inside a G-completion group even when no single completion passes everything,
which is what gives GRPO a usable advantage signal early in training.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import os
from typing import List

# Match the function name in an assertion like `assert fibonacci(10) == 55`.
_FUNC_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")
# Pull code out of a ```python ... ``` fence if the model emitted one.
_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def get_function_name(tests: List[str]) -> str | None:
    """Parse the expected function name from the test assertions.

    MBPP tests call a specific name (`assert similar_elements(...) == ...`) but
    the natural-language prompt never states it. Without this the model invents
    a name, the asserts raise NameError, and every reward is 0 — the silent bug
    that produced 200 dead training steps in the first version of this project.
    """
    for t in tests:
        m = _FUNC_RE.search(t)
        if m:
            return m.group(1)
    return None


def extract_code(text: str) -> str:
    """Strip a markdown fence if present; otherwise return the raw text."""
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _completion_to_text(completion) -> str:
    """TRL hands back either a raw string (standard prompts) or a list of
    chat messages (conversational prompts). Normalise to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return str(completion)


def run_tests(code: str, tests: List[str], setup: str = "", timeout: int = 10) -> float:
    """Run `code` against each assert in `tests`, return fraction passed.

    Each test runs in its own subprocess so an infinite loop or a crash in one
    completion can't take down training. Returns a float in [0, 1].
    """
    if not tests:
        return 0.0

    passed = 0
    for test in tests:
        program = f"{setup}\n{code}\n{test}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(program)
            path = fh.name
        try:
            res = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                timeout=timeout,
            )
            if res.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass  # treat timeout as a failed test, keep going
        finally:
            os.unlink(path)
    return passed / len(tests)


def make_execution_reward(timeout: int = 10):
    """Build a TRL-compatible reward function.

    TRL calls `reward_func(prompts, completions, **kwargs)` where every extra
    dataset column (here `tests`, `test_setup`) arrives as a batch-aligned list.
    Must return a list[float] of length == len(completions).
    """

    def execution_reward(completions, tests, test_setup=None, **kwargs) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            code = extract_code(_completion_to_text(completion))
            setup = (test_setup[i] if test_setup else "") or ""
            rewards.append(run_tests(code, tests[i], setup=setup, timeout=timeout))
        return rewards

    # TRL uses __name__ for the per-reward wandb metric (reward/<name>/mean).
    execution_reward.__name__ = "execution_reward"
    return execution_reward
