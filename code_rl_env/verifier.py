"""Verifier — the *reward source*, fully decoupled from any training algorithm.

`ExecutionVerifier.verify` runs each test independently and returns a structured
result (per-test pass/fail + error text). Partial credit (fraction of tests passed)
gives a dense signal even when no completion fully solves a task, and the captured
error feeds the environment's multi-turn revision feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .sandbox import has_syntax_error, run_code, strip_fences
from .tasks import TaskSpec


@dataclass
class TestResult:
    test: str
    passed: bool
    error: str = ""


@dataclass
class VerificationResult:
    code: str                      # the fence-stripped code that was executed
    syntax_ok: bool
    test_results: List[TestResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(t.passed for t in self.test_results)

    @property
    def n_total(self) -> int:
        return len(self.test_results)

    @property
    def fraction_passed(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0

    @property
    def all_passed(self) -> bool:
        return self.n_total > 0 and self.n_passed == self.n_total

    def feedback(self) -> str:
        """Human-readable hint for the next turn: first failure + its error."""
        if not self.syntax_ok:
            return "Your code failed to parse (SyntaxError). Return a syntactically valid function."
        for t in self.test_results:
            if not t.passed:
                err = t.error.strip()
                err = err[-400:] if err else "assertion failed"
                return f"Your code failed this test:\n{t.test}\nError:\n{err}"
        return ""


class ExecutionVerifier:
    """Runs each test as `code + test` in a sandboxed subprocess."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def verify(self, completion: str, task: TaskSpec) -> VerificationResult:
        code = strip_fences(completion)

        if has_syntax_error(code):
            return VerificationResult(
                code=code, syntax_ok=False,
                test_results=[TestResult(t, False, "SyntaxError") for t in task.tests],
            )

        results: List[TestResult] = []
        for test in task.tests:
            r = run_code(code + "\n\n" + test, timeout=self.timeout)
            results.append(TestResult(
                test=test,
                passed=r.ok,
                error="" if r.ok else (r.stderr or "non-zero exit"),
            ))
        return VerificationResult(code=code, syntax_ok=True, test_results=results)
