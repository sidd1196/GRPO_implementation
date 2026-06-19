"""Rubric — composes one or more weighted reward functions over a verification result.

This is the `verifiers`-style abstraction: a reward is not a single hard-coded number
but a weighted blend of named components. Swap in a denser rubric (syntax + format
bonuses) without touching the environment or the trainer. Each component is a callable
`(completion: str, vr: VerificationResult) -> float` returning a value in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .tasks import TaskSpec
from .verifier import ExecutionVerifier, VerificationResult

RewardFn = Callable[[str, VerificationResult], float]


# ── Reward components ──────────────────────────────────────────────────────────
def reward_tests_passed(completion: str, vr: VerificationResult) -> float:
    """Fraction of unit tests passed — the primary verifiable signal."""
    return vr.fraction_passed


def reward_no_syntax_error(completion: str, vr: VerificationResult) -> float:
    return 1.0 if vr.syntax_ok else 0.0


def reward_clean_format(completion: str, vr: VerificationResult) -> float:
    """Reward raw code; discourage markdown fences / chatter."""
    return 0.0 if "```" in completion else 1.0


@dataclass
class RewardComponent:
    name: str
    fn: RewardFn
    weight: float = 1.0


class Rubric:
    """Holds a verifier and a list of weighted reward components."""

    def __init__(self, verifier: Optional[ExecutionVerifier] = None,
                 components: Optional[List[RewardComponent]] = None):
        self.verifier = verifier or ExecutionVerifier()
        self.components = components or [RewardComponent("tests", reward_tests_passed, 1.0)]

    def score(self, completion: str, task: TaskSpec) -> Tuple[float, VerificationResult, Dict[str, float]]:
        """Returns (scalar_reward, verification_result, per-component breakdown)."""
        vr = self.verifier.verify(completion, task)
        breakdown: Dict[str, float] = {}
        total = wsum = 0.0
        for c in self.components:
            v = c.fn(completion, vr)
            breakdown[c.name] = v
            total += c.weight * v
            wsum += c.weight
        reward = total / wsum if wsum else 0.0
        return reward, vr, breakdown


def default_rubric() -> Rubric:
    """Just unit-test pass fraction (matches the original experiment)."""
    return Rubric()


def dense_rubric() -> Rubric:
    """Tests dominate, with small syntax + format shaping bonuses."""
    return Rubric(components=[
        RewardComponent("tests", reward_tests_passed, 0.8),
        RewardComponent("syntax", reward_no_syntax_error, 0.1),
        RewardComponent("format", reward_clean_format, 0.1),
    ])
