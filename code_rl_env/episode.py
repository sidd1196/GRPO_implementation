"""Trajectory data structures — pure-Python, no ML deps.

A `Trajectory` is one rollout: a sequence of `Turn`s (attempt -> reward -> feedback).
The trainer attaches token tensors separately; these dataclasses stay framework-free so
they can be inspected and tested without torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Turn:
    prompt_text: str          # the instruction shown to the model this turn
    action: str               # the model's completion
    reward: float
    feedback: str             # verifier feedback used to build the next turn ("" if solved/last)
    breakdown: Dict[str, float] = field(default_factory=dict)
    info: Dict = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    turns: List[Turn] = field(default_factory=list)

    @property
    def final_reward(self) -> float:
        return self.turns[-1].reward if self.turns else 0.0

    @property
    def best_reward(self) -> float:
        return max((t.reward for t in self.turns), default=0.0)

    @property
    def solved(self) -> bool:
        return any(t.info.get("solved") for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)
