"""CodeEnv — a Gym-style, multi-turn RL environment for code generation.

Interface:
    obs  = env.reset(task)          # -> Observation (the instruction to give the model)
    step = env.step(completion)     # -> StepResult(observation, reward, done, info)

The environment is model-agnostic: it deals in *text* (prompts in, completions out) and
never imports torch/transformers. The policy/trainer is responsible for chat-templating
and generation. Reward is delegated entirely to a `Rubric`, so the env knows nothing
about *how* reward is computed — that is the decoupling that lets any algorithm (GRPO,
best-of-n, plain eval) consume the same environment.

Episode dynamics (multi-turn):
    turn 0: model sees the problem, writes a function.
    on failure (turn < max_turns): model sees its previous attempt + the failing test /
        traceback, and revises.
    done when all tests pass, or max_turns is reached.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .rubric import Rubric
from .tasks import TaskSpec

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write ONLY the complete Python function "
    "(including its `def` line). No markdown, no code fences, no explanations."
)


@dataclass
class Observation:
    task_id: str
    prompt_text: str            # user-message content for this turn ("" once done)
    turn: int
    feedback: Optional[str] = None
    done: bool = False


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict


class CodeEnv:
    def __init__(self, tasks: List[TaskSpec], rubric: Optional[Rubric] = None,
                 max_turns: int = 3, rng: Optional[random.Random] = None):
        if not tasks:
            raise ValueError("CodeEnv needs at least one task")
        self.tasks = tasks
        self.rubric = rubric or Rubric()
        self.max_turns = max_turns
        self.rng = rng or random.Random(0)
        self.system_prompt = SYSTEM_PROMPT
        self._task: Optional[TaskSpec] = None
        self._turn = 0
        self._history: List[Tuple[str, str]] = []  # (action, feedback) per past turn

    # ── prompt construction ────────────────────────────────────────────────────
    def _build_prompt(self, feedback: Optional[str] = None) -> str:
        t = self._task
        instr = (f"Solve this Python task. The function must be named `{t.entry_point}`.\n\n"
                 f"{t.prompt}")
        if feedback:
            last_action = self._history[-1][0]
            instr += (f"\n\nYour previous attempt was incorrect:\n\n{last_action}\n\n"
                      f"{feedback}\n\nReturn a corrected, complete function.")
        return instr

    # ── gym API ────────────────────────────────────────────────────────────────
    def reset(self, task: Optional[TaskSpec] = None) -> Observation:
        self._task = task or self.rng.choice(self.tasks)
        self._turn = 0
        self._history = []
        return Observation(self._task.task_id, self._build_prompt(), turn=0)

    def step(self, action: str) -> StepResult:
        if self._task is None:
            raise RuntimeError("call reset() before step()")

        reward, vr, breakdown = self.rubric.score(action, self._task)
        feedback = vr.feedback()
        self._history.append((action, feedback))
        self._turn += 1

        solved = vr.all_passed
        done = solved or self._turn >= self.max_turns

        next_obs = Observation(
            task_id=self._task.task_id,
            prompt_text="" if done else self._build_prompt(feedback),
            turn=self._turn,
            feedback=None if done else feedback,
            done=done,
        )
        info = {
            "breakdown": breakdown,
            "n_passed": vr.n_passed,
            "n_total": vr.n_total,
            "solved": solved,
            "syntax_ok": vr.syntax_ok,
        }
        return StepResult(next_obs, reward, done, info)
