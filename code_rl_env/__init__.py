"""code_rl_env — a decoupled, verifiable, multi-turn RL environment for code generation.

Layers (each importable without a GPU):
    tasks       TaskSpec + MBPP/HumanEval loaders
    sandbox     subprocess execution with timeout
    verifier    ExecutionVerifier -> per-test pass/fail + error
    rubric      Rubric -> weighted blend of named reward functions
    episode     Turn / Trajectory dataclasses
    environment CodeEnv -> Gym-style reset()/step(), multi-turn

The trainer (see train_grpo.py at repo root) is just one client of CodeEnv.
"""
from .environment import CodeEnv, Observation, StepResult, SYSTEM_PROMPT
from .episode import Trajectory, Turn
from .rubric import (
    Rubric,
    RewardComponent,
    default_rubric,
    dense_rubric,
    reward_clean_format,
    reward_no_syntax_error,
    reward_tests_passed,
)
from .sandbox import ExecResult, has_syntax_error, run_code, strip_fences
from .tasks import TaskSpec, load_humaneval, load_mbpp
from .verifier import ExecutionVerifier, TestResult, VerificationResult

__all__ = [
    "CodeEnv", "Observation", "StepResult", "SYSTEM_PROMPT",
    "Trajectory", "Turn",
    "Rubric", "RewardComponent", "default_rubric", "dense_rubric",
    "reward_tests_passed", "reward_no_syntax_error", "reward_clean_format",
    "ExecResult", "run_code", "strip_fences", "has_syntax_error",
    "TaskSpec", "load_mbpp", "load_humaneval",
    "ExecutionVerifier", "TestResult", "VerificationResult",
]
