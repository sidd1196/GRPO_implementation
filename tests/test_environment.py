"""CodeEnv rollout-protocol tests — no GPU. A 'stub policy' is just hand-written code."""
from code_rl_env.environment import CodeEnv
from code_rl_env.tasks import TaskSpec


def _task():
    return TaskSpec(
        task_id="t/double",
        prompt="Return x doubled.",
        tests=["assert f(2) == 4", "assert f(3) == 6"],
        entry_point="f",
        source="test",
    )


def test_reset_returns_prompt_with_fn_name():
    env = CodeEnv([_task()], max_turns=3)
    obs = env.reset(_task())
    assert obs.turn == 0 and not obs.done
    assert "`f`" in obs.prompt_text


def test_single_turn_solve_is_done():
    env = CodeEnv([_task()], max_turns=3)
    env.reset(_task())
    sr = env.step("def f(x):\n    return x * 2")
    assert sr.reward == 1.0 and sr.done and sr.info["solved"]


def test_multi_turn_revision_loop():
    env = CodeEnv([_task()], max_turns=3)
    env.reset(_task())

    sr1 = env.step("def f(x):\n    return x + 2")        # wrong
    assert not sr1.done
    assert sr1.reward < 1.0
    assert sr1.observation.feedback
    # next prompt must carry the previous attempt + feedback for revision
    assert "previous attempt" in sr1.observation.prompt_text.lower()

    sr2 = env.step("def f(x):\n    return x * 2")         # corrected
    assert sr2.done and sr2.reward == 1.0 and sr2.info["solved"]


def test_max_turns_terminates_even_if_unsolved():
    env = CodeEnv([_task()], max_turns=2)
    env.reset(_task())
    env.step("def f(x):\n    return 0")
    sr = env.step("def f(x):\n    return 0")
    assert sr.done and not sr.info["solved"]


def test_step_before_reset_raises():
    env = CodeEnv([_task()], max_turns=2)
    try:
        env.step("def f(x): return x")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
