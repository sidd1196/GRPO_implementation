"""Verifier/rubric unit tests — run anywhere, no GPU, no model."""
from code_rl_env.rubric import dense_rubric
from code_rl_env.tasks import TaskSpec
from code_rl_env.verifier import ExecutionVerifier


def _double_task():
    return TaskSpec(
        task_id="t/double",
        prompt="Return x doubled.",
        tests=["assert f(2) == 4", "assert f(3) == 6", "assert f(0) == 0"],
        entry_point="f",
        source="test",
    )


def test_all_pass():
    vr = ExecutionVerifier().verify("def f(x):\n    return x * 2", _double_task())
    assert vr.syntax_ok
    assert vr.all_passed
    assert vr.fraction_passed == 1.0


def test_partial_credit():
    # x + 2 passes only f(2) == 4 ; fails f(3) and f(0)
    vr = ExecutionVerifier().verify("def f(x):\n    return x + 2", _double_task())
    assert vr.n_passed == 1
    assert abs(vr.fraction_passed - 1 / 3) < 1e-9
    assert not vr.all_passed


def test_syntax_error_scores_zero():
    vr = ExecutionVerifier().verify("def f(x) return x", _double_task())
    assert not vr.syntax_ok
    assert vr.fraction_passed == 0.0
    assert "SyntaxError" in vr.feedback() or "parse" in vr.feedback()


def test_fences_are_stripped():
    vr = ExecutionVerifier().verify("```python\ndef f(x):\n    return x * 2\n```", _double_task())
    assert vr.all_passed


def test_timeout_does_not_hang():
    task = TaskSpec("t/loop", "loops forever", ["assert f(1) == 1"], "f", "test")
    vr = ExecutionVerifier(timeout=2).verify("def f(x):\n    while True:\n        pass", task)
    assert not vr.all_passed  # killed by timeout, not a pass


def test_feedback_points_at_failure():
    vr = ExecutionVerifier().verify("def f(x):\n    return x + 2", _double_task())
    fb = vr.feedback()
    assert "failed this test" in fb.lower()


def test_dense_rubric_blends_components():
    rubric = dense_rubric()
    # fenced but fully-correct: tests=1.0*0.8, syntax=1.0*0.1, format=0.0*0.1 -> 0.9
    reward, vr, breakdown = rubric.score("```python\ndef f(x):\n    return x*2\n```", _double_task())
    assert vr.all_passed
    assert breakdown["format"] == 0.0
    assert abs(reward - 0.9) < 1e-9
