"""Sandboxed execution primitives.

Pure-Python, no ML deps — this module (and everything it backs) is importable and
testable without a GPU. Code is run in a separate `python3 -c` subprocess with a
hard timeout so a buggy or malicious completion can't hang or corrupt the trainer.
"""
from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


_FENCE_RE = re.compile(r"```(?:python)?\s*\n?|```")


def strip_fences(code: str) -> str:
    """Remove markdown code fences a chat model often wraps code in."""
    return _FENCE_RE.sub("", code).strip()


def has_syntax_error(code: str) -> bool:
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True


def run_code(code: str, timeout: int = 5) -> ExecResult:
    """Execute `code` in a fresh subprocess. Never raises — failures become ExecResult."""
    try:
        r = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecResult(ok=(r.returncode == 0), returncode=r.returncode,
                          stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, returncode=-1, stdout="",
                          stderr=f"TimeoutExpired after {timeout}s", timed_out=True)
    except Exception as e:  # pragma: no cover - defensive
        return ExecResult(ok=False, returncode=-1, stdout="", stderr=repr(e))
