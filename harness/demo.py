"""Runnable demo: run the scope ladder against a copy of demo_repo/.

The demo patch function stands in for the future Gemma LoRA: on rung 1
it proposes the one-line fix for ``calculator.add``. The ladder + gate
do the rest — trace, pack, apply, verify, accept.

Usage (from harness/):
    python3 demo.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patcher_stub import FileEdit  # noqa: E402
from scope_ladder import run_ladder  # noqa: E402
from test_runner import run_one_test  # noqa: E402
from trace_extractor import extract_trace  # noqa: E402

FAILING_TEST = "tests.test_calculator.TestCalculator.test_add"
REGRESSION_TESTS = [
    "tests.test_calculator.TestCalculator.test_multiply",
    "tests.test_calculator.TestCalculator.test_subtract",
]
ISSUE_TEXT = "add(2, 3) returns -1 instead of 5. The `add` function in calculator.py is wrong."


def demo_patch_fn(context_text: str, rung: int):
    """Stand-in for the Gemma LoRA. Only proposes a patch on rung 1."""
    if rung == 1 and "calculator.py" in context_text:
        # Unambiguous: "return a - b" alone also appears in subtract().
        return [FileEdit("calculator.py", "def add(a, b):\n    return a - b",
                         "def add(a, b):\n    return a + b")]
    return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(prefix="repo-blind-demo-") as tmp:
        repo = os.path.join(tmp, "demo_repo")
        shutil.copytree(os.path.join(here, "demo_repo"), repo)

        failing = run_one_test(FAILING_TEST, repo)
        print(f"Before: failing test passed={failing.passed}")
        trace = extract_trace(failing.output, ISSUE_TEXT)
        print(f"Trace: failing={trace.failing_tests}")
        print(f"       files={trace.implicated_files}")
        print(f"       symbols={trace.implicated_symbols}")

        result = run_ladder(repo, trace, FAILING_TEST, REGRESSION_TESTS, patch_fn=demo_patch_fn)
        print(f"Rung files: {result.rung_files}")
        print(f"Rung secs: {result.rung_secs}")
        print(f"Accepted: {result.accepted} at rung {result.rung} — {result.reason}")
        if result.gate is not None:
            print(
                f"Gate: diff +{result.gate.lines_added}/-{result.gate.lines_removed} "
                f"lines, {result.gate.files_changed} file(s), "
                f"regression tests run: {[r.test_id for r in result.gate.regression.results]}"
            )
        after = run_one_test(FAILING_TEST, repo)
        print(f"After: failing test passed={after.passed}")
        return 0 if result.accepted and after.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
