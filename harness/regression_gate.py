"""Minimal-diff regression gate: accept a patch only if it is safe.

A patch is ACCEPTED only when:
  1. the previously failing test now passes, and
  2. the capped regression suite stays green (abort-on-first-failure).

Otherwise the patch is rolled back and REJECTED. Diff size is measured
with difflib (lines added + lines removed, files changed) so "minimal"
is a number, not a feeling.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from patcher_stub import FileEdit, apply_edits, rollback_edits
from test_runner import (
    DEFAULT_MAX_REGRESSION_TESTS,
    DEFAULT_REGRESSION_TIMEOUT_SECS,
    RegressionReport,
    run_one_test,
    run_regression_suite,
)


@dataclass
class GateResult:
    accepted: bool
    reason: str
    failing_test_passed: bool
    regression: RegressionReport
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: int = 0

    @property
    def diff_size(self) -> int:
        return self.lines_added + self.lines_removed


def measure_diff(originals: dict[str, str], repo_root: str) -> tuple[int, int, int]:
    """Return (lines_added, lines_removed, files_changed) vs originals."""
    import os

    added = removed = 0
    for path, original in originals.items():
        with open(os.path.join(repo_root, path), encoding="utf-8") as fh:
            new = fh.read()
        diff = difflib.unified_diff(
            original.splitlines(), new.splitlines(), lineterm=""
        )
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
    return added, removed, len(originals)


def evaluate_patch(
    repo_root: str,
    edits: list[FileEdit],
    failing_test_id: str,
    regression_test_ids: list[str],
    max_regression_tests: int = DEFAULT_MAX_REGRESSION_TESTS,
    regression_timeout_secs: int = DEFAULT_REGRESSION_TIMEOUT_SECS,
) -> GateResult:
    """Apply a patch, test it, keep it only if it passes the gate."""
    try:
        originals = apply_edits(repo_root, edits)
    except ValueError as exc:
        return GateResult(
            accepted=False,
            reason=f"patch could not be applied: {exc}",
            failing_test_passed=False,
            regression=RegressionReport(),
        )

    failing = run_one_test(failing_test_id, repo_root)
    if not failing.passed:
        rollback_edits(repo_root, originals)
        return GateResult(
            accepted=False,
            reason="failing test still fails after patch",
            failing_test_passed=False,
            regression=RegressionReport(),
        )

    regression = run_regression_suite(
        regression_test_ids,
        repo_root,
        max_tests=max_regression_tests,
        timeout_secs=regression_timeout_secs,
    )
    if not regression.all_passed:
        rollback_edits(repo_root, originals)
        return GateResult(
            accepted=False,
            reason=f"regression failure: {regression.failed_tests}",
            failing_test_passed=True,
            regression=regression,
        )

    added, removed, files_changed = measure_diff(originals, repo_root)
    return GateResult(
        accepted=True,
        reason="failing test passes, regression suite green",
        failing_test_passed=True,
        regression=regression,
        lines_added=added,
        lines_removed=removed,
        files_changed=files_changed,
    )
