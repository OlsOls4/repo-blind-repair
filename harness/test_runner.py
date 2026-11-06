"""Test runner with hard timeouts and a capped regression suite.

Uses the stdlib ``unittest`` runner only (``python -m unittest``), so the
harness has no third-party dependency and works on any Python 3.12 box.
Test ids are unittest dotted ids, e.g. ``tests.test_calculator.TestAdd.test_add``.

Regression discipline (from the Repo-Blind Repair spec):
  * at most ``max_regression_tests`` tests (default 15),
  * total regression wall-clock capped at ``regression_timeout_secs``
    (default 90; the spec allows 60-90),
  * abort on the FIRST regression failure — a patch that breaks a
    previously passing test is rejected without running the rest.

Each test is run as its own subprocess so a hanging test cannot hang the
harness: the per-test timeout kills it and it counts as failed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

DEFAULT_MAX_REGRESSION_TESTS = 15
DEFAULT_REGRESSION_TIMEOUT_SECS = 90
DEFAULT_PER_TEST_TIMEOUT_SECS = 30


@dataclass
class TestResult:
    test_id: str
    passed: bool
    duration_secs: float
    output: str = ""


@dataclass
class RegressionReport:
    results: list[TestResult] = field(default_factory=list)
    aborted_early: bool = False
    timed_out: bool = False
    total_secs: float = 0.0

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failed_tests(self) -> list[str]:
        return [r.test_id for r in self.results if not r.passed]


def run_one_test(
    test_id: str,
    cwd: str,
    timeout_secs: int = DEFAULT_PER_TEST_TIMEOUT_SECS,
) -> TestResult:
    """Run a single unittest id in a subprocess with a hard timeout.

    Stale-bytecode guard (found by the search-baseline run, 6 Nov
    2026): CPython validates a .pyc by (source mtime in whole
    seconds, source size), so a candidate rewritten many times per
    second at an identical byte size loads the PREVIOUS candidate's
    bytecode and is never actually tested — the first baseline run
    scored 14/32 instead of 24/32 for this reason. PYTHONDONTWRITEBYTECODE
    alone is not enough (it stops writes, not reads of an existing
    .pyc), so every run deletes all __pycache__ dirs under cwd first
    AND runs with bytecode writing disabled.
    """
    start = time.monotonic()
    for root, dirs, _files in os.walk(cwd):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", test_id],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=env,
        )
        output = (proc.stdout + proc.stderr)[-2000:]
        # unittest exits 0 only if the test passed.
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        output = f"TIMEOUT after {timeout_secs}s: {(exc.stdout or b'')!r} {(exc.stderr or b'')!r}"[-2000:]
        passed = False
    return TestResult(
        test_id=test_id,
        passed=passed,
        duration_secs=time.monotonic() - start,
        output=output,
    )


def run_regression_suite(
    test_ids: list[str],
    cwd: str,
    max_tests: int = DEFAULT_MAX_REGRESSION_TESTS,
    timeout_secs: int = DEFAULT_REGRESSION_TIMEOUT_SECS,
    per_test_timeout_secs: int = DEFAULT_PER_TEST_TIMEOUT_SECS,
    abort_on_first_failure: bool = True,
) -> RegressionReport:
    """Run a capped regression suite, aborting on the first failure."""
    report = RegressionReport()
    start = time.monotonic()
    for test_id in test_ids[:max_tests]:
        if time.monotonic() - start >= timeout_secs:
            report.timed_out = True
            break
        result = run_one_test(test_id, cwd, timeout_secs=per_test_timeout_secs)
        report.results.append(result)
        if not result.passed and abort_on_first_failure:
            report.aborted_early = True
            break
    report.total_secs = time.monotonic() - start
    return report
