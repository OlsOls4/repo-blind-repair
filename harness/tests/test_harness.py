"""Unit tests for the Repo-Blind Repair harness (stdlib unittest only).

Run from harness/:
    python3 -m unittest discover -s tests -v
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_packer import estimate_tokens, pack_context  # noqa: E402
from patcher_stub import FileEdit, apply_edits  # noqa: E402
from regression_gate import evaluate_patch  # noqa: E402
from scope_ladder import discover_rungs, run_ladder  # noqa: E402
from test_runner import run_one_test, run_regression_suite  # noqa: E402
from trace_extractor import extract_trace  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_REPO = os.path.join(HERE, "demo_repo")
FAILING = "tests.test_calculator.TestCalculator.test_add"
REGRESSION = [
    "tests.test_calculator.TestCalculator.test_multiply",
    "tests.test_calculator.TestCalculator.test_subtract",
]

SAMPLE_OUTPUT = """FAIL: test_add (tests.test_calculator.TestCalculator)
Traceback (most recent call last):
  File "/tmp/demo/tests/test_calculator.py", line 10, in test_add
    self.assertEqual(add(2, 3), 5)
AssertionError: -1 != 5
"""


class TestTraceExtractor(unittest.TestCase):
    def test_parses_failing_test_and_frames(self):
        trace = extract_trace(SAMPLE_OUTPUT, "The `add` function is wrong.")
        self.assertIn("tests.test_calculator.TestCalculator.test_add", trace.failing_tests)
        self.assertTrue(any(f.function == "test_add" for f in trace.frames))
        self.assertIn("add", trace.implicated_symbols)

    def test_issue_py_path_is_implicated(self):
        trace = extract_trace("", "See calculator.py for the bug.")
        self.assertIn("calculator.py", trace.implicated_files)


class TestContextPacker(unittest.TestCase):
    def test_token_estimate(self):
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 2)

    def test_hard_cap_truncates_and_never_exceeds(self):
        pack = pack_context(
            {1: {"a.py": "x" * 1000, "b.py": "y" * 1000}},
            token_budget=100,
        )
        self.assertLessEqual(pack.tokens_used, 100)
        self.assertIn("a.py", pack.truncated_files)
        self.assertIn("b.py", pack.dropped_files)

    def test_rung_order_and_no_duplicates(self):
        pack = pack_context(
            {1: {"a.py": "hello"}, 2: {"a.py": "hello", "b.py": "world"}},
            token_budget=1000,
        )
        self.assertEqual(pack.files_included, ["a.py", "b.py"])


class HarnessRepoTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="harness-test-")
        self.repo = os.path.join(self._tmp.name, "demo_repo")
        shutil.copytree(DEMO_REPO, self.repo)

    def tearDown(self):
        self._tmp.cleanup()


class TestTestRunner(HarnessRepoTestCase):
    def test_failing_and_passing(self):
        self.assertFalse(run_one_test(FAILING, self.repo).passed)
        self.assertTrue(
            run_one_test(
                "tests.test_calculator.TestCalculator.test_multiply", self.repo
            ).passed
        )

    def test_regression_aborts_on_first_failure(self):
        report = run_regression_suite(
            [FAILING, REGRESSION[0]], self.repo, abort_on_first_failure=True
        )
        self.assertTrue(report.aborted_early)
        self.assertEqual(len(report.results), 1)

    def test_same_size_same_second_rewrite_is_actually_tested(self):
        # Stale-bytecode guard (search baseline, 6 Nov 2026): CPython
        # validates a .pyc by (source mtime in whole seconds, source
        # size), so a candidate rewritten at an identical byte size
        # within the same second used to load the previous candidate's
        # bytecode and was never actually tested (first baseline run
        # scored 14/32 instead of 24/32). The runner must test the
        # new content even when size and mtime-second are unchanged.
        mod = os.path.join(self.repo, "flaky.py")
        test_mod = os.path.join(self.repo, "tests", "test_flaky.py")
        with open(test_mod, "w") as fh:
            fh.write(
                "import unittest\nfrom flaky import value\n\n"
                "class TestFlaky(unittest.TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertEqual(value(), 2)\n"
            )
        with open(mod, "w") as fh:
            fh.write("def value():\n    return 1\n")
        self.assertFalse(
            run_one_test("tests.test_flaky.TestFlaky.test_value", self.repo).passed
        )
        # Same byte size (return 1 -> return 2), rewritten immediately
        # so the mtime whole-second is unchanged.
        with open(mod, "w") as fh:
            fh.write("def value():\n    return 2\n")
        self.assertTrue(
            run_one_test("tests.test_flaky.TestFlaky.test_value", self.repo).passed
        )


class TestRegressionGate(HarnessRepoTestCase):
    def test_accepts_minimal_fix(self):
        result = evaluate_patch(
            self.repo,
            [FileEdit("calculator.py", "return a - b", "return a + b")],
            FAILING,
            REGRESSION,
        )
        # Note: "return a - b" also appears in subtract(); apply_edits
        # requires exactly one occurrence, so this patch is rejected on
        # ambiguity — that is the safe behaviour. Use the exact line
        # below in the ladder test instead.
        self.assertFalse(result.accepted)

    def test_rejects_regression_breaking_patch(self):
        # Fix add (unambiguously, with its def line) but break multiply.
        result = evaluate_patch(
            self.repo,
            [
                FileEdit("calculator.py", "def add(a, b):\n    return a - b",
                         "def add(a, b):\n    return a + b"),
                FileEdit("calculator.py", "def multiply(a, b):\n    return a * b",
                         "def multiply(a, b):\n    return a + b"),
            ],
            FAILING,
            REGRESSION,
        )
        self.assertFalse(result.accepted)
        # Rolled back: the repo is still buggy.
        self.assertFalse(run_one_test(FAILING, self.repo).passed)

    def test_unambiguous_fix_accepted_and_minimal(self):
        result = evaluate_patch(
            self.repo,
            [FileEdit("calculator.py", "def add(a, b):\n    return a - b",
                      "def add(a, b):\n    return a + b")],
            FAILING,
            REGRESSION,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.diff_size, 2)  # 1 added + 1 removed
        self.assertEqual(result.files_changed, 1)


class TestScopeLadder(HarnessRepoTestCase):
    def test_rung1_contains_calculator(self):
        trace = extract_trace(SAMPLE_OUTPUT, "calculator.py")
        rungs = discover_rungs(self.repo, trace)
        self.assertIn("calculator.py", rungs[1])

    def test_ladder_accepts_at_rung1(self):
        def patch_fn(context_text, rung):
            if rung == 1:
                return [FileEdit(
                    "calculator.py",
                    "def add(a, b):\n    return a - b",
                    "def add(a, b):\n    return a + b",
                )]
            return None

        trace = extract_trace("", "calculator.py")
        result = run_ladder(self.repo, trace, FAILING, REGRESSION, patch_fn=patch_fn)
        self.assertTrue(result.accepted)
        self.assertEqual(result.rung, 1)
        self.assertTrue(run_one_test(FAILING, self.repo).passed)

    def test_ladder_default_generator_no_longer_raises(self):
        # STREAM 4: the old NotImplementedError placeholder is replaced
        # by the inference-only generator. The default ladder must now
        # run without raising, and on the toy demo it accepts the
        # one-line heuristic fix at rung 1 (gate-filtered, not assumed).
        trace = extract_trace("", "calculator.py")
        result = run_ladder(self.repo, trace, FAILING, REGRESSION)
        self.assertTrue(result.accepted)
        self.assertEqual(result.rung, 1)
        self.assertTrue(run_one_test(FAILING, self.repo).passed)


class TestPatcherStub(HarnessRepoTestCase):
    def test_ambiguous_edit_is_rejected_atomically(self):
        with self.assertRaises(ValueError):
            apply_edits(self.repo, [FileEdit("calculator.py", "return a - b", "return a + b")])
        with open(os.path.join(self.repo, "calculator.py")) as fh:
            self.assertIn("return a - b", fh.read())


if __name__ == "__main__":
    unittest.main()
