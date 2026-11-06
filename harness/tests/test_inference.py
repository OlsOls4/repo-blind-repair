"""Tests for the inference-only patch generator (STREAM 4).

Run from harness/:
    python3 -m unittest discover -s tests -v

These tests assert real behaviour only: no repair rate, score, or
timing is claimed. The demo repo's one-line operator bug is inside
the heuristic generator's narrow scope; the unfixable-repo test pins
down what it cannot do.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_agent import run_inference_only  # noqa: E402
from patcher_stub import (  # noqa: E402
    build_repair_prompt,
    build_trained_format_prompt,
    fixed_file_to_edits,
    generate_candidates,
    generate_patch,
    is_test_file,
    parse_model_patch_output,
    parse_packed_context,
)
from test_runner import run_one_test  # noqa: E402
from trace_extractor import extract_trace  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_REPO = os.path.join(HERE, "demo_repo")
FAILING = "tests.test_calculator.TestCalculator.test_add"
REGRESSION = [
    "tests.test_calculator.TestCalculator.test_multiply",
    "tests.test_calculator.TestCalculator.test_subtract",
]

DEMO_PACK = """--- calculator.py (rung 1) ---
def add(a, b):
    return a - b


def multiply(a, b):
    return a * b


def subtract(a, b):
    return a - b
"""


class TestInferenceGenerator(unittest.TestCase):
    def test_generate_patch_no_longer_raises(self):
        # Previously NotImplementedError; now first candidate or None.
        patch = generate_patch(DEMO_PACK, 1)
        self.assertIsNotNone(patch)
        self.assertEqual(len(patch), 1)
        self.assertEqual(patch[0].path, "calculator.py")
        self.assertEqual(patch[0].old_text, "def add(a, b):\n    return a - b")
        self.assertEqual(patch[0].new_text, "def add(a, b):\n    return a + b")

    def test_generate_patch_returns_none_when_no_candidate(self):
        pack = "--- x.py (rung 1) ---\ndef f():\n    return 42\n"
        self.assertIsNone(generate_patch(pack, 1))

    def test_candidates_are_single_line_single_file(self):
        candidates = generate_candidates(DEMO_PACK, 1)
        self.assertGreaterEqual(len(candidates), 2)
        for patch in candidates:
            self.assertEqual(len(patch), 1)
            edit = patch[0]
            # One line changed (def-anchored edits count 2 lines of text
            # but a single changed line: the def line is unchanged).
            self.assertIn("return", edit.old_text)
            self.assertIn("return", edit.new_text)

    def test_candidates_deduplicated(self):
        candidates = generate_candidates(DEMO_PACK, 1)
        keys = [(p[0].path, p[0].old_text, p[0].new_text) for p in candidates]
        self.assertEqual(len(keys), len(set(keys)))

    def test_parse_packed_context(self):
        files = parse_packed_context(DEMO_PACK)
        self.assertEqual([p for p, _ in files], ["calculator.py"])
        self.assertIn("def add", files[0][1])

    def test_trained_format_prompt_matches_training(self):
        # Inference prompt is the trained shape minus the bug-class
        # label (real tasks carry none; inventing one would be fake).
        content = "def add(a, b):\n    return a - b\n"
        prompt = build_trained_format_prompt(content)
        self.assertEqual(
            prompt,
            "Fix this Python bug:\ndef add(a, b):\n    return a - b\n"
            "Fixed code:\n")
        self.assertNotIn("bug class:", prompt)
        self.assertNotIn("FILE:", prompt)

    def test_legacy_parser_round_trip_retained(self):
        # FILE:/OLD:/NEW: parsing is backwards compatibility only;
        # it is NOT the model format any more (see patcher_stub).
        reply = (
            "FILE: calculator.py\nOLD:\ndef add(a, b):\n    return a - b\n"
            "NEW:\ndef add(a, b):\n    return a + b\n"
        )
        parsed = parse_model_patch_output(reply)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0].new_text, "def add(a, b):\n    return a + b")
        # The legacy prompt builder is likewise retained, not used.
        legacy = build_repair_prompt(DEMO_PACK, 1, FAILING, "AssertionError: -1 != 5")
        self.assertIn("FILE:", legacy)

    def test_fixed_file_to_edits_single_hunk(self):
        original = "def f(x):\n    return x - 1\n\ndef g(x):\n    return x * 2\n"
        fixed = "def f(x):\n    return x + 1\n\ndef g(x):\n    return x * 2\n"
        edits = fixed_file_to_edits("m.py", original, fixed)
        self.assertIsNotNone(edits)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].old_text, "    return x - 1")
        self.assertEqual(edits[0].new_text, "    return x + 1")

    def test_fixed_file_to_edits_rejects_empty_identical_unparsable(self):
        original = "def f(x):\n    return x - 1\n"
        self.assertIsNone(fixed_file_to_edits("m.py", original, ""))
        self.assertIsNone(fixed_file_to_edits("m.py", original, "   \n"))
        self.assertIsNone(fixed_file_to_edits("m.py", original, original))
        self.assertIsNone(fixed_file_to_edits(
            "m.py", original, "def f(x):\n    return x + \n"))
        # A repeated training header is truncated, not treated as code.
        self.assertIsNone(fixed_file_to_edits(
            "m.py", original, "Fix this Python bug:\ndef other():\n    pass\n"))

    def test_fixed_file_to_edits_insertion_and_ambiguous_fallback(self):
        original = "def f(x):\n    y = x\n    return y\n"
        fixed = "def f(x):\n    y = x\n    y = y + 1\n    return y\n"
        edits = fixed_file_to_edits("m.py", original, fixed)
        self.assertIsNotNone(edits)
        # Simulated application must reproduce the fixed file.
        text = original
        for edit in edits:
            self.assertEqual(text.count(edit.old_text), 1)
            text = text.replace(edit.old_text, edit.new_text, 1)
        self.assertEqual(text.splitlines(), fixed.splitlines())
        # Ambiguous changed block (same line twice) falls back to a
        # single whole-file edit, never a force-applied guess.
        ambiguous_orig = "def f(x):\n    return x - 1\ndef g(x):\n    return x - 1\n"
        ambiguous_fixed = "def f(x):\n    return x + 1\ndef g(x):\n    return x - 1\n"
        edits = fixed_file_to_edits("m.py", ambiguous_orig, ambiguous_fixed)
        self.assertIsNotNone(edits)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].old_text, ambiguous_orig)
        self.assertIn("return x + 1", edits[0].new_text)

    def test_is_test_file(self):
        self.assertTrue(is_test_file("tests/test_task.py"))
        self.assertTrue(is_test_file("test_x.py"))
        self.assertTrue(is_test_file("tests/conftest.py"))
        self.assertFalse(is_test_file("task.py"))
        self.assertFalse(is_test_file("pkg/calculator.py"))

    def test_model_parser_rejects_malformed(self):
        self.assertIsNone(parse_model_patch_output(""))
        self.assertIsNone(parse_model_patch_output("just some prose"))
        self.assertIsNone(parse_model_patch_output(
            "FILE: calculator.py\nOLD:\nreturn a - b\nNEW:\nreturn a - b\n"))
        self.assertIsNone(parse_model_patch_output(
            "FILE: ../evil.py\nOLD:\nx\nNEW:\ny\n"))

    def test_model_candidate_tried_first(self):
        # Model path is now the trained format: one full fixed file
        # per source file, diffed into a patch — NOT FILE:/OLD:/NEW:.
        seen_prompts: list[str] = []

        def fake_model(prompt):
            seen_prompts.append(prompt)
            self.assertTrue(prompt.startswith("Fix this Python bug:\n"))
            self.assertIn("Fixed code:\n", prompt)
            self.assertIn("def add(a, b):", prompt)
            self.assertNotIn("FILE:", prompt)
            return ("def add(a, b):\n    return a + b\n\n\n"
                    "def multiply(a, b):\n    return a * b\n\n\n"
                    "def subtract(a, b):\n    return a - b\n")
        candidates = generate_candidates(DEMO_PACK, 1, model_fn=fake_model)
        self.assertEqual(len(seen_prompts), 1)
        self.assertIn("return a + b", candidates[0][0].new_text)

    def test_model_branch_skips_test_files(self):
        pack = ("--- task.py (rung 1) ---\ndef f(x):\n    return x - 1\n"
                "--- tests/test_task.py (rung 1) ---\nimport unittest\n")
        seen: list[str] = []

        def fake_model(prompt):
            seen.append(prompt)
            return "def f(x):\n    return x + 1\n"
        candidates = generate_candidates(pack, 1, model_fn=fake_model)
        self.assertEqual(len(seen), 1)
        self.assertIn("return x - 1", seen[0])
        self.assertNotIn("unittest", seen[0])
        self.assertTrue(any(p[0].path == "task.py" for p in candidates))

    def test_model_error_propagates_not_hidden(self):
        def broken_model(prompt):
            raise RuntimeError("model unavailable")
        with self.assertRaises(RuntimeError):
            generate_candidates(DEMO_PACK, 1, model_fn=broken_model)

    def test_clean_truncates_at_fixed_code_cue(self):
        # Diagnosed 1 Nov 2026: the model never learned EOS after the
        # fixed file, so it loops into repeated "Fixed code:" blocks
        # of corrupted fragments. The cue line truncates the output
        # to the first (candidate) block instead of failing to parse.
        from patcher_stub import clean_generated_file
        raw = ("def f(x):\n    return x + 1\n"
               "Fixed code:\n"
               "def f(x,_ y):\n    return x //orsites\n")
        cleaned = clean_generated_file(raw)
        self.assertEqual(cleaned, "def f(x):\n    return x + 1\n")
        import ast
        ast.parse(cleaned)


class InferenceRepoTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="inference-test-")
        self.repo = os.path.join(self._tmp.name, "demo_repo")
        shutil.copytree(DEMO_REPO, self.repo)

    def tearDown(self):
        self._tmp.cleanup()


class TestInferenceAgent(InferenceRepoTestCase):
    def test_accepts_demo_at_rung1_via_gate(self):
        trace = extract_trace("", "calculator.py")
        result = run_inference_only(self.repo, trace, FAILING, REGRESSION)
        self.assertTrue(result.accepted)
        self.assertEqual(result.rung, 1)
        self.assertIsNotNone(result.gate)
        self.assertEqual(result.gate.diff_size, 2)  # 1 added + 1 removed
        self.assertTrue(run_one_test(FAILING, self.repo).passed)
        self.assertTrue(run_one_test(REGRESSION[0], self.repo).passed)

    def test_unfixable_bug_leaves_repo_unchanged(self):
        # Wrong constant: outside the one-line-flip scope, so the bug
        # itself gets no candidate. Flips to the OTHER functions are
        # tried and gate-rejected, and the repo must stay untouched.
        with open(os.path.join(self.repo, "calculator.py"), "w") as fh:
            fh.write("def add(a, b):\n    return 42\n\n\n"
                     "def multiply(a, b):\n    return a * b\n\n\n"
                     "def subtract(a, b):\n    return a - b\n")
        trace = extract_trace("", "calculator.py")
        result = run_inference_only(self.repo, trace, FAILING, REGRESSION)
        self.assertFalse(result.accepted)
        self.assertGreaterEqual(result.candidates_tried, 1)
        with open(os.path.join(self.repo, "calculator.py")) as fh:
            self.assertIn("return 42", fh.read())


if __name__ == "__main__":
    unittest.main()
