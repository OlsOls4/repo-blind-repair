"""Local tests for the mutation pipeline. Run from mutation/:
    python -m unittest discover -s tests -v
No network, no GPU, no teacher models.
"""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator import generate_composite, generate_single  # noqa: E402
from licence_checker import check_entry  # noqa: E402
from operators import OPERATORS, MutationNotApplicable, apply_mutation  # noqa: E402

TOY = (Path(__file__).resolve().parent.parent / "demo_repo" / "toy.py").read_text()


class TestOperators(unittest.TestCase):
    def test_all_operators_apply_and_parse(self):
        for op in OPERATORS:
            with self.subTest(op=op):
                rec = apply_mutation(TOY, op, 0, filename="toy.py")
                ast.parse(rec["mutated_source"])
                self.assertNotEqual(rec["mutated_source"], TOY)
                self.assertIsNotNone(rec["line"])
                self.assertNotEqual(rec["original_code"], rec["mutated_code"])

    def test_deterministic(self):
        a = apply_mutation(TOY, "wrong_comparator", 3, filename="toy.py")
        b = apply_mutation(TOY, "wrong_comparator", 3, filename="toy.py")
        self.assertEqual(a["mutated_source"], b["mutated_source"])
        c = apply_mutation(TOY, "wrong_comparator", 4, filename="toy.py")
        # A different seed may legitimately pick the same candidate
        # only if there is one candidate; toy.py has several.
        self.assertNotEqual(a["mutated_source"], c["mutated_source"])

    def test_unknown_operator(self):
        with self.assertRaises(ValueError):
            apply_mutation(TOY, "nope", 0)

    def test_no_candidate(self):
        with self.assertRaises(MutationNotApplicable):
            apply_mutation("x = 1\n", "swapped_boolean", 0)

    def test_wrong_comparator_skips_unswappable_compares(self):
        # D1 regression (8 Nov 2026): _Mutator.visit_Compare used
        # to count every Compare node before checking the operator
        # is swappable, while _candidates counts only swappable
        # ones — so with an `is`/`in` compare before the target,
        # the mutation landed on the wrong node or nowhere. The
        # target here is the only swappable compare; seed 0 must
        # flip `a < b` to `a <= b`, not touch the `is` check.
        src = (
            "def f(x, a, b):\n"
            "    if x is None:\n"
            "        return 0\n"
            "    if a < b:\n"
            "        return 1\n"
            "    return 0\n"
        )
        rec = apply_mutation(src, "wrong_comparator", 0, filename="probe.py")
        self.assertEqual(rec["original_code"], "a < b")
        self.assertEqual(rec["mutated_code"], "a <= b")
        self.assertIn("a <= b", rec["mutated_source"])
        self.assertIn("x is None", rec["mutated_source"])


class TestGenerator(unittest.TestCase):
    def test_single_record_restores_original(self):
        rec = generate_single(TOY, "off_by_one", 0, filename="toy.py")
        ast.parse(rec["buggy_code"])
        self.assertEqual(rec["fixed_code"], TOY)
        self.assertEqual(rec["bug_class"], "off_by_one")
        self.assertEqual(rec["failing_test"], "")  # never invented
        self.assertEqual(len(rec["mutations"]), 1)

    def test_composite_record(self):
        rec = generate_composite(TOY, "off_by_one", "wrong_constant", 2, filename="toy.py")
        ast.parse(rec["buggy_code"])
        self.assertEqual(rec["fixed_code"], TOY)
        self.assertEqual(rec["bug_class"], "composite:off_by_one+wrong_constant")
        self.assertEqual(len(rec["mutations"]), 2)


class TestLicenceChecker(unittest.TestCase):
    def test_permissive_accepted(self):
        r = check_entry({"name": "x", "licence": "MIT", "licence_file": ""})
        self.assertEqual(r["verdict"], "ACCEPT")

    def test_copyleft_rejected(self):
        for lic in ("GPL-3.0", "AGPL-3.0", "CC-BY-NC"):
            with self.subTest(lic=lic):
                r = check_entry({"name": "x", "licence": lic, "licence_file": ""})
                self.assertEqual(r["verdict"], "REJECT")

    def test_unverified_when_file_missing(self):
        r = check_entry(
            {"name": "x", "licence": "MIT", "licence_file": "/nonexistent/LICENSE"}
        )
        self.assertEqual(r["verdict"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
