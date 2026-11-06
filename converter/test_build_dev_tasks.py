"""Unit tests for the regression-derivation fix in build_dev_tasks.py
(1 Nov 2026).

The old converter left regression_test_ids empty (fix rate 0 by
construction) and skipped every task whose test_patch modified
existing tests or added several tests. The new converter derives
FAIL_TO_PASS / PASS_TO_PASS by actually running the tests in
scratch copies: a failing candidate must FAIL on buggy code and
PASS on gold-patched code (the gold check guards against buggy-only
mislabelling of environment failures), and regressions must pass on
BOTH, capped at 15, touched files first. The gold patch is applied
only inside a scratch copy and never written to the output.

These tests run WITHOUT Kaggle and WITHOUT a model: tiny fixture
repos (a buggy `add` that subtracts) are built as .tgz snapshots in
a temp dir, converted through build_dev_tasks.main, and the output
records + skip behaviour are asserted. Diffs are built with
"\\n".join so context/addition prefixes are exact — the converter's
strict applier rejects anything else, which is the point.

Run:  python -m unittest test_build_dev_tasks -v   (from kaggle/)
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

import build_dev_tasks
import eval_dev
from build_dev_tasks import (
    _classify_test_patch,
    _enumerate_tests,
    _order_regression_candidates,
)

HERE = Path(__file__).resolve().parent

CALC_BUGGY = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"

GOLD_PATCH = "\n".join([
    "--- a/calc.py",
    "+++ b/calc.py",
    "@@ -1,3 +1,3 @@",
    " def add(a, b):",
    "-    return a - b",
    "+    return a + b",
    "",
])

GOLD_PATCH_BROKEN = "\n".join([
    "--- a/calc.py",
    "+++ b/calc.py",
    "@@ -1,3 +1,3 @@",
    " def add(a, b):",
    "-    return a + b",
    "+    return a - b",
    "",
])

CALC_TESTS_BASE = (
    "import unittest\n"
    "from calc import add, mul\n"
    "\n"
    "\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_mul(self):\n"
    "        self.assertEqual(mul(2, 3), 6)\n"
    "\n"
    "    def test_add_old(self):\n"
    "        self.assertEqual(add(0, 0), 0)\n"
)


def _append_test_patch(new_tests: list[tuple[str, str]],
                       old_len: int = 3, new_len: int | None = None) -> str:
    """Build a test_patch appending `new_tests` to CALC_TESTS_BASE.

    Each new test is (name, assertion-line). The hunk appends after
    the test_add_old assertion (old lines 8-10).
    """
    if new_len is None:
        new_len = old_len + sum(3 for _ in new_tests)
    lines = [
        "--- a/tests/test_calc.py",
        "+++ b/tests/test_calc.py",
        f"@@ -8,{old_len} +8,{new_len} @@",
        " ",
        "     def test_add_old(self):",
        "         self.assertEqual(add(0, 0), 0)",
    ]
    for name, assertion in new_tests:
        lines += ["+", f"+    def {name}(self):",
                  f"+        self.assertEqual({assertion})"]
    lines.append("")
    return "\n".join(lines)


PATCH_ADD_ONE = _append_test_patch([("test_add", "add(2, 3), 5")])

PATCH_ADD_TWO_ONE_PASSES_BUGGY = _append_test_patch([
    ("test_add", "add(2, 3), 5"),
    ("test_add_zero", "add(0, 0), 0"),
])

PATCH_ADD_FAILS_BOTH = _append_test_patch([("test_add", "add(2, 3), 6")])

PATCH_ADD_TWO_BOTH_FAIL_BUGGY = _append_test_patch([
    ("test_add", "add(2, 3), 5"),
    ("test_add2", "add(3, 4), 7"),
])

PATCH_BROKEN_CONTEXT = "\n".join([
    "--- a/tests/test_calc.py",
    "+++ b/tests/test_calc.py",
    "@@ -8,3 +8,6 @@",
    " ",
    "     def test_add_old(self):",
    "         self.assertEqual(add(9, 9), 0)",
    "+",
    "+    def test_add(self):",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])

# Modified-test task: snapshot's test_add asserts -1 (passes buggy);
# the test_patch changes it to 5 (fails buggy, passes gold).
CALC_TESTS_MOD = (
    "import unittest\n"
    "from calc import add, mul\n"
    "\n"
    "\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_mul(self):\n"
    "        self.assertEqual(mul(2, 3), 6)\n"
    "\n"
    "    def test_add(self):\n"
    "        self.assertEqual(add(2, 3), -1)\n"
)

PATCH_MODIFY = "\n".join([
    "--- a/tests/test_calc.py",
    "+++ b/tests/test_calc.py",
    "@@ -9,2 +9,2 @@",
    "     def test_add(self):",
    "-        self.assertEqual(add(2, 3), -1)",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])

# Non-collecting task: the T1 patch plus a brand-new tests/helpers.py
# (not matching unittest's test*.py discovery pattern) carrying a
# test that would pass gold — it must be dropped, not recorded.
HELPERS_NEW = "\n".join([
    "--- /dev/null",
    "+++ b/tests/helpers.py",
    "@@ -0,0 +1,7 @@",
    "+import unittest",
    "+from calc import add",
    "+",
    "+",
    "+class TestHelper(unittest.TestCase):",
    "+    def test_helper(self):",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])

PATCH_ADD_ONE_PLUS_HELPERS = PATCH_ADD_ONE + HELPERS_NEW

# Empty-regression task: the only pre-existing test fails on buggy
# code, so no regression survives.
CALC_TESTS_ONLY_FAILING = (
    "import unittest\n"
    "from calc import add\n"
    "\n"
    "\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_old_fail(self):\n"
    "        self.assertEqual(add(2, 3), 5)\n"
)

PATCH_APPEND_NEW = "\n".join([
    "--- a/tests/test_calc.py",
    "+++ b/tests/test_calc.py",
    "@@ -5,3 +5,6 @@",
    " class TestCalc(unittest.TestCase):",
    "     def test_old_fail(self):",
    "         self.assertEqual(add(2, 3), 5)",
    "+",
    "+    def test_new(self):",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])


def _many_tests(n: int) -> str:
    lines = ["import unittest", "", "", "class TestMany(unittest.TestCase):"]
    for i in range(n):
        lines.append(f"    def test_{i:02d}(self):")
        lines.append("        self.assertTrue(True)")
    return "\n".join(lines) + "\n"


def _make_tgz(tree: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted(tree):
            data = tree[path].encode("utf-8")
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_tgz_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """Build a .tgz from (path, raw bytes) pairs, in given order.

    Unlike _make_tgz this takes raw bytes (so binary members and
    duplicate names can be expressed) and preserves entry order
    (so a "./"-prefixed .git member can be tested as shipped).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, data in entries:
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _record(task_id: str, gold: str, test_patch: str) -> dict:
    return {
        "instance_id": task_id,
        "repo": "fixture/calc",
        "base_commit": "0" * 40,
        "patch": gold,
        "test_patch": test_patch,
        "problem_statement": f"Fix {task_id}",
        "hints_text": "",
        "created_at": "2026-11-01T00:00:00Z",
    }


class TestHelpers(unittest.TestCase):
    def test_enumerate_nested_and_module_level(self):
        tree = {
            "pkg/mod.py": (
                "import unittest\n"
                "\n"
                "\n"
                "class Outer(unittest.TestCase):\n"
                "    class Inner(unittest.TestCase):\n"
                "        def test_deep(self):\n"
                "            pass\n"
                "\n"
                "\n"
                "def test_module_level():\n"
                "    pass\n"
            )
        }
        enum = _enumerate_tests(tree)
        # Nested class uses its qualname (dotted unittest id);
        # module-level functions use pytest ids (third-round fix:
        # the old enumerator excluded them, a blind spot for
        # pytest-style suites like fastapi/requests/rich).
        self.assertEqual(list(enum), [
            "pkg/mod.py::test_module_level",
            "pkg.mod.Outer.Inner.test_deep",
        ])

    def test_classify_added_modified_deleted(self):
        old = {"tests/test_x.py": (
            "import unittest\n"
            "class TestX(unittest.TestCase):\n"
            "    def test_a(self):\n"
            "        self.assertTrue(True)\n"
            "    def test_b(self):\n"
            "        self.assertTrue(True)\n"
        )}
        new = {"tests/test_x.py": (
            "import unittest\n"
            "class TestX(unittest.TestCase):\n"
            "    def test_a(self):\n"
            "        self.assertTrue(False)\n"
            "    def test_c(self):\n"
            "        self.assertTrue(True)\n"
        )}
        added, modified = _classify_test_patch(old, new)
        self.assertEqual(added, ["tests.test_x.TestX.test_c"])
        self.assertEqual(modified, ["tests.test_x.TestX.test_a"])

    def test_order_regression_touched_first_then_id(self):
        enum = {
            "tests.test_a.TestA.test_z": ("tests/test_a.py", 1, 2, ()),
            "tests.test_b.TestB.test_a": ("tests/test_b.py", 1, 2, ()),
            "tests.test_c.TestC.test_m": ("tests/test_c.py", 1, 2, ()),
        }
        ordered = _order_regression_candidates(
            ["tests.test_c.TestC.test_m", "tests.test_a.TestA.test_z",
             "tests.test_b.TestB.test_a"],
            enum, added=[], modified=[],
            test_touched={"tests/test_b.py"},
            gold_touched={"tests/test_a.py"})
        self.assertEqual(ordered, [
            "tests.test_b.TestB.test_a",
            "tests.test_a.TestA.test_z",
            "tests.test_c.TestC.test_m",
        ])


class TestConvertFixture(unittest.TestCase):
    """End-to-end conversion over an 11-task fixture file."""

    def _build_fixture(self, tmpdir: Path) -> tuple[Path, Path]:
        snaps = tmpdir / "snapshots"
        snaps.mkdir()
        base_tree = {
            "calc.py": CALC_BUGGY,
            "tests/__init__.py": "",
            "tests/test_calc.py": CALC_TESTS_BASE,
        }
        tasks = [
            # Converts: single added test, fails buggy / passes gold.
            ("t1-single", base_tree, GOLD_PATCH, PATCH_ADD_ONE),
            # Converts: two added, one passes on buggy -> only one
            # FAIL_TO_PASS survivor.
            ("t2-two-added", base_tree, GOLD_PATCH,
             PATCH_ADD_TWO_ONE_PASSES_BUGGY),
            # Converts: modified test (old converter skipped these).
            ("t3-modified", {**base_tree,
                             "tests/test_calc.py": CALC_TESTS_MOD},
             GOLD_PATCH, PATCH_MODIFY),
            # Skips: added test fails on gold too (env/unfixable).
            ("t4-fails-both", base_tree, GOLD_PATCH, PATCH_ADD_FAILS_BOTH),
            # Converts: helpers.py test is non-collecting -> dropped.
            ("t5-noncollecting", base_tree, GOLD_PATCH,
             PATCH_ADD_ONE_PLUS_HELPERS),
            # Skips: test_patch context mismatch (broken task).
            ("t6-broken", base_tree, GOLD_PATCH, PATCH_BROKEN_CONTEXT),
            # Converts: 20 pre-existing tests -> regression capped
            # at 15, touched-file tests first.
            ("t7-cap", {**base_tree,
                        "tests/test_many.py": _many_tests(20)},
             GOLD_PATCH, PATCH_ADD_ONE),
            # Skips: gold patch does not apply.
            ("t8-gold-broken", base_tree, GOLD_PATCH_BROKEN, PATCH_ADD_ONE),
            # Skips: two FAIL_TO_PASS survivors -> ambiguous.
            ("t9-ambiguous", base_tree, GOLD_PATCH,
             PATCH_ADD_TWO_BOTH_FAIL_BUGGY),
            # Skips: no surviving regression.
            ("t10-no-regression", {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_ONLY_FAILING,
            }, GOLD_PATCH, PATCH_APPEND_NEW),
            # Skips: no snapshot archive at all.
            ("t11-no-snapshot", None, GOLD_PATCH, PATCH_ADD_ONE),
        ]
        records = []
        for task_id, tree, gold, patch in tasks:
            if tree is not None:
                (snaps / f"{task_id}.tgz").write_bytes(_make_tgz(tree))
            records.append(_record(task_id, gold, patch))
        tasks_path = tmpdir / "tasks.jsonl"
        tasks_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        return tasks_path, snaps

    def test_multi_task_conversion(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-test-") as td:
            tmpdir = Path(td)
            tasks_path, snaps = self._build_fixture(tmpdir)
            out_path = tmpdir / "dev_tasks.jsonl"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = build_dev_tasks.main([
                    "convert", "--tasks", str(tasks_path),
                    "--snapshots", str(snaps), "--out", str(out_path),
                    "--python", sys.executable,
                    "--env-label", "test-fixture",
                ])
            self.assertEqual(code, 0)
            out = stdout.getvalue()
            err = stderr.getvalue()
            recs = [json.loads(line) for line in
                    out_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            by_id = {r["task_id"]: r for r in recs}
            # Converts: t1, t2, t3, t5, t7. Skips: t4, t6, t8, t9,
            # t10, t11 — and the run continued past every skip.
            self.assertEqual(sorted(by_id), [
                "t1-single", "t2-two-added", "t3-modified",
                "t5-noncollecting", "t7-cap",
            ])
            for skipped in ("t4-fails-both", "t6-broken", "t8-gold-broken",
                            "t9-ambiguous", "t10-no-regression",
                            "t11-no-snapshot"):
                self.assertIn(f"SKIP task {skipped}", err)
            self.assertIn("converted 5/11 tasks", out)
            self.assertIn("FAIL_TO_PASS coverage: 5/11", out)
            self.assertIn("PASS_TO_PASS coverage: 5/11", out)
            self.assertIn("dropped non-collecting ids: 1", out)
            self.assertIn("derivation env: test-fixture", out)
            # Failing ids are the single FAIL_TO_PASS survivor each.
            self.assertEqual(
                by_id["t1-single"]["failing_test_id"],
                "tests.test_calc.TestCalc.test_add")
            self.assertEqual(
                by_id["t2-two-added"]["failing_test_id"],
                "tests.test_calc.TestCalc.test_add")
            self.assertEqual(
                by_id["t3-modified"]["failing_test_id"],
                "tests.test_calc.TestCalc.test_add")
            self.assertEqual(
                by_id["t5-noncollecting"]["failing_test_id"],
                "tests.test_calc.TestCalc.test_add")
            # Non-collecting id dropped + counted, not recorded.
            self.assertEqual(
                by_id["t5-noncollecting"]["derivation"][
                    "dropped_noncollecting"],
                ["tests.helpers.TestHelper.test_helper"])
            self.assertEqual(
                by_id["t5-noncollecting"]["derivation"]["env"],
                "test-fixture")
            # Every record: non-empty regression within the gate cap.
            for rec in recs:
                self.assertTrue(1 <= len(rec["regression_test_ids"]) <= 15)
                self.assertNotIn(rec["failing_test_id"],
                                 rec["regression_test_ids"])
            # Regression cap: touched-file tests first, then id order.
            self.assertEqual(
                by_id["t7-cap"]["regression_test_ids"],
                ["tests.test_calc.TestCalc.test_add_old",
                 "tests.test_calc.TestCalc.test_mul"]
                + [f"tests.test_many.TestMany.test_{i:02d}"
                   for i in range(13)])
            # Gold patch NEVER written to the output: every record's
            # repo stays buggy (subtraction), never the gold fix.
            for rec in recs:
                self.assertIn("return a - b", rec["files"]["calc.py"])
                self.assertNotIn("return a + b", rec["files"]["calc.py"])
                self.assertEqual(rec["bug_class"], "unknown")
            # eval_dev.py --dry-run on the converted file shows
            # non-empty regression lists (previously all empty).
            dry = io.StringIO()
            with redirect_stdout(dry):
                eval_dev.main(["--dev-tasks", str(out_path),
                               "--no-adapter", "--dry-run"])
            dry_out = dry.getvalue()
            self.assertIn("regression lists non-empty for 5/5 tasks",
                          dry_out)
            self.assertIn("t7-cap: failing=tests.test_calc.TestCalc"
                          ".test_add regression=15", dry_out)

    def test_file_level_problems_still_abort(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-test-") as td:
            tmpdir = Path(td)
            tasks_path, snaps = self._build_fixture(tmpdir)
            # Duplicate task_id is a file-level integrity problem:
            # it must abort, not silently skip.
            dup = tasks_path.read_text(encoding="utf-8").splitlines()
            (tmpdir / "dup.jsonl").write_text(
                "\n".join([dup[0], dup[0]]) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                build_dev_tasks.main([
                    "convert", "--tasks", str(tmpdir / "dup.jsonl"),
                    "--snapshots", str(snaps),
                    "--out", str(tmpdir / "dup_out.jsonl"),
                    "--python", sys.executable,
                ])


class TestBinarySkip(unittest.TestCase):
    """Binary-skip allowlist (1 Nov 2026): real snapshots are git
    checkouts, so .git/ binaries must not abort the task, and
    other binaries must be listed, not hidden.

    Fixtures: a binary .git/index (the first offender on the real
    20-task subset, 0/20 converted), a nested .git/objects member,
    a "./"-prefixed .git member, a valid-UTF-8 .git/config (skipped
    by location, not content), a duplicate .git/index entry
    (deduplication), and a binary PNG asset (other-binary skip).
    """

    GIT_INDEX = b"\x00\x01\x02DIRC\x00\x00\x00\x02" + b"\xff" * 32
    GIT_OBJECT = b"x\x9c\xcb\x48\xcd\xc9\xc9\x07\x00\x06\x2c\x02\x15"
    PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    def _run_convert(self, tmpdir: Path, tasks, snaps) -> tuple[
            int, str, str, list[dict]]:
        records = [_record(tid, gold, patch)
                   for tid, gold, patch in tasks]
        tasks_path = tmpdir / "tasks.jsonl"
        tasks_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        out_path = tmpdir / "dev_tasks.jsonl"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = build_dev_tasks.main([
                "convert", "--tasks", str(tasks_path),
                "--snapshots", str(snaps), "--out", str(out_path),
                "--python", sys.executable,
                "--env-label", "test-fixture",
            ])
        recs = [json.loads(line) for line in
                out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if out_path.exists() else []
        return code, stdout.getvalue(), stderr.getvalue(), recs

    def test_git_and_asset_binary_skipped_and_listed(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-bin-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            entries: list[tuple[str, bytes]] = [
                ("calc.py", CALC_BUGGY.encode("utf-8")),
                ("tests/__init__.py", b""),
                ("tests/test_calc.py", CALC_TESTS_BASE.encode("utf-8")),
                # .git/ members: binary, nested, "./"-prefixed,
                # text-by-content (skipped by location anyway),
                # and duplicated (deduplicated in the record).
                (".git/index", self.GIT_INDEX),
                (".git/index", self.GIT_INDEX),
                (".git/objects/ab/cdef1234", self.GIT_OBJECT),
                ("./.git/packed-refs", b"\x00\xff\x00\xff"),
                (".git/config", b"[core]\n\trepositoryformatversion = 0\n"),
                # Other binary: a PNG asset, not under .git/.
                ("assets/logo.png", self.PNG_BYTES),
            ]
            (snaps / "tbin.tgz").write_bytes(_make_tgz_bytes(entries))
            code, out, _err, recs = self._run_convert(
                tmpdir, [("tbin", GOLD_PATCH, PATCH_ADD_ONE)], snaps)
            self.assertEqual(code, 0)
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            # Task converts: text files intact, gold never written.
            self.assertIn("return a - b", rec["files"]["calc.py"])
            self.assertNotIn("return a + b", rec["files"]["calc.py"])
            self.assertEqual(rec["failing_test_id"],
                             "tests.test_calc.TestCalc.test_add")
            self.assertEqual(rec["bug_class"], "unknown")
            # Every skipped path listed, sorted, deduplicated.
            self.assertEqual(rec["skipped_binary_files"], [
                ".git/config",
                ".git/index",
                ".git/objects/ab/cdef1234",
                ".git/packed-refs",
                "assets/logo.png",
            ])
            # No skipped member leaked into the evaluated files.
            for skipped in rec["skipped_binary_files"]:
                self.assertNotIn(skipped, rec["files"])
            self.assertIn("skipped binary members: 5 in 1 converted "
                          "task(s)", out)
            self.assertIn("converted 1/1 tasks", out)
            # eval_dev.py --dry-run accepts the record (extra key
            # ignored) with a non-empty regression list.
            dry = io.StringIO()
            out_path = tmpdir / "dev_tasks.jsonl"
            with redirect_stdout(dry):
                eval_dev.main(["--dev-tasks", str(out_path),
                               "--no-adapter", "--dry-run"])
            self.assertIn("regression lists non-empty for 1/1 tasks",
                          dry.getvalue())

    def test_single_top_dir_strips_skipped_paths(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-bin-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            entries: list[tuple[str, bytes]] = [
                ("repo/calc.py", CALC_BUGGY.encode("utf-8")),
                ("repo/tests/__init__.py", b""),
                ("repo/tests/test_calc.py",
                 CALC_TESTS_BASE.encode("utf-8")),
                ("repo/.git/index", self.GIT_INDEX),
                # Top-level binary asset outside the single top
                # dir: kept as-is, never dropped by stripping.
                ("logo.png", self.PNG_BYTES),
            ]
            (snaps / "tstrip.tgz").write_bytes(_make_tgz_bytes(entries))
            code, out, _err, recs = self._run_convert(
                tmpdir, [("tstrip", GOLD_PATCH, PATCH_ADD_ONE)], snaps)
            self.assertEqual(code, 0)
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            # Files are repo-relative after the strip, as eval
            # expects; skipped paths are stripped the same way.
            self.assertEqual(sorted(rec["files"]), [
                "calc.py", "tests/__init__.py", "tests/test_calc.py",
            ])
            self.assertEqual(rec["skipped_binary_files"], [
                ".git/index", "logo.png",
            ])
            self.assertIn("skipped binary members: 2 in 1 converted "
                          "task(s)", out)

    def test_corrupt_and_all_binary_snapshots_skip_loudly(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-bin-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            # Direct: a corrupt tar aborts loudly via _abort.
            corrupt = snaps / "tcorrupt.tgz"
            corrupt.write_bytes(b"this is not a tar archive at all")
            with self.assertRaises(SystemExit) as ctx:
                build_dev_tasks._extract_snapshot(corrupt, "tcorrupt")
            self.assertIn("cannot open snapshot", str(ctx.exception))
            # Direct: an all-binary snapshot has no text files.
            allbin = snaps / "tallbin.tgz"
            allbin.write_bytes(_make_tgz_bytes([
                (".git/index", self.GIT_INDEX),
                ("assets/logo.png", self.PNG_BYTES),
            ]))
            with self.assertRaises(SystemExit):
                build_dev_tasks._extract_snapshot(allbin, "tallbin")
            # End-to-end: corrupt + all-binary tasks are skipped
            # (counted, listed) and the run continues past them.
            good = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            (snaps / "tgood.tgz").write_bytes(_make_tgz(good))
            code, out, err, recs = self._run_convert(tmpdir, [
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
                ("tcorrupt", GOLD_PATCH, PATCH_ADD_ONE),
                ("tallbin", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0)
            self.assertEqual([r["task_id"] for r in recs], ["tgood"])
            self.assertIn("SKIP task tcorrupt", err)
            self.assertIn("SKIP task tallbin", err)
            self.assertIn("converted 1/3 tasks", out)


# Added test file: the test_patch ADDS tests/test_new.py, whose ---
# path names a real path (not /dev/null) while the snapshot has no
# such file — the normal SWE-bench new-test-file case (7 tasks on
# the second 20-task subset run). Pure-creation hunks only.
PATCH_ADD_NEW_FILE = "\n".join([
    "--- a/tests/test_new.py",
    "+++ b/tests/test_new.py",
    "@@ -0,0 +1,7 @@",
    "+import unittest",
    "+from calc import add",
    "+",
    "+",
    "+class TestNew(unittest.TestCase):",
    "+    def test_add(self):",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])

# Missing file with a context line: NOT a pure creation — the
# baseline cannot be guessed, so this must skip, not convert.
PATCH_MISSING_WITH_CONTEXT = "\n".join([
    "--- a/tests/test_missing.py",
    "+++ b/tests/test_missing.py",
    "@@ -1,2 +1,3 @@",
    " import unittest",
    " from calc import add",
    "+",
    "",
])

# Missing file, pure additions anchored at old line 10: an append
# to a file whose content we do not have, NOT a creation — must
# skip, not silently create a file from the added lines alone.
PATCH_MISSING_APPEND = "\n".join([
    "--- a/tests/test_append.py",
    "+++ b/tests/test_append.py",
    "@@ -10,0 +11,2 @@",
    "+    def test_append(self):",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])

# Snapshot already containing the test the patch adds, with a
# broken context line elsewhere: the mismatch diagnostic's
# already-applied flag must read True for this task.
CALC_TESTS_WITH_ADD = (
    CALC_TESTS_BASE
    + "\n"
    + "    def test_add(self):\n"
    + "        self.assertEqual(add(2, 3), 5)\n"
)

PATCH_ALREADY_APPLIED_BROKEN = "\n".join([
    "--- a/tests/test_calc.py",
    "+++ b/tests/test_calc.py",
    "@@ -8,3 +8,6 @@",
    " ",
    "     def test_add_old(self):",
    "         self.assertEqual(add(9, 9), 0)",
    "+",
    "+    def test_add(self):",
    "+        self.assertEqual(add(2, 3), 5)",
    "",
])


class TestSecondRoundFixes(unittest.TestCase):
    """Second-round converter fixes (1 Nov 2026, after the second
    20-task subset run converted 0/20 for three new reasons):
    (1) test_patch adding a file absent from the snapshot,
    (2) oversized members aborting the task, (3) context
    mismatches with no diagnostic. See build_dev_tasks.py (f),
    (g), (h)."""

    def _run(self, tmpdir: Path, tasks, snaps) -> tuple[
            int, str, str, list[dict]]:
        records = [_record(tid, gold, patch)
                   for tid, gold, patch in tasks]
        tasks_path = tmpdir / "tasks.jsonl"
        tasks_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        out_path = tmpdir / "dev_tasks.jsonl"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = build_dev_tasks.main([
                "convert", "--tasks", str(tasks_path),
                "--snapshots", str(snaps), "--out", str(out_path),
                "--python", sys.executable,
                "--env-label", "test-fixture",
            ])
        recs = [json.loads(line) for line in
                out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if out_path.exists() else []
        return code, stdout.getvalue(), stderr.getvalue(), recs

    def test_added_test_file_converts_and_missing_baselines_skip(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-add-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            base_tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            (snaps / "tgood.tgz").write_bytes(_make_tgz(base_tree))
            (snaps / "tadd.tgz").write_bytes(_make_tgz(base_tree))
            (snaps / "tmiss-ctx.tgz").write_bytes(_make_tgz(base_tree))
            (snaps / "tmiss-app.tgz").write_bytes(_make_tgz(base_tree))
            code, out, err, recs = self._run(tmpdir, [
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
                ("tadd", GOLD_PATCH, PATCH_ADD_NEW_FILE),
                ("tmiss-ctx", GOLD_PATCH, PATCH_MISSING_WITH_CONTEXT),
                ("tmiss-app", GOLD_PATCH, PATCH_MISSING_APPEND),
            ], snaps)
            self.assertEqual(code, 0)
            by_id = {r["task_id"]: r for r in recs}
            # tadd converts; the two missing-baseline tasks skip;
            # the run continues past every skip.
            self.assertEqual(sorted(by_id), ["tadd", "tgood"])
            self.assertIn("SKIP task tmiss-ctx", err)
            self.assertIn("SKIP task tmiss-app", err)
            self.assertIn("not a pure", err)
            self.assertIn("converted 2/4 tasks", out)
            # Failing id comes from the ADDED file, validated the
            # same way (fails buggy, passes gold); gold never
            # written; the new file is in the evaluated repo.
            rec = by_id["tadd"]
            self.assertEqual(rec["failing_test_id"],
                             "tests.test_new.TestNew.test_add")
            self.assertIn("tests/test_new.py", rec["files"])
            self.assertIn("return a - b", rec["files"]["calc.py"])
            self.assertNotIn("return a + b", rec["files"]["calc.py"])
            self.assertTrue(1 <= len(rec["regression_test_ids"]) <= 15)
            self.assertNotIn(rec["failing_test_id"],
                             rec["regression_test_ids"])

    def test_oversized_asset_skipped_with_reason_and_py_guarded(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-big-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            base_tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            big = build_dev_tasks.MAX_SNAPSHOT_FILE_BYTES
            # Oversized binary asset + oversized .git/ member (git
            # check precedes the size check, so no guard fires) +
            # small binary PNG: all skipped with explicit reasons.
            entries: list[tuple[str, bytes]] = [
                ("calc.py", CALC_BUGGY.encode("utf-8")),
                ("tests/__init__.py", b""),
                ("tests/test_calc.py", CALC_TESTS_BASE.encode("utf-8")),
                ("assets/logo.ai", b"A" * (big + 1)),
                (".git/index", b"\x00\x01DIRC" + b"\xff" * 32),
                (".git/hooks/big.py", b"x = 1\n" * 400_000),
                ("assets/logo.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
            ]
            (snaps / "tbig.tgz").write_bytes(_make_tgz_bytes(entries))
            (snaps / "tgood.tgz").write_bytes(_make_tgz(base_tree))
            # Oversized .py / .pyi members: counted skips, and the
            # run continues past them.
            big_py = {"calc.py": CALC_BUGGY,
                      "tests/__init__.py": "",
                      "tests/test_calc.py": CALC_TESTS_BASE,
                      "bigmod.py": "x = 1\n" * 400_000}
            (snaps / "tbig-py.tgz").write_bytes(_make_tgz(big_py))
            big_pyi = {"calc.py": CALC_BUGGY,
                       "tests/__init__.py": "",
                       "tests/test_calc.py": CALC_TESTS_BASE,
                       "bigmod.pyi": "x: int\n" * 300_000}
            (snaps / "tbig-pyi.tgz").write_bytes(_make_tgz(big_pyi))
            code, out, err, recs = self._run(tmpdir, [
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
                ("tbig", GOLD_PATCH, PATCH_ADD_ONE),
                ("tbig-py", GOLD_PATCH, PATCH_ADD_ONE),
                ("tbig-pyi", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0)
            by_id = {r["task_id"]: r for r in recs}
            self.assertEqual(sorted(by_id), ["tbig", "tgood"])
            self.assertIn("SKIP task tbig-py", err)
            self.assertIn("SKIP task tbig-pyi", err)
            self.assertIn("oversized evaluated source", err)
            self.assertIn("converted 2/4 tasks", out)
            rec = by_id["tbig"]
            self.assertEqual(rec["skipped_binary_files"], [
                ".git/hooks/big.py",
                ".git/index",
                "assets/logo.ai",
                "assets/logo.png",
            ])
            self.assertEqual(rec["skipped_binary_reasons"], {
                ".git/hooks/big.py": "git",
                ".git/index": "git",
                "assets/logo.ai": "oversized",
                "assets/logo.png": "binary",
            })
            for skipped in rec["skipped_binary_files"]:
                self.assertNotIn(skipped, rec["files"])
            self.assertIn("skipped binary reasons: binary=1, git=2, "
                          "oversized=1", out)

    def test_context_mismatch_diagnostic_flag_and_summary(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-diag-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            base_tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            applied_tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_WITH_ADD,
            }
            (snaps / "tgood.tgz").write_bytes(_make_tgz(base_tree))
            (snaps / "tmis.tgz").write_bytes(_make_tgz(base_tree))
            (snaps / "tapplied.tgz").write_bytes(
                _make_tgz(applied_tree))
            (snaps / "tgold.tgz").write_bytes(_make_tgz(base_tree))
            code, out, err, recs = self._run(tmpdir, [
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
                ("tmis", GOLD_PATCH, PATCH_BROKEN_CONTEXT),
                ("tapplied", GOLD_PATCH, PATCH_ALREADY_APPLIED_BROKEN),
                ("tgold", GOLD_PATCH_BROKEN, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0)
            self.assertEqual([r["task_id"] for r in recs], ["tgood"])
            for skipped in ("tmis", "tapplied", "tgold"):
                self.assertIn(f"SKIP task {skipped}", err)
            # Plain mismatch: task id, file, hunk location, the
            # expected line, the actual snapshot line in its
            # ±5 window, and the flag reading False.
            self.assertIn("CONTEXT MISMATCH task tmis", err)
            self.assertIn("tests/test_calc.py", err)
            self.assertIn("add(9, 9)", err)
            self.assertIn("add(0, 0)", err)
            self.assertIn("added-lines-already-in-snapshot: "
                          "anywhere=False", err)
            # Already-applied hypothesis: the added lines ARE in
            # the snapshot, so the flag reads True — reported,
            # not coded around (the task still skips).
            self.assertIn("CONTEXT MISMATCH task tapplied", err)
            self.assertIn("added-lines-already-in-snapshot: "
                          "anywhere=True", err)
            # Gold-patch mismatch is labelled as the gold patch
            # and names the source file, not a test file.
            self.assertIn("CONTEXT MISMATCH task tgold (gold patch)",
                          err)
            self.assertIn("calc.py", err)
            self.assertIn("return a + b", err)
            self.assertIn("return a - b", err)
            # Strictness kept: all three still skipped; and the
            # diagnostics are repeated in the run summary.
            self.assertIn("converted 1/4 tasks", out)
            self.assertIn("context-mismatch diagnostics: 3 mismatching "
                          "hunk(s)", out)
            self.assertIn("CONTEXT MISMATCH task tmis", out)
            self.assertIn("CONTEXT MISMATCH task tapplied", out)


def _make_tgz_links(entries: list[tuple]) -> bytes:
    """Build a .tgz from ("f", path, bytes) / ("sym", path, target)
    / ("lnk", path, target) entries, preserving order (so a link
    may precede its target, as in real snapshots)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for entry in entries:
            kind = entry[0]
            if kind == "f":
                _, path, data = entry
                info = tarfile.TarInfo(name=path)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            elif kind in ("sym", "lnk"):
                _, path, target = entry
                info = tarfile.TarInfo(name=path)
                info.type = (tarfile.SYMTYPE if kind == "sym"
                             else tarfile.LNKTYPE)
                info.linkname = target
                tf.addfile(info)
            else:  # pragma: no cover - test helper misuse
                raise ValueError(f"unknown entry kind {kind!r}")
    return buf.getvalue()


# Minimal fake `pytest` module (used ONLY as a test stub: real
# pytest is not installed in this VM). It speaks the two
# interfaces the converter uses: `--collect-only -q <dir>` prints
# module-level test ids (file::function, one per line), and a run
# of a single id imports the file and calls the function, exiting
# non-zero on assertion failure. The __main__ guard matters: plain
# `import pytest` (the availability probe) must be a no-op.
_FAKE_PYTEST_SRC = '''
import ast
import importlib.util
import os
import sys


def _test_files(root):
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if (name.startswith("test_") and name.endswith(".py")) or (
                    name.endswith("_test.py")):
                yield os.path.join(dirpath, name)


def _module_tests(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test")]


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--collect-only" in args:
        # Round 5 passes a single FILE target (scoped
        # collection); real pytest handles that natively, so
        # the fake must too: a file target yields its own
        # tests, with paths relative to the repo cwd.
        target = args[-1]
        if os.path.isfile(target):
            files = [target]
            base = os.getcwd()
        else:
            files = list(_test_files(target))
            base = target
        for path in files:
            rel = os.path.relpath(path, base).replace(os.sep, "/")
            for name in _module_tests(path):
                print(f"{rel}::{name}")
    else:
        tid = args[-1]
        path, func = tid.rsplit("::", 1)
        spec = importlib.util.spec_from_file_location(
            "fake_pytest_target", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, func)()
'''

CALC_TESTS_PYTEST = (
    "from calc import add, mul\n"
    "\n"
    "\n"
    "def test_mul():\n"
    "    assert mul(2, 3) == 6\n"
    "\n"
    "\n"
    "def test_add_old():\n"
    "    assert add(0, 0) == 0\n"
)

PATCH_ADD_PYTEST = "\n".join([
    "--- a/tests/test_py.py",
    "+++ b/tests/test_py.py",
    # CALC_TESTS_PYTEST lines 7-9 are the blank line, the
    # test_add_old def, and its assertion (two blank lines
    # separate the module-level functions).
    "@@ -7,3 +7,6 @@",
    " ",
    " def test_add_old():",
    "     assert add(0, 0) == 0",
    "+",
    "+def test_add():",
    "+    assert add(2, 3) == 5",
    "",
])


class TestThirdRoundFixes(unittest.TestCase):
    """Third-round converter fixes (1 Nov 2026, after the third
    20-task subset run still converted 0/20):
    (A) multi-hunk diffs shifted by earlier hunks, (B) symlink
    refusal, (C) the pytest module-level blind spot."""

    # ---- Cause A: hunk coordinates (old file, no delta) ----

    def test_apply_diff_insert_shift(self):
        tree = {"f.py": "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\n"}
        diff = "\n".join([
            "--- a/f.py", "+++ b/f.py",
            "@@ -1,3 +1,5 @@",
            " l1",
            " l2",
            "+ins_a",
            "+ins_b",
            " l3",
            "@@ -5,3 +7,3 @@",
            " l5",
            "-l6",
            "+l6_fixed",
            " l7",
            "",
        ])
        out = build_dev_tasks._apply_diff(
            tree, build_dev_tasks._parse_unified_diff(diff, "t"),
            "t", "test_patch")
        # Second hunk applies at its ORIGINAL line 5, not at the
        # shifted line 7 the old delta code aimed at (which
        # aborted with a context mismatch at line 7).
        self.assertEqual(
            out["f.py"],
            "l1\nl2\nins_a\nins_b\nl3\nl4\nl5\nl6_fixed\nl7\nl8\n")

    def test_apply_diff_delete_shift(self):
        tree = {"f.py": "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\n"}
        diff = "\n".join([
            "--- a/f.py", "+++ b/f.py",
            "@@ -2,4 +2,2 @@",
            " l2",
            "-l3",
            "-l4",
            " l5",
            "@@ -8,2 +6,2 @@",
            " l8",
            "-l9",
            "+l9_fixed",
            "",
        ])
        out = build_dev_tasks._apply_diff(
            tree, build_dev_tasks._parse_unified_diff(diff, "t"),
            "t", "test_patch")
        # Second hunk applies at original line 8, not the shifted
        # line 6 the old delta code aimed at (negative offset).
        self.assertEqual(
            out["f.py"], "l1\nl2\nl5\nl6\nl7\nl8\nl9_fixed\n")

    def test_apply_diff_single_hunk_unchanged(self):
        tree = {"f.py": "a\nb\nc\n"}
        diff = "\n".join([
            "--- a/f.py", "+++ b/f.py",
            "@@ -1,3 +1,3 @@",
            " a",
            "-b",
            "+B",
            " c",
            "",
        ])
        out = build_dev_tasks._apply_diff(
            tree, build_dev_tasks._parse_unified_diff(diff, "t"),
            "t", "test_patch")
        self.assertEqual(out["f.py"], "a\nB\nc\n")

    # ---- Cause B: link resolution ----

    def test_extract_intra_snapshot_symlink(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-link-") as td:
            arch = Path(td) / "tlink.tgz"
            # Link FIRST, target second: resolution is a second
            # pass, so archive order must not matter. The target
            # path mirrors the real requests case
            # (tests/certs/mtls/client/ca inside the snapshot).
            arch.write_bytes(_make_tgz_links([
                ("sym", "tests/certs/mtls/client/ca", "../ca.crt"),
                ("f", "tests/certs/mtls/ca.crt",
                 b"-----BEGIN CERTIFICATE-----\nFAKE\n"),
                ("f", "calc.py", CALC_BUGGY.encode("utf-8")),
            ]))
            tree, skipped, reasons = build_dev_tasks._extract_snapshot(
                arch, "tlink")
            self.assertEqual(
                tree["tests/certs/mtls/client/ca"],
                "-----BEGIN CERTIFICATE-----\nFAKE\n")
            self.assertNotIn("tests/certs/mtls/client/ca", skipped)

    def test_extract_link_chain_and_hard_link(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-link-") as td:
            arch = Path(td) / "tchain.tgz"
            arch.write_bytes(_make_tgz_links([
                ("f", "calc.py", CALC_BUGGY.encode("utf-8")),
                ("sym", "a-link", "b-link"),
                ("sym", "b-link", "calc.py"),
                ("lnk", "calc-link.py", "calc.py"),
            ]))
            tree, skipped, _reasons = build_dev_tasks._extract_snapshot(
                arch, "tchain")
            self.assertEqual(tree["a-link"], CALC_BUGGY)
            self.assertEqual(tree["b-link"], CALC_BUGGY)
            self.assertEqual(tree["calc-link.py"], CALC_BUGGY)
            self.assertEqual(skipped, [])

    def test_extract_outside_and_absolute_links_skip_listed(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-link-") as td:
            arch = Path(td) / "tout.tgz"
            arch.write_bytes(_make_tgz_links([
                ("f", "calc.py", CALC_BUGGY.encode("utf-8")),
                ("sym", "missing.pem", "no-such-file.pem"),
                ("sym", "abs.pem", "/etc/passwd"),
                ("sym", "escape.pem", "../../etc/passwd"),
                ("sym", "loop-a", "loop-b"),
                ("sym", "loop-b", "loop-a"),
            ]))
            tree, skipped, reasons = build_dev_tasks._extract_snapshot(
                arch, "tout")
            self.assertIn("calc.py", tree)
            self.assertEqual(skipped, [
                "abs.pem", "escape.pem", "loop-a", "loop-b",
                "missing.pem",
            ])
            self.assertEqual(
                set(reasons.values()), {"link-outside-snapshot"})
            for rel in skipped:
                self.assertNotIn(rel, tree)

    def test_extract_unsafe_link_path_still_aborts(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-link-") as td:
            arch = Path(td) / "tevil.tgz"
            arch.write_bytes(_make_tgz_links([
                ("f", "calc.py", CALC_BUGGY.encode("utf-8")),
                ("sym", "../evil", "calc.py"),
            ]))
            with self.assertRaises(SystemExit) as ctx:
                build_dev_tasks._extract_snapshot(arch, "tevil")
            self.assertIn("unsafe snapshot member", str(ctx.exception))

    def test_link_snapshot_converts_end_to_end(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-link-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            # Intra-snapshot cert link + outside link: the task
            # converts, the resolved link is NOT in the skipped
            # list, the outside link IS (with its reason).
            (snaps / "tlink.tgz").write_bytes(_make_tgz_links([
                ("f", "calc.py", CALC_BUGGY.encode("utf-8")),
                ("f", "tests/__init__.py", b""),
                ("f", "tests/test_calc.py",
                 CALC_TESTS_BASE.encode("utf-8")),
                ("sym", "tests/certs/mtls/client/ca", "../ca.crt"),
                ("f", "tests/certs/mtls/ca.crt",
                 b"-----BEGIN CERTIFICATE-----\nFAKE\n"),
                ("sym", "missing.pem", "no-such-file.pem"),
            ]))
            tasks_path = tmpdir / "tasks.jsonl"
            tasks_path.write_text(
                json.dumps(_record("tlink", GOLD_PATCH,
                                   PATCH_ADD_ONE)) + "\n",
                encoding="utf-8")
            out_path = tmpdir / "dev_tasks.jsonl"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = build_dev_tasks.main([
                    "convert", "--tasks", str(tasks_path),
                    "--snapshots", str(snaps), "--out", str(out_path),
                    "--python", sys.executable,
                    "--env-label", "test-fixture",
                ])
            self.assertEqual(code, 0)
            recs = [json.loads(line) for line in
                    out_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            self.assertEqual(rec["failing_test_id"],
                             "tests.test_calc.TestCalc.test_add")
            self.assertEqual(
                rec["files"]["tests/certs/mtls/client/ca"],
                "-----BEGIN CERTIFICATE-----\nFAKE\n")
            self.assertEqual(rec["skipped_binary_files"],
                             ["missing.pem"])
            self.assertEqual(rec["skipped_binary_reasons"],
                             {"missing.pem": "link-outside-snapshot"})

    # ---- Cause C: pytest module-level tests ----

    def test_enumerate_pytest_module_level(self):
        tree = {"tests/test_api.py": (
            "from app import get_status\n"
            "\n"
            "\n"
            "def test_status_ok():\n"
            "    assert get_status() == 200\n"
            "\n"
            "\n"
            "def test_status_404():\n"
            "    assert get_status('/nope') == 404\n"
        )}
        enum = _enumerate_tests(tree)
        self.assertEqual(sorted(enum), [
            "tests/test_api.py::test_status_404",
            "tests/test_api.py::test_status_ok",
        ])

    def test_classify_pytest_modified_function(self):
        old = {"tests/test_api.py": (
            "def test_status_ok():\n"
            "    assert get_status() == 200\n"
        )}
        new = {"tests/test_api.py": (
            "def test_status_ok():\n"
            "    assert get_status() == 201\n"
        )}
        added, modified = _classify_test_patch(old, new)
        self.assertEqual(added, [])
        self.assertEqual(modified, ["tests/test_api.py::test_status_ok"])

    def test_run_test_dispatches_by_id_shape(self):
        seen: list[tuple[str, str]] = []
        orig_unittest = build_dev_tasks._run_unittest
        orig_pytest = build_dev_tasks._run_pytest
        try:
            build_dev_tasks._run_unittest = lambda *a: seen.append(
                ("unittest", a[1])) or True
            build_dev_tasks._run_pytest = lambda *a: seen.append(
                ("pytest", a[1])) or True
            self.assertTrue(build_dev_tasks._run_test(
                sys.executable, "tests.test_x.TestX.test_a", "/tmp", 5))
            self.assertTrue(build_dev_tasks._run_test(
                sys.executable, "tests/test_x.py::test_a", "/tmp", 5))
        finally:
            build_dev_tasks._run_unittest = orig_unittest
            build_dev_tasks._run_pytest = orig_pytest
        self.assertEqual(seen, [
            ("unittest", "tests.test_x.TestX.test_a"),
            ("pytest", "tests/test_x.py::test_a"),
        ])

    def test_pytest_fixture_converts_end_to_end_with_fake_pytest(self):
        import tempfile

        # Real pytest is not installed in this VM, so the
        # end-to-end pytest path runs against the fake module
        # above via PYTHONPATH. What this proves: enumeration,
        # pytest collection merging, runner dispatch, and the
        # FAIL-on-buggy / PASS-on-gold semantics for pytest ids.
        # What it does NOT prove (Kaggle re-run must capture):
        # parsing of REAL pytest --collect-only output and real
        # pytest exit codes/conftest behaviour.
        with tempfile.TemporaryDirectory(prefix="build-dev-py-") as td:
            tmpdir = Path(td)
            fake_dir = tmpdir / "fakebin"
            fake_dir.mkdir()
            (fake_dir / "pytest.py").write_text(
                _FAKE_PYTEST_SRC, encoding="utf-8")
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_py.py": CALC_TESTS_PYTEST,
            }
            (snaps / "tpy.tgz").write_bytes(_make_tgz(tree))
            tasks_path = tmpdir / "tasks.jsonl"
            tasks_path.write_text(
                json.dumps(_record("tpy", GOLD_PATCH,
                                   PATCH_ADD_PYTEST)) + "\n",
                encoding="utf-8")
            out_path = tmpdir / "dev_tasks.jsonl"
            old_path = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = (
                str(fake_dir) + (f":{old_path}" if old_path else ""))
            try:
                self.assertTrue(
                    build_dev_tasks._pytest_available(sys.executable))
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = build_dev_tasks.main([
                        "convert", "--tasks", str(tasks_path),
                        "--snapshots", str(snaps),
                        "--out", str(out_path),
                        "--python", sys.executable,
                        "--env-label", "test-fixture-fake-pytest",
                    ])
                self.assertEqual(code, 0, stderr.getvalue())
            finally:
                if old_path is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old_path
            recs = [json.loads(line) for line in
                    out_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            self.assertEqual(rec["failing_test_id"],
                             "tests/test_py.py::test_add")
            self.assertTrue(rec["derivation"]["pytest_available"])
            # Pre-existing module-level tests pass on both sides.
            self.assertEqual(sorted(rec["regression_test_ids"]), [
                "tests/test_py.py::test_add_old",
                "tests/test_py.py::test_mul",
            ])
            self.assertNotIn("return a + b", rec["files"]["calc.py"])


def _make_tgz_mixed(entries: list[tuple]) -> bytes:
    """Build a .tgz from ("d", path) directory / ("f", path, bytes)
    file entries, preserving order — so a "./" root directory
    entry can lead the archive, as in real snapshots (fourth
    subset fix). The earlier helpers only ever created regular
    file members, which is exactly why no fixture caught the
    root-entry abort: no fixture archive contained a directory
    entry at all."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for entry in entries:
            kind = entry[0]
            if kind == "d":
                info = tarfile.TarInfo(name=entry[1])
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            elif kind == "f":
                _, path, data = entry
                info = tarfile.TarInfo(name=path)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:  # pragma: no cover - test helper misuse
                raise ValueError(f"unknown entry kind {kind!r}")
    return buf.getvalue()


class TestFourthRoundFixes(unittest.TestCase):
    """Fourth-round converter fix (1 Nov 2026, after the round-3
    re-run converted 0/20): every real snapshot begins with a
    "./" root directory entry, and the extractor aborted on it
    ("unsafe snapshot member '.'") because the zero-parts check
    ran before the non-file directory branch — so none of the
    round-3 fixes (hunk coordinates, symlinks, pytest) was ever
    exercised. Round-3's 0/20 therefore does NOT count toward
    the pivot criterion; round 4 is the true test."""

    def _run(self, tmpdir: Path, tasks, snaps) -> tuple[
            int, str, str, list[dict]]:
        records = [_record(tid, gold, patch)
                   for tid, gold, patch in tasks]
        tasks_path = tmpdir / "tasks.jsonl"
        tasks_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        out_path = tmpdir / "dev_tasks.jsonl"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = build_dev_tasks.main([
                "convert", "--tasks", str(tasks_path),
                "--snapshots", str(snaps), "--out", str(out_path),
                "--python", sys.executable,
                "--env-label", "test-fixture",
            ])
        recs = [json.loads(line) for line in
                out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if out_path.exists() else []
        return code, stdout.getvalue(), stderr.getvalue(), recs

    def test_real_snapshot_shape_converts(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r4-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            # Real snapshot shape, in real order: "./" root dir
            # first, then "./.git/" dir + HEAD, then "./"-prefixed
            # files. Pre-fix this aborted with "unsafe snapshot
            # member '.'" before any patch was applied.
            (snaps / "troot.tgz").write_bytes(_make_tgz_mixed([
                ("d", "./"),
                ("d", "./.git/"),
                ("f", "./.git/HEAD", b"ref: refs/heads/main\n"),
                ("f", "./calc.py", CALC_BUGGY.encode("utf-8")),
                ("f", "./tests/__init__.py", b""),
                ("f", "./tests/test_calc.py",
                 CALC_TESTS_BASE.encode("utf-8")),
            ]))
            code, out, err, recs = self._run(
                tmpdir, [("troot", GOLD_PATCH, PATCH_ADD_ONE)], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            self.assertEqual(rec["failing_test_id"],
                             "tests.test_calc.TestCalc.test_add")
            self.assertIn("return a - b", rec["files"]["calc.py"])
            self.assertNotIn("return a + b", rec["files"]["calc.py"])
            # The root entry is a directory, not a file: ignored,
            # NOT listed as a skipped binary. The .git/HEAD file
            # IS listed (skipped by location, reason "git").
            self.assertEqual(rec["skipped_binary_files"], [".git/HEAD"])
            self.assertEqual(rec["skipped_binary_reasons"],
                             {".git/HEAD": "git"})
            self.assertIn("converted 1/1 tasks", out)

    def test_bare_top_dir_entry_and_dot_member_ignored(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r4-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            base_tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            # Bare top-level directory entry (no trailing "./"
            # prefix): already ignored pre-fix via the
            # not-isfile branch — regression guard.
            (snaps / "tdir.tgz").write_bytes(_make_tgz_mixed([
                ("d", "repo/"),
                ("f", "repo/calc.py", CALC_BUGGY.encode("utf-8")),
                ("f", "repo/tests/__init__.py", b""),
                ("f", "repo/tests/test_calc.py",
                 CALC_TESTS_BASE.encode("utf-8")),
            ]))
            # A lone "." member (zero parts without any "./"
            # prefix) is the same root-entry shape.
            (snaps / "tdot.tgz").write_bytes(_make_tgz_mixed([
                ("d", "."),
                ("f", "calc.py", CALC_BUGGY.encode("utf-8")),
                ("f", "tests/__init__.py", b""),
                ("f", "tests/test_calc.py",
                 CALC_TESTS_BASE.encode("utf-8")),
            ]))
            code, out, err, recs = self._run(tmpdir, [
                ("tdir", GOLD_PATCH, PATCH_ADD_ONE),
                ("tdot", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual(sorted(r["task_id"] for r in recs),
                             ["tdir", "tdot"])
            # Single-top-dir strip still applies to the "repo/"
            # shape: files are repo-relative after conversion.
            by_id = {r["task_id"]: r for r in recs}
            self.assertEqual(sorted(by_id["tdir"]["files"]), [
                "calc.py", "tests/__init__.py", "tests/test_calc.py",
            ])
            self.assertIn("converted 2/2 tasks", out)

    def test_root_only_snapshot_skips_and_run_continues(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r4-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            (snaps / "tempty.tgz").write_bytes(_make_tgz_mixed([
                ("d", "./"),
            ]))
            (snaps / "tgood.tgz").write_bytes(_make_tgz({
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }))
            code, out, err, recs = self._run(tmpdir, [
                ("tempty", GOLD_PATCH, PATCH_ADD_ONE),
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual([r["task_id"] for r in recs], ["tgood"])
            self.assertIn("SKIP task tempty", err)
            self.assertIn("contains no files", err)
            self.assertIn("converted 1/2 tasks", out)

    def test_escaping_paths_still_abort(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r4-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            base = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            # "..", "./.." and "a/../b" members must still abort
            # the task (counted skip, run continues) — the
            # zero-parts fix must not weaken the ".." guard.
            for tid, bad in [("tesc1", ".."), ("tesc2", "./.."),
                             ("tesc3", "a/../evil.py")]:
                entries = [("d", "./")]
                entries += [("f", p, c.encode("utf-8"))
                            for p, c in base.items()]
                entries.append(("f", bad, b"x = 1\n"))
                (snaps / f"{tid}.tgz").write_bytes(
                    _make_tgz_mixed(entries))
            (snaps / "tgood.tgz").write_bytes(_make_tgz(base))
            code, out, err, recs = self._run(tmpdir, [
                ("tesc1", GOLD_PATCH, PATCH_ADD_ONE),
                ("tesc2", GOLD_PATCH, PATCH_ADD_ONE),
                ("tesc3", GOLD_PATCH, PATCH_ADD_ONE),
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual([r["task_id"] for r in recs], ["tgood"])
            for tid in ("tesc1", "tesc2", "tesc3"):
                self.assertIn(f"SKIP task {tid}", err)
                self.assertIn("unsafe snapshot member", err)
            self.assertIn("converted 1/4 tasks", out)


# Round-5 fixtures: scoped collection (1 Nov 2026). The old
# whole-tree discovery skipped a task whenever ANY module failed
# to import; round 4 showed that kills all 10 fastapi tasks on one
# unrelated broken module (benchmarks, ~300 modules discovered).

# A second fine test module, for regression collection across
# modules (its test passes on both buggy and gold code).
CALC_TESTS_OTHER = (
    "import unittest\n"
    "from calc import mul\n"
    "\n"
    "\n"
    "class TestOther(unittest.TestCase):\n"
    "    def test_other(self):\n"
    "        self.assertEqual(mul(2, 3), 6)\n"
)

# Unrelated broken module: imports a missing dependency, so
# whole-tree discovery yields a _FailedTest for it — but scoped
# collection never touches it, so it must not kill the task.
CALC_TESTS_BENCH_BROKEN = (
    "import definitely_missing_dep_xyz\n"
    "import unittest\n"
    "\n"
    "\n"
    "class TestBench(unittest.TestCase):\n"
    "    def test_bench(self):\n"
    "        self.assertTrue(True)\n"
)

# Candidate module that itself fails to import: the bad import
# goes AFTER the appended region, so PATCH_ADD_ONE still applies
# (lines 8-10 unchanged) but importing the module raises.
CALC_TESTS_BAD_IMPORT = CALC_TESTS_BASE + "\nimport definitely_missing_dep_xyz\n"

# Test patch touching only non-test code: a pure-creation hunk
# adding a helper function to a new non-test file. No test
# function is added or modified, so the task must skip — with
# the new diagnostic naming what the patch touched.
PATCH_ADD_HELPER_ONLY = "\n".join([
    "--- a/tests/util.py",
    "+++ b/tests/util.py",
    "@@ -0,0 +1,2 @@",
    "+def helper():",
    "+    return 1",
    "",
])


class TestFifthRoundFixes(unittest.TestCase):
    """Fifth-round converter fixes (1 Nov 2026, FINAL round before
    the paper window — scoped to collection only):
    (1) whole-tree discovery replaced by per-module collection,
    so a broken unrelated module no longer kills the task;
    (2) dropped candidates carry the probed collection error;
    (3) the no-test-function skip names what the patch touched."""

    def _run(self, tmpdir: Path, tasks, snaps) -> tuple[
            int, str, str, list[dict]]:
        records = [_record(tid, gold, patch)
                   for tid, gold, patch in tasks]
        tasks_path = tmpdir / "tasks.jsonl"
        tasks_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        out_path = tmpdir / "dev_tasks.jsonl"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = build_dev_tasks.main([
                "convert", "--tasks", str(tasks_path),
                "--snapshots", str(snaps), "--out", str(out_path),
                "--python", sys.executable,
                "--env-label", "test-fixture",
            ])
        recs = [json.loads(line) for line in
                out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if out_path.exists() else []
        return code, stdout.getvalue(), stderr.getvalue(), recs

    def test_unrelated_broken_module_does_not_kill_task(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r5-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
                "tests/test_other.py": CALC_TESTS_OTHER,
                # Unrelated broken module (the fastapi benchmarks
                # shape): whole-tree discovery would yield a
                # _FailedTest here and skip the task.
                "tests/benchmarks/test_bench.py":
                    CALC_TESTS_BENCH_BROKEN,
            }
            (snaps / "tgood.tgz").write_bytes(_make_tgz(tree))
            code, out, err, recs = self._run(tmpdir, [
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            self.assertEqual(rec["failing_test_id"],
                             "tests.test_calc.TestCalc.test_add")
            # Regressions come from the fine modules only; the
            # broken module's test is absent, not fatal.
            self.assertIn("tests.test_calc.TestCalc.test_mul",
                          rec["regression_test_ids"])
            self.assertIn("tests.test_other.TestOther.test_other",
                          rec["regression_test_ids"])
            self.assertNotIn("tests.benchmarks.TestBench.test_bench",
                             rec["regression_test_ids"])
            self.assertIn("converted 1/1 tasks", out)

    def test_candidate_module_import_failure_names_error(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r5-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            tree = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BAD_IMPORT,
            }
            (snaps / "tbad.tgz").write_bytes(_make_tgz(tree))
            (snaps / "tgood.tgz").write_bytes(_make_tgz({
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }))
            code, out, err, recs = self._run(tmpdir, [
                ("tbad", GOLD_PATCH, PATCH_ADD_ONE),
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual([r["task_id"] for r in recs], ["tgood"])
            # The skip names the underlying import error, not
            # just the bare dropped id.
            self.assertIn("SKIP task tbad", err)
            self.assertIn("definitely_missing_dep_xyz", err)
            self.assertIn("ModuleNotFoundError", err)
            self.assertIn("converted 1/2 tasks", out)

    def test_non_test_patch_skip_names_what_it_touched(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="build-dev-r5-") as td:
            tmpdir = Path(td)
            snaps = tmpdir / "snapshots"
            snaps.mkdir()
            base = {
                "calc.py": CALC_BUGGY,
                "tests/__init__.py": "",
                "tests/test_calc.py": CALC_TESTS_BASE,
            }
            (snaps / "tnontest.tgz").write_bytes(_make_tgz(base))
            (snaps / "tgood.tgz").write_bytes(_make_tgz(base))
            code, out, err, recs = self._run(tmpdir, [
                ("tnontest", GOLD_PATCH, PATCH_ADD_HELPER_ONLY),
                ("tgood", GOLD_PATCH, PATCH_ADD_ONE),
            ], snaps)
            self.assertEqual(code, 0, err)
            self.assertEqual([r["task_id"] for r in recs], ["tgood"])
            # The skip distinguishes "touches no test code" from
            # "enumeration missed it": the diagnostic lists the
            # touched file and the function found in it.
            self.assertIn("SKIP task tnontest", err)
            self.assertIn("test_patch adds or modifies no test "
                          "function", err)
            self.assertIn("tests/util.py", err)
            self.assertIn("helper", err)
            self.assertIn("converted 1/2 tasks", out)


if __name__ == "__main__":
    unittest.main()
