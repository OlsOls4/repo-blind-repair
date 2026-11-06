"""Convert the Gemma 4 public dev set into eval_dev.py's JSONL format.

STATUS — READ FIRST
-------------------
The dev set's HOME is established (see DEV_SET_SOURCING.md): it ships
inside the competition's Data tab dataset (22.42 GB, Apache 2.0),
alongside repository snapshots, code graphs, embeddings, offline
wheels, the Docker sandbox spec, and a sample_submission folder. That
dataset is only obtainable inside Kaggle by a joined competitor — it
has NO verified public mirror.

The schema below is VERIFIED, not assumed: on 1 Nov 2026 a
Kaggle-side --inspect run (key names + value types of the first
record only, never contents) established the real layout:

  Top level: HARNESS_README.md, tasks.jsonl, docker/ (4 files),
  embeddings/ (127 files), graphs/ (127 files), sample_submission/
  (adapters/ with main_lora/ and tool_lora/, configs/, prompts/,
  sub_agents/), sandbox/ (1 file), snapshots/ (129 .tgz archives,
  one per dev task), wheels/ (124 files); total 524 files,
  ~22.42 GB.

  First-record schema of tasks.jsonl (key names + types only):
  instance_id: str, repo: str, base_commit: str, patch: str,
  test_patch: str, problem_statement: str, hints_text: str,
  created_at: str.

  There are NO FAIL_TO_PASS / PASS_TO_PASS keys. The gold patch is
  in `patch` and the test changes in `test_patch`. The earlier
  SWE-bench-style conversion mode (which assumed those keys) has
  therefore been REMOVED and replaced by the `convert` mode below.
  Do not resurrect it: running real records through the old mode
  would abort on the missing keys at best and mis-convert at worst.

This script has TWO modes, used in order:

1.  --inspect <data-dir>
    Kaggle-side discovery. Prints the dataset tree (top 3 levels,
    file counts and sizes) and, for every *.jsonl / *.json file, the
    KEY NAMES and value TYPES of the first record only — never task
    contents. Run this first on the attached competition data and
    compare its output against the schema above. If the schema
    differs, STOP and adapt the converter; do not force the data
    through a wrong schema.

2.  convert --tasks <tasks.jsonl> --snapshots <dir> --out <file>
    Converts the VERIFIED schema into eval_dev.py's JSONL format.
    Design decisions (each documented here and in
    DEV_SET_SOURCING.md):

    (a) REPO FILES come from extracting the per-task snapshot .tgz
        at base_commit. The snapshot archive for a task is
        <snapshots>/<instance_id>.tgz; if that name is wrong for
        the real layout, the converter aborts naming the expected
        file — check the name with --inspect, do not guess a
        different one. If the archive contains a single top-level
        directory, it is stripped (the usual "repo at a commit"
        tarball shape); otherwise the archive root is the repo
        root. The snapshot is taken as the base_commit state as
        shipped; the converter cannot verify the commit hash from
        the tarball, and does not claim to.
        The gold `patch` is NEVER applied and never written to
        the output: applying it would fix the bug before
        evaluation. The snapshot stays in its buggy state.

    (b) FAILING TEST ID is DERIVED by running the tests, never
        invented (regression-derivation fix, 1 Nov 2026). The
        converter parses the test diff, applies it to the snapshot
        (strict unified-diff application: every context/removal
        line must match exactly, or the task is skipped — see
        Strictness below), then classifies the test_patch's test
        changes by AST comparison of the old vs patched trees:
        ADDED ids (present only in the patched tree) and MODIFIED
        ids (present in both, function source text differs). This
        replaces the old static derivation, which skipped every
        task whose test_patch modified existing tests or added
        several tests. Each candidate is then validated against
        REAL collection in a scratch copy of the patched tree
        (non-collecting ids are dropped and counted, with the
        underlying collection error probed and recorded per id),
        run on the buggy code (must FAIL — a candidate that
        passes on buggy code is not a failing test), and run on
        gold-patched code in a second scratch copy (must PASS).
        SCOPED COLLECTION (round 5, 1 Nov 2026): collection runs
        per module, only for modules containing a candidate or a
        regression candidate — never whole-tree. The old
        whole-tree run skipped a task whenever ANY module failed
        to import, which killed all 10 fastapi subset tasks on one
        unrelated broken module (benchmarks, ~300 modules
        discovered). A failed module now only removes its own ids:
        every id is validated by running, so partial collection
        cannot mislabel anything.
        The gold check is the guard against buggy-only
        mislabelling: a test that fails on buggy code because the
        environment is broken (missing dependency, bad import)
        also fails on gold code, and is dropped instead of being
        recorded as FAIL_TO_PASS. A task converts ONLY if exactly
        one candidate survives both runs; zero or several
        survivors means the task is SKIPPED with a counted reason
        on stderr. It is never given a guessed id.

    (c) REGRESSION TEST IDS are DERIVED the same way (1 Nov 2026;
        previously left EMPTY, which made every converted task's
        fix rate 0 by construction). Candidates are the
        pre-existing enumerated tests in the scratch tree,
        excluding added/modified ids, ordered with tests in
        test_patch-touched files first, then gold-patch-touched
        files, then the rest, each group in sorted id order (a
        fixed seed, no randomness); each module is collected on
        demand in that order until at most REGRESSION_POOL (60)
        collected candidates accumulate, then re-sorted and
        capped. The pool differs from whole-tree collection only
        in which ids are preferred when it overflows; every
        pooled id is still validated by running. A candidate is kept only
        if it passes on BOTH buggy and gold code, capped at 15
        (the harness gate's cap). A regression that passes on
        buggy but fails on gold would be broken by the fix itself,
        so it is not a regression. A task with no surviving
        regression is SKIPPED (counted), not converted with an
        empty list: an empty list would reintroduce the 0-by-
        construction result for that task. The gold patch is
        applied ONLY inside the gold scratch copy and NEVER
        written to the output: the output `files` stay the buggy
        snapshot + test_patch, exactly as eval_dev.py evaluates.
        Environment fidelity: tests run under --python, and the
        environment is recorded per record in the output
        `derivation.env` field (default: the local interpreter,
        explicitly NOT the dataset docker spec). On Kaggle, run
        under the dataset's docker spec / offline wheels and pass
        --env-label accordingly; never claim scorer fidelity the
        run did not have.

    (d) BUG CLASS is `unknown` for every real task: the verified
        schema carries no bug-class label, and eval_dev.py already
        handles `unknown` (it is the expected value for hidden
        real bugs).

    (e) BINARY-SKIP ALLOWLIST (1 Nov 2026; see _extract_snapshot).
        Real snapshots are git checkouts, so they contain .git/
        binary files (first offender on the first real-data
        attempt: .git/index), plus binary assets such as images.
        Members under any .git/ directory (any depth) are skipped
        without being read — VCS metadata can never be an
        evaluated test target — and any other non-UTF-8 member
        is skipped too, but NEVER silently: every skipped path
        is listed, sorted and deduplicated, in the output record
        as `skipped_binary_files`, with a per-path reason in
        `skipped_binary_reasons` ("git" / "binary" /
        "oversized"), and the summary prints the totals. A task
        is never aborted for binary content alone. The
        allowlist's limit is explicit: if a future task's
        actually-evaluated source ever were binary, the record's
        skipped list makes that visible instead of hidden —
        check the list on the Kaggle re-run, do not assume it
        only ever holds .git/ paths.

    (f) OVERSIZED MEMBERS (1 Nov 2026, second subset fix; see
        _extract_snapshot). A member over MAX_SNAPSHOT_FILE_BYTES
        (2 MB) no longer aborts the task: it is skipped, unread,
        with reason "oversized" (the second subset run lost 10
        tasks to logo .ai/.gif assets this way). Guard: an
        oversized .py/.pyi member aborts the task as a counted
        SKIP ("oversized evaluated source") — converting with a
        hole in evaluated code would corrupt the measurement.
        .git/ members never trigger the guard (checked first).
        LIMIT: the guard covers .py/.pyi only; an oversized
        .json/.txt fixture a test actually reads would be
        skipped as "oversized" and listed — visible, but check
        the list on the re-run.

    (g) ADDED TEST FILES (1 Nov 2026, second subset fix; see
        _apply_diff). A test_patch section whose --- path names
        a file absent from the snapshot is treated as a file
        creation ONLY when every hunk is a pure creation (all
        '+' lines, old_len 0, old_start 0 or 1) — the normal
        SWE-bench case of a test_patch ADDING a new test file
        (7 tasks on the second subset run). The missing file is
        the empty baseline for classification, so every test in
        the created file is "added", and it goes through the
        same real-collection / FAIL-on-buggy / PASS-on-gold
        validation and non-collecting drop rule as any other
        candidate. Anything else against a missing file (context
        / removal lines, or a pure-addition append anchored past
        line 1) still skips: guessing that baseline would risk
        silently dropping original file content.

    (h) CONTEXT-MISMATCH DIAGNOSTICS (1 Nov 2026, second subset
        fix; see _mismatch_diagnostic). Strict application is
        UNCHANGED — no fuzzy matching, a mismatching task still
        skips. What changed is instrumentation: on any context
        or removal mismatch (3 tasks on the second subset run:
        one gold-patch hunk, two test_patch hunks), the converter
        prints to stderr AND repeats in the run summary the task
        id, the file, the hunk location, the expected hunk
        lines, the actual snapshot lines in a ±5-line window,
        and a flag for the already-applied hypothesis (whether
        the hunk's added lines already exist in the snapshot
        file, anywhere and contiguously). That flag is a
        hypothesis to test on the Kaggle re-run — a true flag
        would indicate the snapshot already contains the
        test_patch — NOT an assumption to code around: a
        mismatching task still skips even when the flag is true.

    issue_text is problem_statement, with hints_text appended
    (labelled "Hints:") when non-empty — both are free text for
    the trace extractor, and neither is a test id or a patch.

Strictness mirrors eval_dev.py, split in two tiers since 1 Nov
2026. FILE-LEVEL problems abort the whole conversion: malformed
JSON, a record missing a required field or carrying a non-string
field, a duplicate task_id, or an empty output (no records at
all) — these indicate corrupt input or a schema mismatch, and a
half-converted task file must never produce a half-credible
report. PER-TASK derivation problems skip ONLY that task, counted
and listed on stderr, and the run continues: missing snapshot,
unsafe snapshot member, corrupt/unreadable archive,
unappliable test_patch or gold hunk (with the (h) diagnostics
on stderr and in the summary),
test-collection failure, zero/ambiguous FAIL_TO_PASS survivors,
no surviving regression, an oversized .py/.pyi snapshot member,
or a test_patch against a missing file that is not a pure
creation (see (g)). Binary and oversized non-source snapshot
members are NOT a skip reason: they are skipped and listed per
(e)/(f) above. A skipped task is never in the output file, and
the summary prints converted/skipped counts plus FAIL_TO_PASS
and PASS_TO_PASS coverage (converted tasks / tasks seen),
skipped-binary totals and reasons, and any context-mismatch
diagnostics, so a partial conversion is visible, not hidden.

PROVENANCE OF THIS VERSION (read before trusting it)
----------------------------------------------------
This converter is proven ONLY against HAND-BUILT fixtures that
mimic the verified schema (tasks.jsonl first-record keys + tiny
.tgz snapshots + sample test_patches, now including gold-pass/fail
combinations, modified tests, non-collecting ids, broken tasks,
binary .git/ members and binary assets, a corrupt tar, added
test files, oversized binary assets, oversized .py/.pyi members,
and context-mismatch diagnostics — see test_build_dev_tasks.py).
It has been run TWICE against real data, both times a 20-task
subset on CPU, 1 Nov 2026, both converting 0/20. Run 1: every
snapshot is a git checkout and the pre-allowlist code aborted on
.git/index; that failure is what the binary-skip allowlist (e)
above fixes. Run 2 (with the allowlist; no .git/ skips in the
log): 7 tasks skipped because test_patch adds a file absent from
the snapshot, 10 tasks aborted on oversized logo .ai/.gif
assets, 3 tasks hit context mismatches (one gold-patch hunk, two
test_patch hunks). Those three causes are what (f), (g) and (h)
above fix / instrument. The Kaggle-side re-run must therefore go
SUBSET FIRST (the same 20-task subset, or a fresh small subset),
and its report must state converted/skipped counts,
FAIL_TO_PASS/PASS_TO_PASS coverage, skipped-binary totals AND
reasons, and — for every context-mismatch skip — the diagnostic's
added-lines-already-in-snapshot flag (the converter prints them;
check them, do not assume 129/129). Run the full 129-task
conversion ONLY after that subset coverage is stated:

  python build_dev_tasks.py convert --tasks <tasks.jsonl> \
      --snapshots <snapshots-dir> --out dev_tasks.jsonl
  python eval_dev.py --dev-tasks dev_tasks.jsonl \
      --adapter-dir <saved-adapter-dir> --dry-run

Save the notebook VERSION with output immediately afterwards. A
dev-tasks file that exists only in a non-persistent working
directory does not exist (see the 31 Oct adapter loss).

This script never touches hidden/private leaderboard tasks. It only
converts the PUBLIC dev set (129 tasks, fastapi/rich/requests/httpx).
Do NOT point it at any private-task data.

Usage on Kaggle (competition notebook, internet OFF):
  1. Attach the competition data (Data tab dataset) as a notebook
     input, plus this script.
  2. python build_dev_tasks.py --inspect /kaggle/input/<data-dir>
     (replace <data-dir> with the attached dataset's directory name;
     list /kaggle/input to find it).
  3. Run mode 2 with the real tasks file and snapshots dir from
     step 2, then validate with eval_dev.py --dry-run (above).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# eval_dev.py's path rules, duplicated here so conversion fails the
# same way evaluation would: relative paths only, no '..', no ''.
def _is_safe_rel_path(path: str) -> bool:
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    parts = path.replace("\\", "/").split("/")
    return all(p not in ("", ".", "..") for p in parts)


def _abort(msg: str) -> None:
    raise SystemExit(f"[build-dev] ABORT: {msg}")


class _TaskSkip(Exception):
    """Per-task derivation failure: skip + count + list, never abort.

    Raised by the derivation stage (snapshot/diff/test-run problems
    for one task). cmd_convert catches it, records the task in the
    skipped list, and continues with the next task. File-level
    problems (unreadable tasks file, schema mismatch, duplicate
    task_id) still abort via _abort: they indicate corrupt input,
    not a bad task.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Mode 1: inspect (UNCHANGED — do not edit this function)
# ---------------------------------------------------------------------------

def cmd_inspect(data_dir: Path) -> int:
    if not data_dir.is_dir():
        _abort(f"data dir {data_dir} is not a directory")
    print(f"[build-dev] inspecting {data_dir}")
    total_files = 0
    total_bytes = 0
    for root in sorted(data_dir.iterdir()):
        if root.is_dir():
            entries = list(root.rglob("*"))
            files = [p for p in entries if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            total_files += len(files)
            total_bytes += size
            print(f"  DIR  {root.name}/  files={len(files)}  bytes={size}")
            # One level deeper, counts only (no contents).
            for sub in sorted(root.iterdir()):
                if sub.is_dir():
                    sub_files = [p for p in sub.rglob("*") if p.is_file()]
                    print(f"    DIR  {root.name}/{sub.name}/  "
                          f"files={len(sub_files)}")
                else:
                    print(f"    FILE {root.name}/{sub.name}  "
                          f"bytes={sub.stat().st_size}")
        else:
            print(f"  FILE {root.name}  bytes={root.stat().st_size}")
            total_files += 1
            total_bytes += root.stat().st_size
    print(f"[build-dev] total: files={total_files} bytes={total_bytes}")
    print("[build-dev] task-file schemas (key names + value types of "
          "first record only; no contents shown):")
    found = False
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".jsonl", ".json"):
            continue
        if path.stat().st_size > 200_000_000:
            print(f"  SKIP {path.relative_to(data_dir)} (>{200} MB, "
                  "inspect by hand)")
            continue
        try:
            first = None
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            first = json.loads(line)
                            break
            else:
                obj = json.loads(path.read_text(encoding="utf-8"))
                first = obj[0] if isinstance(obj, list) and obj else obj
        except (OSError, ValueError) as exc:
            print(f"  UNREADABLE {path.relative_to(data_dir)}: {exc}")
            continue
        if not isinstance(first, dict):
            print(f"  {path.relative_to(data_dir)}: first record is "
                  f"not an object ({type(first).__name__})")
            continue
        found = True
        schema = {k: type(v).__name__ for k, v in first.items()}
        print(f"  {path.relative_to(data_dir)}: {json.dumps(schema)}")
    if not found:
        print("  (no small .jsonl/.json task file found at top levels — "
              "tasks may live inside per-task directories; inspect the "
              "tree above and adapt the converter)")
    return 0


# ---------------------------------------------------------------------------
# Mode 2: verified-schema conversion
# ---------------------------------------------------------------------------

MAX_SNAPSHOT_FILE_BYTES = 2_000_000  # refuse to inline larger files

# Verified first-record keys of tasks.jsonl; all must be strings.
REQUIRED_STR_FIELDS = (
    "instance_id", "repo", "base_commit", "patch", "test_patch",
    "problem_statement", "hints_text", "created_at",
)

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# NOTE: the old static derivation's _CLASS_RE / _DEF_RE (mapping a
# diff-recorded def line to a dotted id by indentation walking) were
# removed with it on 1 Nov 2026; test classification now uses AST
# comparison (_classify_test_patch) plus real unittest collection.


def _clean_diff_path(raw: str, task_id: str) -> str | None:
    """Normalise a --- / +++ path: None for /dev/null, else repo path.

    Strips one leading a/ or b/ prefix. Quoted or otherwise exotic
    paths abort: guessing a filename mapping would corrupt the tree.
    """
    raw = raw.strip()
    if raw == "/dev/null":
        return None
    if raw.startswith('"'):
        _abort(f"task {task_id}: quoted diff path {raw!r} is not "
               "handled; refusing to guess the filename")
    for prefix in ("a/", "b/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if not _is_safe_rel_path(raw):
        _abort(f"task {task_id}: unsafe diff path {raw!r}")
    return raw


class _FileDiff:
    """One file section of a unified diff, with added-test positions."""

    def __init__(self, old_path: str | None, new_path: str | None):
        self.old_path = old_path  # None => file is created
        self.new_path = new_path  # None => file is deleted
        # hunks: (old_start, old_len, new_start, new_len,
        #         [(tag, text), ...]) with tag in ' ', '-', '+'.
        self.hunks: list[tuple[int, int, int, int, list[tuple[str, str]]]] = []
        # NOTE: test functions are classified by AST comparison of
        # the old vs patched trees (_classify_test_patch), NOT from
        # diff-recorded def lines: the old static derivation that
        # tracked added def positions here was superseded by the
        # 1 Nov 2026 run-based derivation and has been removed.


def _parse_unified_diff(test_patch: str, task_id: str) -> list[_FileDiff]:
    """Parse a unified diff strictly; abort on anything malformed.

    Binary diffs, missing ---/+++ headers, bad hunk headers, and
    hunk line-count mismatches all abort: a half-parsed test_patch
    must never produce a half-patched tree.
    """
    if not test_patch.strip():
        return []
    lines = test_patch.split("\n")
    sections: list[_FileDiff] = []
    cur: _FileDiff | None = None
    cur_hunk: list[tuple[str, str]] | None = None
    cur_hunk_meta: tuple[int, int, int, int] | None = None
    i = 0

    def hunk_done() -> bool:
        if cur_hunk is None or cur_hunk_meta is None:
            return True
        _os, old_len, _ns, new_len = cur_hunk_meta
        old_count = sum(1 for tag, _ in cur_hunk if tag in (" ", "-"))
        new_count = sum(1 for tag, _ in cur_hunk if tag in (" ", "+"))
        return old_count >= old_len and new_count >= new_len

    def close_hunk() -> None:
        nonlocal cur_hunk, cur_hunk_meta
        if cur_hunk is None or cur is None or cur_hunk_meta is None:
            return
        old_start, old_len, new_start, new_len = cur_hunk_meta
        old_count = sum(1 for tag, _ in cur_hunk if tag in (" ", "-"))
        new_count = sum(1 for tag, _ in cur_hunk if tag in (" ", "+"))
        if old_count != old_len or new_count != new_len:
            _abort(f"task {task_id}: hunk @@ -{old_start},{old_len} "
                   f"+{new_start},{new_len} @@ has {old_count} old / "
                   f"{new_count} new lines; diff is malformed")
        cur.hunks.append((old_start, old_len, new_start, new_len,
                          cur_hunk))
        cur_hunk = None
        cur_hunk_meta = None

    while i < len(lines):
        # A hunk is closed as soon as its declared line counts are
        # satisfied. This matters: a removed line whose content
        # starts with "--" must be consumed as hunk content, never
        # misread as a "---" section header.
        if cur_hunk is not None and hunk_done():
            close_hunk()
        line = lines[i]
        if line.startswith("diff --git "):
            if cur_hunk is not None:
                _abort(f"task {task_id}: hunk closed early by a "
                       "diff header (line {i + 1}); diff malformed")
            cur = None
            i += 1
            continue
        if line.startswith("Binary files "):
            _abort(f"task {task_id}: binary diff in test_patch is "
                   "not handled")
        if line.startswith("--- ") and cur_hunk is None:
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                _abort(f"task {task_id}: diff section at line {i + 1} "
                       "has a --- header without a +++ header")
            old_path = _clean_diff_path(line[4:], task_id)
            new_path = _clean_diff_path(lines[i + 1][4:], task_id)
            if old_path is None and new_path is None:
                _abort(f"task {task_id}: diff section with two "
                       "/dev/null headers is malformed")
            cur = _FileDiff(old_path, new_path)
            sections.append(cur)
            i += 2
            continue
        m = _HUNK_RE.match(line)
        if m and cur_hunk is None:
            if cur is None:
                _abort(f"task {task_id}: hunk header without a file "
                       "section (line {i + 1})")
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            cur_hunk = []
            cur_hunk_meta = (old_start, old_len, new_start, new_len)
            i += 1
            continue
        if cur_hunk is not None:
            if line.startswith("\\ "):
                # "\ No newline at end of file": the converter
                # normalises the final newline (see _apply_diff);
                # test semantics are unaffected. Skip the marker.
                i += 1
                continue
            if line == "":
                # Some diff generators emit an empty context line
                # bare (no leading space). Treat it as context.
                cur_hunk.append((" ", ""))
                i += 1
                continue
            if line[0] in (" ", "-", "+"):
                cur_hunk.append((line[0], line[1:]))
                i += 1
                continue
            _abort(f"task {task_id}: unexpected line inside a hunk "
                   f"(line {i + 1}): {line[:60]!r}")
        if line and line[0] in (" ", "-", "+"):
            _abort(f"task {task_id}: diff content line outside any "
                   f"hunk (line {i + 1}); diff malformed")
        # Lines outside any hunk/section (index lines, mode lines,
        # "new file mode", etc.) are informational only.
        i += 1
    if cur_hunk is not None and not hunk_done():
        _abort(f"task {task_id}: diff ends inside an incomplete hunk")
    close_hunk()
    return sections


# Context-mismatch diagnostics, collected during a convert run and
# printed in the run summary (see cmd_convert). Each entry is the
# full diagnostic text for one mismatching hunk. Reset at the start
# of cmd_convert so repeated main() calls in one process (unit
# tests) do not leak diagnostics across runs.
_MISMATCH_DIAGNOSTICS: list[str] = []


def _mismatch_diagnostic(task_id: str, patch_label: str, path: str,
                         old_start: int, pos: int,
                         current: list[str],
                         hunk: list[tuple[str, str]],
                         kind: str, expected: str) -> str:
    """Build the diagnostic for one context/removal mismatch.

    Strictness is NOT relaxed: the caller still aborts (the task
    skips). This only makes the failure diagnosable. The diagnostic
    carries the task id, the file, the hunk location, the expected
    hunk lines, the actual snapshot lines in a window of ±5 lines
    around the mismatch, and a flag for the already-applied
    hypothesis: whether the hunk's non-empty added lines already
    exist in the snapshot file (anywhere, and as one contiguous
    block). If that flag is true, the snapshot may already contain
    the patch — a hypothesis to test on the Kaggle re-run, NOT an
    assumption to code around: the task still skips.
    """
    lo = max(0, pos - 5)
    hi = min(len(current), pos + 6)
    actual = "\n".join(
        f"    {i + 1}: {current[i][:120]}" for i in range(lo, hi))
    hunk_shown = hunk[:40]
    hunk_text = "\n".join(f"    {tag} {text[:120]}"
                           for tag, text in hunk_shown)
    if len(hunk) > 40:
        hunk_text += f"\n    ... ({len(hunk) - 40} more hunk lines)"
    added = [text for tag, text in hunk if tag == "+"]
    nonempty = [t for t in added if t.strip()]
    if not nonempty:
        flag = ("added-lines-already-in-snapshot: n/a "
                "(no non-empty added lines in this hunk)")
    else:
        curset = set(current)
        anywhere = all(t in curset for t in nonempty)
        contiguous = False
        for i in range(len(current) - len(added) + 1):
            if current[i:i + len(added)] == added:
                contiguous = True
                break
        flag = (f"added-lines-already-in-snapshot: anywhere={anywhere} "
                f"contiguous={contiguous}")
    diag = (
        f"[build-dev] CONTEXT MISMATCH task {task_id} "
        f"({patch_label}): {path!r} hunk @@ -{old_start} @@, "
        f"{kind} at snapshot line {pos + 1}\n"
        f"  expected: {expected[:120]!r}\n"
        f"  expected hunk lines:\n{hunk_text}\n"
        f"  actual snapshot lines {lo + 1}-{hi} (±5 around line "
        f"{pos + 1}):\n{actual if actual else '    (file is empty)'}\n"
        f"  {flag}")
    _MISMATCH_DIAGNOSTICS.append(diag)
    return diag


def _apply_diff(tree: dict[str, str], sections: list[_FileDiff],
                task_id: str, patch_label: str = "test_patch"
                ) -> dict[str, str]:
    """Apply parsed diff sections to a snapshot tree, strictly.

    Every context/removal line must match the current file exactly
    at the hunk position; any mismatch aborts (the task skips).
    No fuzzy matching, no silent skipping: a patch that does not
    apply cleanly to its snapshot is a schema/layout mismatch, not
    something to paper over. Final newlines are normalised to one
    trailing "\\n" (test semantics are unaffected).

    ADDED-FILE RULE (1 Nov 2026, second subset fix): a section
    whose --- path names a file that is NOT in the snapshot is
    treated as a file creation ONLY when every hunk is a pure
    creation — all '+' lines, old_len 0, old_start 0 or 1 (the
    signature of a new-file diff). The missing file is then the
    empty baseline for classification: every test in the created
    file is "added" (see _classify_test_patch), and it goes
    through the same collection / FAIL-on-buggy / PASS-on-gold
    validation as any other candidate. Anything else against a
    missing file aborts: a hunk with context/removal lines, or a
    pure-addition hunk anchored at old line > 1 (an append to a
    file whose content we do not have — creating it from the
    added lines alone would silently drop the original content),
    is NOT a creation and must not be guessed into one.

    HUNK COORDINATES (1 Nov 2026, third subset fix): hunk headers
    are in OLD-file coordinates, and `current` below is the
    UNMODIFIED snapshot content — so each hunk is located at
    old_start - 1, with NO cumulative offset. An earlier version
    of this function added a running new-minus-old delta to each
    later hunk's position while still indexing into the original
    `current` array: that mixed coordinate systems and shifted
    every later hunk by exactly the net lines added/removed
    before it (observed on the third subset run as mismatches at
    +23/+14/+10 lines after insertions and -3 after deletions).
    The delta variable was therefore REMOVED, not supplemented:
    there is no second offset mechanism, because with an
    immutable old file there is nothing to offset. Strict
    context matching is unchanged.
    """
    out = dict(tree)
    for fd in sections:
        if fd.new_path is None:
            # File deleted by the patch.
            if fd.old_path not in out:
                _abort(f"task {task_id}: {patch_label} deletes "
                       f"{fd.old_path!r}, which is not in the snapshot")
            del out[fd.old_path]
            continue
        if fd.old_path is None:
            # File created by the patch (--- /dev/null): hunks may
            # only add.
            if fd.new_path in out:
                _abort(f"task {task_id}: {patch_label} creates "
                       f"{fd.new_path!r}, which already exists in "
                       "the snapshot")
            created: list[str] = []
            for _os, _ol, _ns, _nl, hunk in fd.hunks:
                for tag, text in hunk:
                    if tag != "+":
                        _abort(f"task {task_id}: created file "
                               f"{fd.new_path!r} has a non-addition "
                               "hunk line; refusing to guess")
                    created.append(text)
            out[fd.new_path] = "\n".join(created) + ("\n" if created else "")
            continue
        if fd.old_path not in out:
            # Missing target: creation ONLY for pure-creation
            # hunks (see docstring above); otherwise abort.
            if fd.old_path != fd.new_path:
                _abort(f"task {task_id}: {patch_label} renames "
                       f"{fd.old_path!r} to {fd.new_path!r}; renames "
                       "are not handled")
            is_creation = bool(fd.hunks) and all(
                all(tag == "+" for tag, _ in hunk)
                and old_len == 0 and old_start in (0, 1)
                for old_start, old_len, _ns, _nl, hunk in fd.hunks)
            if not is_creation:
                _abort(f"task {task_id}: {patch_label} modifies "
                       f"{fd.old_path!r}, which is not in the "
                       "snapshot, and its hunks are not a pure "
                       "file creation (context/removal lines, or "
                       "an append anchored past line 1); refusing "
                       "to guess the baseline")
            created = []
            for _os, _ol, _ns, _nl, hunk in fd.hunks:
                created.extend(text for tag, text in hunk if tag == "+")
            out[fd.new_path] = "\n".join(created) + ("\n" if created else "")
            continue
        # The --- and +++ paths may differ only by the a//b/ prefix;
        # if they genuinely differ, that is a rename, which the
        # converter does not handle (abort, do not guess).
        if fd.old_path != fd.new_path:
            _abort(f"task {task_id}: {patch_label} renames "
                   f"{fd.old_path!r} to {fd.new_path!r}; renames "
                   "are not handled")
        current = out[fd.old_path].splitlines()
        result: list[str] = []
        cur = 0  # next unprocessed line of `current` (old coords)
        for old_start, _ol, _ns, _nl, hunk in fd.hunks:
            # Old coordinates, no delta: `current` is never mutated
            # (see docstring, HUNK COORDINATES).
            start = old_start - 1
            if start < 0 or start > len(current):
                diag = _mismatch_diagnostic(
                    task_id, patch_label, fd.old_path, old_start,
                    min(max(start, 0), len(current)), current, hunk,
                    "hunk-start-outside-file",
                    f"hunk starts at old line {old_start}")
                print(diag, file=sys.stderr)
                _abort(f"task {task_id}: hunk at old line "
                       f"{old_start} in {fd.old_path!r} starts "
                       "outside the file; full diagnostic on stderr")
            result.extend(current[cur:start])
            pos = start
            for tag, text in hunk:
                if tag == " ":
                    if pos >= len(current) or current[pos] != text:
                        diag = _mismatch_diagnostic(
                            task_id, patch_label, fd.old_path,
                            old_start, pos, current, hunk,
                            "context mismatch", text)
                        print(diag, file=sys.stderr)
                        _abort(f"task {task_id}: context mismatch in "
                               f"{fd.old_path!r} at line {pos + 1}: "
                               f"expected {text[:60]!r}; full "
                               "diagnostic on stderr")
                    result.append(text)
                    pos += 1
                elif tag == "-":
                    if pos >= len(current) or current[pos] != text:
                        diag = _mismatch_diagnostic(
                            task_id, patch_label, fd.old_path,
                            old_start, pos, current, hunk,
                            "removal mismatch", text)
                        print(diag, file=sys.stderr)
                        _abort(f"task {task_id}: removal mismatch in "
                               f"{fd.old_path!r} at line {pos + 1}: "
                               f"expected {text[:60]!r}; full "
                               "diagnostic on stderr")
                    pos += 1
                else:  # '+'
                    result.append(text)
            cur = pos
        result.extend(current[cur:])
        out[fd.old_path] = "\n".join(result) + ("\n" if result else "")
    return out


# ---------------------------------------------------------------------------
# Regression derivation (1 Nov 2026): FAIL_TO_PASS / PASS_TO_PASS by
# actually running the tests, in scratch copies, never by guessing.
# ---------------------------------------------------------------------------

# Mirrors harness test_runner.DEFAULT_MAX_REGRESSION_TESTS: the gate
# caps the regression suite at 15 tests, so deriving more is waste.
MAX_REGRESSION_TESTS = 15
# Pre-existing candidates considered, in preference order (touched
# files first, then id order — a fixed seed, no randomness), before
# the buggy+gold pass filter below. Bounds the subprocess cost of a
# repo with hundreds of pre-existing tests.
REGRESSION_POOL = 60
# unittest discovery (importing every test module) gets longer than
# a single test run.
COLLECT_TIMEOUT_SECS = 120


def _run_unittest(python: str, test_id: str, cwd: str,
                  timeout_secs: int) -> bool:
    """Run one unittest id in a subprocess; True only on exit 0.

    Fail, error, and timeout all count as not-passed: unittest exits
    0 only when the test passes. A missing interpreter is checked
    once in cmd_convert, so OSError here is treated as not-passed
    rather than aborting a whole conversion on one task.
    """
    try:
        proc = subprocess.run(
            [python, "-m", "unittest", "-v", test_id],
            cwd=cwd, capture_output=True, text=True,
            timeout=timeout_secs)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


_COLLECTOR_SRC = (
    "import json, sys, unittest\n"
    "loader = unittest.TestLoader()\n"
    "suite = loader.discover(start_dir=sys.argv[1],\n"
    "                        pattern=sys.argv[2],\n"
    "                        top_level_dir=sys.argv[3])\n"
    "ids, failed = [], []\n"
    "def walk(s):\n"
    "    for t in s:\n"
    "        if isinstance(t, unittest.TestSuite):\n"
    "            walk(t)\n"
    "        else:\n"
    "            tid = t.id()\n"
    "            if tid.startswith('unittest.loader._FailedTest'):\n"
    "                failed.append(tid)\n"
    "            else:\n"
    "                ids.append(tid)\n"
    "walk(suite)\n"
    "print(json.dumps({'ids': sorted(set(ids)), 'failed': failed}))\n"
)


def _pytest_available(python: str) -> bool:
    """True when `python -m pytest` imports in the test environment.

    Checked against the --python interpreter, not this process:
    on Kaggle that is the notebook interpreter with the dataset's
    dependencies. A missing/slow pytest is False, never an abort:
    the converter then falls back to unittest-only collection and
    module-level candidates are dropped as non-collecting (the
    pre-fix behaviour), not guessed.
    """
    try:
        proc = subprocess.run(
            [python, "-c", "import pytest"], capture_output=True,
            text=True, timeout=30)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


_PYTEST_ID_RE = re.compile(r"^(.+\.py)::([A-Za-z_][A-Za-z0-9_]*)$")


def _collect_pytest_ids(python: str, repo_dir: str,
                        target: str | None = None) -> list[str]:
    """Collect module-level pytest ids via `pytest --collect-only`.

    With target=None the whole repo dir is collected (the old
    behaviour); with a target (a repo-relative file path) only
    that file is collected — the scoped-collection path. Only
    two-part ids (file.py::test_fn, i.e. module-level
    functions) are kept: TestCase methods are already collected
    in dotted form by unittest discovery, and merging pytest's
    three-part Class::method ids would duplicate them. Parametrised
    ids (with [..]) never match the pattern and are dropped: the
    enumerator produces no parametrised ids, so they could only
    enter as unvalidated regressions. Absolute paths in pytest
    output are relativised to repo_dir so they match enumeration
    ids; anything else is kept as printed (it simply won't match
    a candidate). A pytest collection failure returns [] rather
    than aborting: unlike a unittest loader failure it does not
    invalidate the unittest namespace, and module-level candidates
    then drop as non-collecting through the normal path.
    """
    try:
        proc = subprocess.run(
            [python, "-m", "pytest", "--collect-only", "-q",
             target if target is not None else repo_dir],
            cwd=repo_dir, capture_output=True, text=True,
            timeout=COLLECT_TIMEOUT_SECS)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    root = Path(repo_dir)
    ids: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        m = _PYTEST_ID_RE.match(line)
        if not m:
            continue
        path, func = m.group(1), m.group(2)
        p = Path(path)
        if p.is_absolute():
            try:
                path = p.relative_to(root).as_posix()
            except ValueError:
                continue
        else:
            path = path.replace("\\", "/")
        ids.add(f"{path}::{func}")
    return sorted(ids)


def _run_pytest(python: str, test_id: str, cwd: str,
                timeout_secs: int) -> bool:
    """Run one pytest id in a subprocess; True only on exit 0.

    Fail, error, collection error (exit 5 counts here as
    non-zero), and timeout all count as not-passed, mirroring
    _run_unittest's exit-0 rule.
    """
    try:
        proc = subprocess.run(
            [python, "-m", "pytest", "-q", test_id],
            cwd=cwd, capture_output=True, text=True,
            timeout=timeout_secs)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _run_test(python: str, test_id: str, cwd: str,
              timeout_secs: int) -> bool:
    """Run one test id with the runner its namespace implies.

    Ids containing "::" are pytest ids (module-level functions);
    everything else is a dotted unittest id. The dispatch is by
    id shape only, never by guessing from file contents.
    """
    if "::" in test_id:
        return _run_pytest(python, test_id, cwd, timeout_secs)
    return _run_unittest(python, test_id, cwd, timeout_secs)


def _collect_unittest_module_ids(python: str, repo_dir: str,
                                 rel_path: str) -> tuple[list[str], list[str]]:
    """Collect unittest ids for ONE module file, by discovery.

    Returns (sorted collected ids, loader-failure ids) for that
    module only. Discovery runs with start_dir = the module's
    directory, pattern = the module's exact basename, and
    top_level_dir = repo_dir, so dotted ids match whole-tree
    discovery exactly. The basename must match unittest's
    'test*.py' pattern, as in whole-tree discovery: a file that
    does not match (e.g. tests/helpers.py) yields no ids here,
    exactly as it yielded none before — it is not smuggled in
    by passing its exact name as the pattern. A module that fails
    to import yields its ids in `failed` (unittest _FailedTest
    entries), never an abort: the caller drops its candidates as
    non-collecting and probes the underlying error separately.
    A collector crash or timeout also yields no ids (the probe,
    not this function, reports why). The collector script runs
    from a temp file outside the repo so it is never discovered.
    """
    import fnmatch

    if not fnmatch.fnmatch(Path(rel_path).name, "test*.py"):
        return [], []
    start_dir = str(Path(repo_dir) / Path(rel_path).parent)
    script = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False,
                encoding="utf-8") as fh:
            fh.write(_COLLECTOR_SRC)
            script = fh.name
        proc = subprocess.run(
            [python, script, start_dir, Path(rel_path).name, repo_dir],
            cwd=repo_dir, capture_output=True, text=True,
            timeout=COLLECT_TIMEOUT_SECS)
    except (subprocess.TimeoutExpired, OSError):
        return [], []
    finally:
        if script is not None:
            try:
                os.unlink(script)
            except OSError:
                pass
    if proc.returncode != 0:
        return [], []
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        ids, failed = data["ids"], data["failed"]
    except (ValueError, KeyError, IndexError):
        return [], []
    if not isinstance(ids, list) or not isinstance(failed, list):
        return [], []
    return sorted(set(ids)), failed


def _collect_module_ids(python: str, repo_dir: str, rel_path: str,
                        pytest_ok: bool) -> tuple[list[str], list[str]]:
    """Collect both namespaces' ids for ONE module file.

    The union of unittest discovery (dotted ids) and, when pytest
    is importable, pytest --collect-only module-level ids
    (path::function) for that file only. Pytest collection is
    restricted to files matching pytest's default python_files
    patterns (test_*.py, *_test.py): an explicitly passed file
    outside those patterns would otherwise collect here while
    whole-directory collection never produced it. Returns
    (sorted ids, unittest loader-failure ids); the failure list
    is informational only — a failed module removes its own ids
    from the pool, it never aborts the task (see
    _convert_one_task).
    """
    ids, failed = _collect_unittest_module_ids(python, repo_dir, rel_path)
    if pytest_ok:
        name = Path(rel_path).name
        if name.startswith("test_") or name.endswith("_test.py"):
            ids = sorted(set(ids) | set(
                _collect_pytest_ids(python, repo_dir, rel_path)))
    return ids, failed


def _probe_collection_error(python: str, repo_dir: str, test_id: str,
                            rel_path: str, pytest_ok: bool) -> str:
    """One-line WHY for a candidate that failed to collect.

    Runs the collection the candidate needed and returns the
    most informative error line (the exception line, e.g.
    "ModuleNotFoundError: No module named 'httpretty'"), never a
    bare id. Pytest ids (path::function): pytest --collect-only
    on the file when pytest is importable, else a plain statement
    that pytest is unavailable. Dotted ids: a plain import of the
    module, which surfaces the import error without running any
    test. If the module imports/collects cleanly but the id still
    was not produced, that is reported as such (the usual cause
    is a file outside the discovery patterns). Only ever called
    for dropped candidates, so its subprocess cost is bounded by
    the number of dropped ids, not the tree size.
    """
    def error_line(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        for ln in reversed(lines):
            if "Error" in ln:
                return ln[:200]
        return lines[-1][:200]

    try:
        if "::" in test_id:
            if not pytest_ok:
                return ("pytest not importable in test environment "
                        "(module-level candidates cannot collect)")
            proc = subprocess.run(
                [python, "-m", "pytest", "--collect-only", "-q", rel_path],
                cwd=repo_dir, capture_output=True, text=True,
                timeout=COLLECT_TIMEOUT_SECS)
            if proc.returncode != 0:
                line = error_line(proc.stderr) or error_line(proc.stdout)
                return line or "pytest collection failed (no output)"
            return ("collected cleanly but id not produced by "
                    "pytest --collect-only")
        dotted = rel_path[:-3].replace("/", ".") if rel_path.endswith(".py") else rel_path
        proc = subprocess.run(
            [python, "-c", f"import {dotted}"],
            cwd=repo_dir, capture_output=True, text=True,
            timeout=COLLECT_TIMEOUT_SECS)
        if proc.returncode != 0:
            line = error_line(proc.stderr) or error_line(proc.stdout)
            return line or "module import failed (no output)"
        return ("module imports cleanly but id not produced by "
                "discovery (file may not match the test*.py pattern, "
                "or the test is not in a collected class)")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"collection probe failed: {exc}"


def _enumerate_tests(tree: dict[str, str]) -> dict[str, tuple[str, int, int, tuple[str, ...]]]:
    """Map test id -> (path, lineno, end_lineno, source).

    Two id namespaces, matching the two runners that validate
    them (see _run_test):

    * TestCase-style methods (any base class: unittest
      collectability is confirmed later by the loader, so a local
      TestCase base class is not missed) use dotted unittest ids:
      module.Outer.Inner.test_x for nested classes, matching
      unittest's id format.
    * Module-level `def test_*` / `async def test_*` functions use
      pytest ids: path::function (e.g. tests/test_api.py::
      test_status_ok). These are the tests fastapi/requests/rich
      suites are largely built from, and the pre-fix enumerator
      excluded them entirely — a systematic blind spot: a
      test_patch touching only module-level functions yielded
      added == modified == [] and the task skipped with
      "test_patch adds or modifies no test function", even though
      real failing tests existed. pytest collectability is
      confirmed later by scoped per-module collection, exactly
      as unittest collectability is for dotted ids.

    Files that fail to parse are skipped here (the loader will
    surface real problems); a test_patch that introduces a syntax
    error therefore yields no candidate from that file, and the
    task is skipped downstream rather than converted on a guessed
    id.
    """
    out: dict[str, tuple[str, int, int, tuple[str, ...]]] = {}
    for path in sorted(tree):
        if not path.endswith(".py"):
            continue
        lines = tree[path].splitlines()
        try:
            module_node = ast.parse(tree[path])
        except SyntaxError:
            continue
        module = path[:-3].replace("/", ".")

        # Module-level test functions -> pytest ids.
        for node in module_node.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")):
                end = node.end_lineno or node.lineno
                src = tuple(lines[node.lineno - 1:end])
                out[f"{path}::{node.name}"] = (path, node.lineno, end, src)

        def visit(body, qual: list[str]) -> None:
            for node in body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name.startswith("test")):
                        tid = ".".join([module] + qual + [node.name, item.name])
                        end = item.end_lineno or item.lineno
                        src = tuple(lines[item.lineno - 1:end])
                        out[tid] = (path, item.lineno, end, src)
                visit(node.body, qual + [node.name])

        visit(module_node.body, [])
    return out


def _classify_test_patch(old_tree: dict[str, str],
                         new_tree: dict[str, str]) -> tuple[list[str], list[str]]:
    """Split test_patch's test changes into (added, modified) ids.

    Added: ids present in the new tree but not the old. Modified:
    ids present in both whose function source text differs. This
    catches the case the old converter skipped entirely — a
    test_patch that only modifies existing tests — because the
    failing test is then identified by running, not by guessing.
    Deleted ids are ignored (a deleted test cannot fail).
    """
    old = _enumerate_tests(old_tree)
    new = _enumerate_tests(new_tree)
    added = sorted(t for t in new if t not in old)
    modified = sorted(t for t in new if t in old and new[t][3] != old[t][3])
    return added, modified


def _order_regression_candidates(collected: list[str],
                                 enum: dict[str, tuple[str, int, int, tuple[str, ...]]],
                                 added: list[str], modified: list[str],
                                 test_touched: set[str],
                                 gold_touched: set[str]) -> list[str]:
    """Order pre-existing tests: touched files first, then id order.

    Added/modified tests are excluded (they are failing-test
    candidates, not regressions). Group 0: tests in files touched
    by test_patch; group 1: tests in files touched by the gold
    patch; group 2: everything else. Within a group, sorted id
    order — a fixed seed, no randomness, so two conversions of the
    same data produce the same regression list. The input list
    is enumerated ids (not collected ids): collection happens
    per module on demand in _convert_one_task, so an id whose
    module fails to import simply contributes nothing downstream.
    """
    skip = set(added) | set(modified)

    def group(tid: str) -> int:
        path = enum[tid][0] if tid in enum else ""
        if path in test_touched:
            return 0
        if path in gold_touched:
            return 1
        return 2

    cands = [t for t in collected if t not in skip]
    return sorted(cands, key=lambda t: (group(t), t))


def _write_tree(root: str, tree: dict[str, str]) -> None:
    for rel, content in tree.items():
        full = Path(root) / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")


def _describe_test_patch_touch(old_tree: dict[str, str],
                               patched: dict[str, str],
                               sections: list) -> str:
    """Compact description of what a test_patch actually touches.

    Used only in the "adds or modifies no test function" skip, so
    the next log distinguishes "the patch touches no test code at
    all" from "enumeration missed a test function". For each
    touched file: new vs modified, and the top-level function and
    class names found in the patched version (capped); non-Python
    files and unparsable files are labelled as such. Never
    guesses: names come from parsing the patched tree.
    """
    parts: list[str] = []
    for fd in sections:
        path = fd.new_path
        if path is None:
            continue
        if not path.endswith(".py"):
            parts.append(f"{path} (non-Python)")
            continue
        status = "new" if path not in old_tree else "modified"
        try:
            node = ast.parse(patched.get(path, ""))
        except SyntaxError:
            parts.append(f"{path} ({status}, unparsable)")
            continue
        names = [n.name for n in node.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))]
        if not names:
            parts.append(f"{path} ({status}: no top-level defs)")
        else:
            shown = ", ".join(names[:12])
            suffix = ", …" if len(names) > 12 else ""
            parts.append(f"{path} ({status}: {shown}{suffix})")
    return "; ".join(parts[:10]) if parts else "no files"


def _convert_one_task(obj: dict, snapshots_dir: Path, python: str,
                      env_label: str, per_test_timeout: int) -> tuple[dict, dict]:
    """Convert one task, raising _TaskSkip on any derivation failure.

    Three scratch states are used; the gold patch is applied ONLY
    inside the gold scratch copy and NEVER written to the output
    record (the output `files` stay the buggy snapshot + test_patch,
    exactly as eval_dev.py evaluates):

    1. buggy+test_patch (the evaluated repo): failing candidates
       are the added + modified test ids, validated against real
       test collection — scoped per module (unittest discovery
       for dotted ids plus, when pytest is importable, pytest
       --collect-only for path::function ids), never whole-tree
       — and non-collecting ids are dropped and counted, with a
       probed collection error per dropped id. A candidate that
       PASSES here is not a failing test and is dropped.
    2. buggy+test_patch+gold: a surviving candidate must PASS
       here. This is the guard against buggy-only mislabelling:
       a test that fails on buggy code because the environment is
       broken (missing dep, bad import) also fails on gold code,
       and is dropped instead of being recorded as FAIL_TO_PASS.
    3. regressions: pre-existing collected tests, in preference
       order (touched files first), that pass on BOTH buggy and
       gold code, capped at MAX_REGRESSION_TESTS (the gate's cap).
       A regression that passes on buggy but fails on gold would
       be broken by the fix itself, so it is not a regression.

    Returns (output record, derivation info). Raises _TaskSkip with
    a counted reason when exactly one FAIL_TO_PASS id cannot be
    derived or no passing regression can be derived.
    """
    task_id = obj["instance_id"]
    archive = snapshots_dir / f"{task_id}.tgz"
    if not archive.is_file():
        raise _TaskSkip(
            f"no snapshot archive {archive.name} — snapshot naming "
            "mismatch: run --inspect, list the snapshots dir, and "
            "adapt the converter; do not guess a different name")
    tree, skipped_binary, skipped_reasons = _extract_snapshot(
        archive, task_id)
    sections = _parse_unified_diff(obj["test_patch"], task_id)
    patched = _apply_diff(tree, sections, task_id, "test_patch")
    if not patched:
        raise _TaskSkip("patched snapshot has no files")
    gold_sections = _parse_unified_diff(obj["patch"], task_id)
    gold_tree = _apply_diff(patched, gold_sections, task_id,
                            "gold patch")
    added, modified = _classify_test_patch(tree, patched)
    if not added and not modified:
        raise _TaskSkip(
            "test_patch adds or modifies no test function "
            f"(touched: {_describe_test_patch_touch(tree, patched, sections)})")
    test_touched = {fd.new_path for fd in sections
                    if fd.new_path is not None and fd.new_path.endswith(".py")}
    gold_touched = {fd.new_path for fd in gold_sections
                    if fd.new_path is not None and fd.new_path.endswith(".py")}
    buggy_dir = tempfile.mkdtemp(prefix="build_dev_buggy_")
    gold_dir = tempfile.mkdtemp(prefix="build_dev_gold_")
    try:
        _write_tree(buggy_dir, patched)
        _write_tree(gold_dir, gold_tree)
        # SCOPED COLLECTION (round 5, 1 Nov 2026). The old code
        # ran whole-tree unittest discovery and skipped the task
        # whenever ANY module failed to import ("partial
        # collection untrustworthy") — on the round-4 subset that
        # killed all 10 fastapi tasks, each with ~300 discovered
        # modules, on one unrelated broken module (benchmarks).
        # Collection now runs per module, only for modules that
        # contain a candidate failing test or a regression
        # candidate, in enumeration order. This is safe because
        # every id is validated by actually running it (fail on
        # buggy + pass on gold for FAIL_TO_PASS, pass on both for
        # regressions): a failed module only removes its own ids
        # from the pool, it cannot mislabel an id. A module that
        # fails to import therefore never kills the task, and a
        # dropped candidate carries the underlying import error
        # (see _probe_collection_error) instead of a bare id.
        enum_patched = _enumerate_tests(patched)
        pytest_ok = _pytest_available(python)
        collected_cache: dict[str, list[str]] = {}

        def collect_module(rel_path: str) -> list[str]:
            if rel_path not in collected_cache:
                ids, _failed = _collect_module_ids(
                    python, buggy_dir, rel_path, pytest_ok)
                collected_cache[rel_path] = ids
            return collected_cache[rel_path]

        collected_set: set[str] = set()
        for cid in added + modified:
            if cid in enum_patched:
                collected_set.update(
                    collect_module(enum_patched[cid][0]))
        dropped = sorted(t for t in added + modified
                         if t not in collected_set)
        dropped_errors: dict[str, str] = {}
        for tid in dropped:
            if tid in enum_patched:
                dropped_errors[tid] = _probe_collection_error(
                    python, buggy_dir, tid, enum_patched[tid][0],
                    pytest_ok)
            else:
                dropped_errors[tid] = ("no enumerated source path "
                                       "for candidate")
        candidates = sorted(t for t in added + modified
                            if t in collected_set)
        if not candidates:
            errors = "; ".join(f"{tid}: {dropped_errors[tid]}"
                               for tid in dropped)
            raise _TaskSkip(
                "all candidate failing test(s) failed collection "
                f"(dropped non-collecting: {dropped}; "
                f"collection errors: {errors})")
        survivors: list[str] = []
        for cid in candidates:
            if _run_test(python, cid, buggy_dir, per_test_timeout):
                continue  # passes on buggy code: not a failing test
            if _run_test(python, cid, gold_dir, per_test_timeout):
                survivors.append(cid)
            # Else fails on gold too: environment failure or a test
            # the gold patch does not fix — dropped, never recorded.
        if len(survivors) != 1:
            if not survivors:
                raise _TaskSkip(
                    "no FAIL_TO_PASS test derived: every candidate "
                    "either passes on buggy code or fails on "
                    "gold-patched code")
            raise _TaskSkip(
                f"{len(survivors)} FAIL_TO_PASS candidates survive; "
                f"ambiguous, cannot pick one: {survivors}")
        # Regression pool, scoped the same way: walk the
        # pre-existing enumerated tests in touched-first order,
        # collecting each module on demand (cached) until 60
        # collected regression ids accumulate, then re-sort by
        # (group, id) and cap. The pool therefore differs from
        # whole-tree collection ONLY in which ids are preferred
        # when it overflows (a module-order walk, not a
        # whole-tree sort) — every pooled id is still validated
        # by running on both trees below, so this cannot
        # mislabel a regression. Modules that fail to import
        # contribute nothing and are passed over, never fatal.
        skip_ids = set(added) | set(modified)
        preexisting = sorted(t for t in enum_patched
                             if t not in skip_ids)
        ordered_enum = _order_regression_candidates(
            preexisting, enum_patched, added, modified,
            test_touched, gold_touched)
        pool_set: set[str] = set()
        for tid in ordered_enum:
            if len(pool_set) >= REGRESSION_POOL:
                break
            for cid in collect_module(enum_patched[tid][0]):
                if cid not in skip_ids:
                    pool_set.add(cid)
        ordered = _order_regression_candidates(
            sorted(pool_set), enum_patched, added, modified,
            test_touched, gold_touched)[:REGRESSION_POOL]
        regression: list[str] = []
        for cid in ordered:
            if len(regression) >= MAX_REGRESSION_TESTS:
                break
            if not _run_test(python, cid, buggy_dir, per_test_timeout):
                continue
            if _run_test(python, cid, gold_dir, per_test_timeout):
                regression.append(cid)
        if not regression:
            raise _TaskSkip(
                "no passing regression tests derived (every "
                "pre-existing test fails on buggy or gold code)")
        issue_text = obj["problem_statement"]
        if obj["hints_text"]:
            issue_text += "\n\nHints:\n" + obj["hints_text"]
        record = {
            "task_id": task_id,
            "files": patched,
            "failing_test_id": survivors[0],
            "regression_test_ids": regression,
            "bug_class": "unknown",
            "issue_text": issue_text,
            # Snapshot members skipped by the binary-skip allowlist
            # (see _extract_snapshot): .git/ VCS metadata at any
            # depth plus any other non-UTF-8 member, sorted and
            # deduplicated, repo-relative. Empty for snapshots
            # with no skipped members. eval_dev.py ignores extra
            # keys, so this is audit trail only — but it is the
            # guarantee that a binary evaluated source would be
            # visible here instead of silently dropped.
            "skipped_binary_files": skipped_binary,
            # Per-path reason for each skipped member: "git"
            # (.git/ VCS metadata), "binary" (non-UTF-8 member),
            # "oversized" (member over MAX_SNAPSHOT_FILE_BYTES
            # that is not .py/.pyi — oversized .py/.pyi aborts
            # the task instead, see _extract_snapshot), or
            # "link-outside-snapshot" (a symlink/hard link whose
            # target is not inlined text in the snapshot — absent,
            # absolute/escaping, skipped as git/binary/oversized,
            # or a link loop; see _extract_snapshot). Keys are
            # the same repo-relative paths as skipped_binary_files.
            "skipped_binary_reasons": skipped_reasons,
            # Derivation audit trail (eval_dev.py ignores extra keys).
            # `env` names the environment the tests ran in: the
            # dataset's docker spec / offline wheels when the
            # operator passes --env-label, otherwise the local
            # interpreter — never silently the scorer's environment.
            "derivation": {
                "env": env_label,
                "pytest_available": pytest_ok,
                "failing_candidates": candidates,
                "dropped_noncollecting": dropped,
                # One-line collection error per dropped candidate
                # (see _probe_collection_error): the next Kaggle
                # log must be read from these, not guessed from
                # the bare ids.
                "dropped_collection_errors": dropped_errors,
                "regression_pool": len(ordered),
            },
        }
        return record, record["derivation"]
    finally:
        shutil.rmtree(buggy_dir, ignore_errors=True)
        shutil.rmtree(gold_dir, ignore_errors=True)


def _is_git_member(parts: list[str]) -> bool:
    """True if any path component is `.git` (VCS metadata, any depth).

    Checked AFTER path validation (see _extract_snapshot): a member
    whose path escapes the tree via '..' still aborts even if it
    also mentions .git — the allowlist never bypasses safety.
    """
    return ".git" in parts


def _resolve_link_target(link_rel: str, target: str,
                         is_symlink: bool) -> str | None:
    """Resolve a tar link target to a repo-relative member path.

    Symlink targets resolve relative to the link's directory;
    hard-link targets are root-relative member names. Returns the
    normalised target path, or None when the target is absolute
    or escapes the snapshot root (both are unresolvable, never
    followed). Backslashes are NOT separators here: tar member
    names use '/'; a backslash in a target is kept literal.
    """
    if not target or target.startswith("/"):
        return None
    target = target.replace("\\", "/")
    if is_symlink:
        base = link_rel.split("/")[:-1]
    else:
        base = []
    parts: list[str] = []
    for part in base + target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        return None
    rel = "/".join(parts)
    return rel if _is_safe_rel_path(rel) else None


def _extract_snapshot(archive: Path, task_id: str
                      ) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Extract a per-task .tgz snapshot into (tree, skipped, reasons).

    Returns the repo tree, the sorted, deduplicated list of
    snapshot members that were SKIPPED rather than inlined, and a
    per-path reason mapping (path -> "git" | "binary" |
    "oversized"), all in repo-relative form (after the
    single-top-directory strip below, when it applies). The caller
    records the list on the output record as
    `skipped_binary_files` and the mapping as
    `skipped_binary_reasons` — nothing is skipped silently.

    BINARY-SKIP ALLOWLIST (1 Nov 2026 — the version the old
    docstring called for). The first real-data conversion attempt
    (20-task subset, CPU) converted 0/20: every real snapshot is a
    git checkout, and the old code aborted a task on ANY non-UTF-8
    member (first offender: .git/index). Two categories are now
    skipped, never aborting the task for binary content alone:

    (1) Members under any `.git/` directory, at any depth
        (e.g. ./.git/index, ./.git/objects/...). These are VCS
        metadata, never evaluated source: they are not importable
        Python, unittest discovery (pattern test*.py from the repo
        root) never collects them, and test_patch paths never point
        into .git/. They are skipped WITHOUT being read, so the
        size cap below does not apply to them, and they are listed
        even when their content happens to be valid UTF-8 text
        (e.g. .git/config) — the skip is by location, not content.

    (2) Any other member that fails UTF-8 decoding (a binary
        asset such as a PNG). It is listed rather than hidden, so
        that if a future task's actually-evaluated source ever
        were binary, the record's skipped list makes that visible
        instead of silently changing what the harness evaluates.

    OVERSIZED MEMBERS (1 Nov 2026, second subset fix). An
    oversized member NO LONGER aborts the task (the second subset
    run lost 10 tasks to logo .ai/.gif assets this way). Instead
    it is routed into the skipped list with reason "oversized",
    WITHOUT being read (so inlining a multi-MB blob is never
    attempted). One guard, and it is absolute: an oversized
    member that looks like evaluated source — any .py or .pyi
    file, at any path — aborts the task (counted SKIP
    "oversized evaluated source"), because converting with a hole
    in evaluated code would corrupt the measurement. Binary
    assets (.ai, .gif, .png, etc.) and .git/ members never
    trigger that guard: .git/ is checked first, so an oversized
    .py under .git/ is skipped as "git", not guarded. LIMIT:
    the guard covers .py/.pyi only; an oversized member of
    another type that a test actually reads (a .json/.txt
    fixture, say) is skipped with reason "oversized" and listed
    — visible in the record, but check the list on the Kaggle
    re-run rather than assuming every oversized skip is a logo.

    LINKS (1 Nov 2026, third subset fix). The second version
    aborted a task on ANY symlink/hard link; the third subset run
    lost 8 requests tasks that way (e.g. tests/certs/mtls/client/ca
    inside the snapshot). Links are now resolved, never followed
    out of the snapshot:

    * The link's OWN path is validated exactly like a regular
      member's (normalisation, '..' rejection, _is_safe_rel_path)
      BEFORE any resolution — an unsafe link path still aborts.
    * Resolution happens in a second pass, after all regular
      members are collected, so a link may precede its target in
      the archive. A symlink target is resolved relative to the
      link's directory; a hard-link target is a root-relative
      member name (tar semantics). '.' and duplicate '/' are
      dropped, '..' pops one component, and escaping above the
      snapshot root, or an absolute target, is unresolvable.
    * A target that resolves to a member present as TEXT in the
      tree contributes that member's content (chains of links
      are followed, at most 10 hops; a loop is unresolvable).
      A link to a member that is NOT inlined text — absent, or
      skipped as git/binary/oversized — is NOT guessed: the link
      itself is skipped with reason "link-outside-snapshot" and
      listed, and the task continues. Downstream validation
      handles the consequence correctly: a test needing an
      unavailable fixture fails on both buggy and gold code, so
      it is dropped from both sets, not misrecorded.
    * A regular member with the same path as a link wins over
      the link (duplicate-name snapshots are already degenerate;
      silently overwriting real content with link content would
      be worse).

    What still aborts loudly, as before: a corrupt or unreadable
    tar archive (open/list/read failures), an unsafe member path
    (absolute path or '..' component — including a link's OWN
    path), an oversized .py/.pyi member (see guard above), a
    snapshot with no text files at all, and an ambiguous
    single-top-directory layout. A link TARGET that is absolute
    or escapes the snapshot root does NOT abort: it is simply
    unresolvable, so the link is skipped with reason
    "link-outside-snapshot" like any other unresolvable target.
    Path validation always runs before the .git check, so no
    member can hide an unsafe path behind a .git component.

    DIRECTORY MEMBERS (fourth subset fix, 1 Nov 2026). Non-file,
    non-link members are ignored, never listed and never
    aborting — that was already true for "repo/"-style entries,
    which reach the `not member.isfile()` branch below. The one
    shape that escaped it was a member reducing to ZERO path
    parts after "./" stripping: the "./" root directory entry
    every real snapshot begins with (followed by "./.git/",
    "./.git/HEAD", ...). The old code aborted on zero parts
    before reaching the directory branch, so the round-3
    re-run skipped all 20 tasks at extraction with "unsafe
    snapshot member '.'" and exercised none of the round-3
    fixes. Zero-parts members are now ignored like any other
    directory member. The ".." abort is unchanged and is
    checked separately, so an escaping path ("..", "./..",
    "a/../b") still aborts.
    """
    try:
        tf = tarfile.open(archive, "r:*")
    except (tarfile.TarError, OSError) as exc:
        _abort(f"task {task_id}: cannot open snapshot {archive}: {exc}")
    raw: dict[str, str] = {}
    skipped: list[str] = []
    reasons: dict[str, str] = {}
    links: list[tuple[str, str, bool]] = []  # (rel, target, is_symlink)
    with tf:
        try:
            members = tf.getmembers()
        except (tarfile.TarError, OSError) as exc:
            _abort(f"task {task_id}: cannot list snapshot {archive}: "
                   f"{exc}")
        for member in members:
            # The link's OWN path is validated exactly like a
            # regular member's, BEFORE any resolution (see the
            # LINKS section of the docstring). Non-file,
            # non-link members (directories etc.) are ignored.
            name = member.name.replace("\\", "/")
            while name.startswith("./"):
                name = name[2:]
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if not parts:
                # ROOT DIRECTORY ENTRY (fourth subset fix, 1 Nov
                # 2026). Real snapshots are tarred git checkouts
                # and begin with a "./" root directory entry,
                # which strips to zero path parts. The old code
                # lumped "no parts" in with the ".." abort above,
                # so that harmless entry aborted every task in
                # the round-3 re-run (0/20, before any round-3
                # fix was exercised). A zero-parts member is a
                # directory entry, not a file: ignore it exactly
                # like every other directory member below (not
                # listed as a skipped binary — it is not a file
                # at all). Only a ".." component still aborts.
                continue
            if any(p == ".." for p in parts):
                _abort(f"task {task_id}: unsafe snapshot member "
                       f"{member.name!r}")
            rel = "/".join(parts)
            if not _is_safe_rel_path(rel):
                _abort(f"task {task_id}: unsafe snapshot path "
                       f"{rel!r}")
            if member.issym() or member.islnk():
                # Record now, resolve in the second pass below:
                # a link may precede its target in the archive.
                links.append((rel, member.linkname, member.issym()))
                continue
            if not member.isfile():
                continue
            if _is_git_member(parts):
                # VCS metadata: skip without reading (no size cap,
                # no decode), but list it — see docstring above.
                skipped.append(rel)
                reasons[rel] = "git"
                continue
            if member.size > MAX_SNAPSHOT_FILE_BYTES:
                # Oversized: skip with reason, never inline — but
                # NEVER silently drop evaluated source (see guard
                # in the docstring above). .git/ already returned
                # above, so this guard cannot fire on VCS paths.
                if rel.endswith(".py") or rel.endswith(".pyi"):
                    _abort(f"task {task_id}: oversized evaluated "
                           f"source {rel!r} exceeds "
                           f"{MAX_SNAPSHOT_FILE_BYTES} bytes; "
                           "refusing to convert with a hole in "
                           "evaluated code")
                skipped.append(rel)
                reasons[rel] = "oversized"
                continue
            try:
                data = tf.extractfile(member)
                blob = data.read() if data is not None else b""
            except (tarfile.TarError, OSError) as exc:
                _abort(f"task {task_id}: cannot read snapshot member "
                       f"{rel!r}: {exc}")
            try:
                raw[rel] = blob.decode("utf-8")
            except UnicodeDecodeError:
                # Binary asset: skip, but list it — never abort
                # the task for binary content alone.
                skipped.append(rel)
                reasons[rel] = "binary"
    # Second pass: resolve recorded links against the collected
    # text tree. A link to inlined text contributes that content
    # (chains followed, loop-guarded); anything else — absent
    # target, absolute/escaping target, target skipped as
    # git/binary/oversized, or a link loop — is skipped with
    # reason "link-outside-snapshot" and the task continues.
    link_targets: dict[str, tuple[str, bool]] = {
        rel: (target, is_sym) for rel, target, is_sym in links}
    for rel, target, is_sym in links:
        if rel in raw:
            # A regular member with the same path wins; never
            # overwrite real content with link content.
            continue
        seen: set[str] = set()
        cur_rel, cur_target, cur_sym = rel, target, is_sym
        resolved: str | None = None
        for _ in range(10):
            if cur_rel in seen:
                break
            seen.add(cur_rel)
            nxt = _resolve_link_target(cur_rel, cur_target, cur_sym)
            if nxt is None:
                break
            if nxt in raw:
                resolved = raw[nxt]
                break
            if nxt in link_targets:
                cur_rel, cur_target, cur_sym = (
                    nxt, link_targets[nxt][0], link_targets[nxt][1])
                continue
            break
        if resolved is not None:
            raw[rel] = resolved
        else:
            skipped.append(rel)
            reasons[rel] = "link-outside-snapshot"
    if not raw:
        _abort(f"task {task_id}: snapshot {archive} contains no "
               "files (all members skipped or empty)")
    # Single top-level directory shape: strip it so paths are
    # repo-relative (the layout eval_dev.py runs tests from).
    # Skipped paths get the same strip, so the record's list is
    # expressed in the same repo-relative terms as `files`.
    tops = {rel.split("/", 1)[0] for rel in raw}
    if len(tops) == 1:
        top = next(iter(tops))
        if any("/" in rel for rel in raw):
            stripped = {}
            for rel, content in raw.items():
                if "/" in rel:
                    stripped[rel.split("/", 1)[1]] = content
                # A lone top-level file cannot exist alongside the
                # single-dir shape check above unless it IS the top
                # itself; such a file would be dropped by stripping,
                # so refuse rather than silently lose it.
                else:
                    _abort(f"task {task_id}: snapshot mixes a "
                           f"top-level file {rel!r} with a single "
                           "top-level directory; layout ambiguous")
            if not stripped:
                _abort(f"task {task_id}: snapshot stripping left no "
                       "files")
            stripped_skipped = []
            stripped_reasons: dict[str, str] = {}
            for rel in skipped:
                if rel.startswith(top + "/"):
                    rest = rel[len(top) + 1:]
                    if rest:
                        stripped_skipped.append(rest)
                        stripped_reasons[rest] = reasons[rel]
                    else:
                        stripped_skipped.append(rel)
                        stripped_reasons[rel] = reasons[rel]
                else:
                    # A skipped member outside the single top dir
                    # (e.g. a top-level binary asset): keep its
                    # path as-is; stripping must never drop it.
                    stripped_skipped.append(rel)
                    stripped_reasons[rel] = reasons[rel]
            return stripped, sorted(set(stripped_skipped)), stripped_reasons
    return raw, sorted(set(skipped)), dict(reasons)


def cmd_convert(tasks_path: Path, snapshots_dir: Path,
                out_path: Path, python: str, env_label: str,
                per_test_timeout: int) -> int:
    if not tasks_path.is_file():
        _abort(f"tasks file {tasks_path} not found")
    if not snapshots_dir.is_dir():
        _abort(f"snapshots dir {snapshots_dir} not found")
    if not (Path(python).exists() or shutil.which(python)):
        _abort(f"test interpreter {python!r} not found; pass a valid "
               "--python")
    _MISMATCH_DIAGNOSTICS.clear()
    records: list[dict] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    n_dropped_noncollecting = 0
    with tasks_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                _abort(f"tasks line {lineno} is not valid JSON: {exc}")
            if not isinstance(obj, dict):
                _abort(f"tasks line {lineno} is not a JSON object")
            for req in REQUIRED_STR_FIELDS:
                if req not in obj:
                    _abort(f"tasks line {lineno} misses required field "
                           f"{req!r} — schema mismatch: run --inspect "
                           "and adapt the converter")
                if not isinstance(obj[req], str):
                    _abort(f"tasks line {lineno}: field {req!r} must "
                           "be a string")
            task_id = obj["instance_id"]
            if not task_id:
                _abort(f"tasks line {lineno}: instance_id must be a "
                       "non-empty string")
            if task_id in seen:
                _abort(f"tasks line {lineno}: duplicate task_id "
                       f"{task_id!r}")
            seen.add(task_id)
            # Per-task derivation: ANY failure here skips this task
            # (counted + listed) and the run continues. The strict
            # helpers raise SystemExit via _abort; a message naming
            # this task is a per-task failure, anything else is a
            # genuine abort and is re-raised.
            try:
                record, deriv = _convert_one_task(
                    obj, snapshots_dir, python, env_label,
                    per_test_timeout)
            except _TaskSkip as exc:
                skipped.append((task_id, exc.reason))
                print(f"[build-dev] SKIP task {task_id}: {exc.reason}",
                      file=sys.stderr)
                continue
            except SystemExit as exc:
                msg = str(exc)
                if f"task {task_id}:" in msg:
                    reason = msg.split("ABORT:", 1)[-1].strip()
                    # The strict helpers already prefix with
                    # "task <id>:"; drop it so the SKIP line does
                    # not print the id twice.
                    prefix = f"task {task_id}:"
                    if reason.startswith(prefix):
                        reason = reason[len(prefix):].strip()
                    skipped.append((task_id, reason))
                    print(f"[build-dev] SKIP task {task_id}: {reason}",
                          file=sys.stderr)
                    continue
                raise
            n_dropped_noncollecting += len(
                deriv.get("dropped_noncollecting", []))
            records.append(record)
    if not records:
        _abort("tasks file converted to no records (all tasks "
               "skipped, or file empty)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    n_seen = len(seen)
    n_failpass = len(records)  # conversion requires exactly one
    n_pass2pass = sum(1 for r in records if r["regression_test_ids"])
    n_skipped_binary = sum(len(r.get("skipped_binary_files", []))
                           for r in records)
    n_tasks_with_skips = sum(
        1 for r in records if r.get("skipped_binary_files"))
    print(f"[build-dev] wrote {len(records)} tasks to {out_path}")
    print(f"[build-dev] converted {len(records)}/{n_seen} tasks; "
          f"skipped {len(skipped)} (listed on stderr; NOT in the "
          "output file)")
    print(f"[build-dev] FAIL_TO_PASS coverage: {n_failpass}/{n_seen} "
          "tasks with a derived failing id (fail on buggy, pass on "
          "gold)")
    print(f"[build-dev] PASS_TO_PASS coverage: {n_pass2pass}/{n_seen} "
          "tasks with a non-empty regression list (pass on buggy "
          "and gold, capped at "
          f"{MAX_REGRESSION_TESTS}, touched files first)")
    print(f"[build-dev] dropped non-collecting ids: "
          f"{n_dropped_noncollecting} (derived from test_patch but "
          "absent from real test collection (unittest + pytest))")
    print(f"[build-dev] skipped binary members: {n_skipped_binary} "
          f"in {n_tasks_with_skips} converted task(s) "
          "(listed per record in `skipped_binary_files`: .git/ VCS "
          "metadata at any depth plus other non-UTF-8 members; "
          "counted here for converted tasks only — a task skipped "
          "for another reason has no output record to list them "
          "in)")
    reason_totals: dict[str, int] = {}
    for r in records:
        for reason in r.get("skipped_binary_reasons", {}).values():
            reason_totals[reason] = reason_totals.get(reason, 0) + 1
    if reason_totals:
        breakdown = ", ".join(
            f"{k}={v}" for k, v in sorted(reason_totals.items()))
        print(f"[build-dev] skipped binary reasons: {breakdown} "
              "(per record in `skipped_binary_reasons`)")
    if _MISMATCH_DIAGNOSTICS:
        # The diagnostics were already printed to stderr at the
        # moment of each mismatch; repeat them in the summary so
        # the run report carries them without grepping stderr.
        # Strictness is unchanged: every one of these tasks
        # skipped (counted above); the flag inside each entry
        # (added-lines-already-in-snapshot) is the check the
        # next Kaggle re-run must report on, not a reason to
        # convert a mismatching task.
        print(f"[build-dev] context-mismatch diagnostics: "
              f"{len(_MISMATCH_DIAGNOSTICS)} mismatching hunk(s) "
              f"(all corresponding tasks skipped):")
        for diag in _MISMATCH_DIAGNOSTICS:
            print(diag)
    print(f"[build-dev] derivation env: {env_label}")
    print("[build-dev] Validate with eval_dev.py --dry-run before "
          "any eval run, and save the notebook version.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the Gemma 4 dev-set data or convert it "
                    "to eval_dev.py JSONL (see module docstring).")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_insp = sub.add_parser("inspect", help="Kaggle-side discovery "
                            "(no contents shown)")
    p_insp.add_argument("data_dir", type=Path)
    p_cv = sub.add_parser("convert", help="Verified-schema "
                          "conversion (tasks.jsonl + snapshot .tgz "
                          "-> eval_dev.py JSONL)")
    p_cv.add_argument("--tasks", type=Path, required=True)
    p_cv.add_argument("--snapshots", type=Path, required=True)
    p_cv.add_argument("--out", type=Path, required=True)
    p_cv.add_argument("--python", default=sys.executable,
                      help="Interpreter to run tests with (default: "
                           "this interpreter). On Kaggle this must be "
                           "the notebook interpreter with the dataset's "
                           "dependencies available (docker spec / "
                           "offline wheels); see --env-label.")
    p_cv.add_argument("--env-label",
                      default="local-interpreter (NOT the dataset "
                              "docker spec)",
                      help="Environment label recorded per record in "
                           "the output `derivation` field. Pass an "
                           "explicit label (e.g. 'dataset-docker-spec' "
                           "or 'offline-wheels') when the tests ran "
                           "under the dataset's environment; never "
                           "claim scorer fidelity the run did not have.")
    p_cv.add_argument("--per-test-timeout", type=int, default=30,
                      help="Seconds per unittest subprocess during "
                           "derivation (default 30, mirroring the "
                           "harness gate).")
    args = parser.parse_args(argv)
    if args.mode == "inspect":
        return cmd_inspect(args.data_dir)
    return cmd_convert(args.tasks, args.snapshots, args.out,
                       args.python, args.env_label,
                       args.per_test_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
