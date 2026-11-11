"""Inverse-operator SEARCH BASELINE for Repo-Blind Repair (CPU-only).

What this is: a non-learned baseline on the frozen kill set (n=32
stratified held-out singles, 4 per operator, from
dataset/dev_tasks_heldout.jsonl (the companion Kaggle dataset,
placed alongside this repo as `dataset/`) — the exact set
described in preregistration/KILL_TEST_READINESS.md). For each task, every single-operator
INVERSE is tried exhaustively over every candidate site in task.py,
and a candidate counts as a fix only if the task's failing test
passes under the harness test runner (harness/test_runner.py
``run_one_test``). Regression behaviour is reported separately, using
the same capped suite the gate uses (``run_regression_suite``).

The search is SEED-BLIND: the held-out task_ids embed the mutation
seed (``heldout-<template>-<op>-<seed>``), but the search never reads
it — it tries every site and lets the test decide. Using the seed to
pick the site would be an oracle, not a baseline.

Inverses (deterministic from mutation/operators.py):
- off_by_one:        try value+1 and value-1 at every int constant
                     (mutation shifted by +-1; direction unknown).
- swapped_operands:  swap left/right at every BinOp (self-inverse).
- wrong_comparator:  re-apply the comparator swap at every Compare
                     (the swap map is an involution).
- swapped_boolean:   flip every bool constant (self-inverse).
- wrong_constant:    try value-d for d in 2..6 at every int constant
                     (mutation only ADDS 2..6, so the inverse only
                     subtracts).
- dropped_condition: NO deterministic inverse — the dropped operand
                     is not present in the buggy code. Not attempted.
- removed_return:    NO true inverse — the returned expression is
                     gone. A separate, clearly labelled reconstruction
                     guess is tried instead (see below); it is NOT
                     counted as an inverse fix.
- removed_assignment: NO deterministic inverse — the assignment's
                     target and value are both gone. Not attempted.

Reconstruction guess (NOT an inverse, reported separately): for each
bare ``return`` (value None), try ``return <last single-Name
assignment target textually before it in the same function>``. This
recovers accumulator-style returns (``return total``) and fails on
expression returns; the test decides, and the guess is recorded under
its own key so the inverse headline stays pure.

Usage:
  python3 search_baseline/search_baseline.py [--dry-run] [--validate-inverses]
      [--task-set all116] [--out search_baseline/search_baseline_results.json]
  (or run the whole 116-task check with: python3 reproduce_search.py)
  --dry-run: count candidates per task/operator, run no tests.
  --validate-inverses: check the inverse set recovers the fixed
      source on locally generated (operator, seed) examples; exits
      non-zero if any invertible operator fails to recover it.
      Run this first; it uses training-generator code, not held-out
      data, so it cannot leak the answer key into the search.

Outputs: JSON results (see --out) plus a human-readable
search_baseline/SEARCH_BASELINE_116.md written by hand from the
JSON, not by this script.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # published repo root (this file lives in repo/search_baseline/)
sys.path.insert(0, str(REPO / "mutation"))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "data_generation"))  # build_heldout_synthetic (validate only)

from operators import OPERATORS, _CMP_SWAP, _is_int_constant  # noqa: E402
from test_runner import run_one_test, run_regression_suite  # noqa: E402

# Inverses that are true inverses of a mutation operator.
TRUE_INVERSES = (
    "off_by_one",
    "swapped_operands",
    "wrong_comparator",
    "swapped_boolean",
    "wrong_constant",
)
# Operators with no deterministic inverse (information destroyed).
NO_INVERSE = ("dropped_condition", "removed_assignment")
# Reconstruction guess key (not an inverse).
RECON_KEY = "removed_return_reconstruction"

def _find_heldout() -> Path:
    """Held-out tasks: the companion Kaggle dataset, placed
    alongside this repo as ``dataset/`` (see repo README); fall
    back to a copy inside the repo tree if present."""
    for cand in (REPO.parent / "dataset" / "dev_tasks_heldout.jsonl",
                 REPO / "data" / "dev_tasks_heldout.jsonl",
                 HERE / "dev_tasks_heldout.jsonl"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "dev_tasks_heldout.jsonl not found: place the companion Kaggle "
        "dataset alongside this repo as dataset/ (see repo README)")


HELDOUT = _find_heldout()
FROZEN_PER_OPERATOR = 4


# ---------------------------------------------------------------------------
# Wilson interval — identical formula to eval_dev._wilson_interval
# (training/eval_dev.py, z=1.96); copied here so this CPU script does not
# import the GPU eval module (torch). Self-checked in --validate-inverses:
# Wilson(0,10) -> (0.00, 0.28); Wilson(0,8) -> (0.00, 0.32), matching
# the intervals already published in the paper.
# ---------------------------------------------------------------------------
def wilson(fixed: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = fixed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _iter_dfs(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _iter_dfs(child)


def _inverse_params(inverse: str) -> list:
    """Parameter values tried for one inverse at one site."""
    if inverse == "off_by_one":
        return [1, -1]  # try both directions; mutation used one of them
    if inverse == "wrong_constant":
        return [-2, -3, -4, -5, -6]  # mutation only added 2..6
    return [None]  # self-inverse, no parameter


class _Inverter(ast.NodeTransformer):
    """Applies one inverse at the target_index-th candidate site."""

    def __init__(self, inverse: str, target_index: int, param):
        self.inverse = inverse
        self.target_index = target_index
        self.param = param
        self._seen = -1

    def _is_target(self, node) -> bool:
        self._seen += 1
        return self._seen == self.target_index

    def visit_Constant(self, node):
        if self.inverse == "off_by_one" and _is_int_constant(node):
            if self._is_target(node):
                new = copy.copy(node)
                new.value = node.value + self.param
                return new
            return node
        if self.inverse == "swapped_boolean" and type(node.value) is bool:
            if self._is_target(node):
                new = copy.copy(node)
                new.value = not node.value
                return new
            return node
        if self.inverse == "wrong_constant" and _is_int_constant(node):
            if self._is_target(node):
                new = copy.copy(node)
                new.value = node.value + self.param
                return new
            return node
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.inverse == "swapped_operands":
            if self._is_target(node):
                new = copy.copy(node)
                new.left, new.right = node.right, node.left
                return new
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if self.inverse == "wrong_comparator":
            # Only swappable operators are candidate sites
            # (_count_sites counts only those). Check swappability
            # BEFORE counting: an unswappable comparison (is / in /
            # is not / not in) must not consume a site index, or
            # every later swappable site shifts by one and the
            # target flip is silently dropped (D1, 8 Nov 2026).
            if node.ops and type(node.ops[0]) in _CMP_SWAP:
                if self._is_target(node):
                    op_type = type(node.ops[0])
                    new = copy.copy(node)
                    new.ops = [_CMP_SWAP[op_type]()] + list(node.ops[1:])
                    return new
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        # Negative int literal: one int-constant site (see
        # _is_neg_int_literal). Do NOT generic_visit it: the inner
        # Constant(1) is not a separate site.
        if self.inverse in ("off_by_one", "wrong_constant") and _is_neg_int_literal(node):
            if self._is_target(node):
                new = ast.Constant(value=_neg_int_value(node) + self.param)
                return ast.copy_location(new, node)
            return node
        return self.generic_visit(node)


def _is_neg_int_literal(node: ast.AST) -> bool:
    """True for a negative int literal, e.g. ``-1``.

    NOTE: ``ast.unparse`` renders Constant(-1) as ``-1``, which
    re-parses as UnaryOp(USub, Constant(1)) — so a mutation that
    shifted 0 to -1 (off_by_one, odd seed) leaves a UnaryOp in the
    buggy code, not a plain Constant. The inverse must treat that
    UnaryOp as ONE int-constant site (value -1); its inner
    Constant(1) is not a separate site.
    """
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and _is_int_constant(node.operand)
    )


def _neg_int_value(node: ast.UnaryOp) -> int:
    return -node.operand.value


def _count_sites(tree: ast.AST, inverse: str) -> int:
    """Number of candidate sites for ``inverse`` (same walk order as
    operators._candidates, so index k means the same node). A
    negative int literal counts as one site; its inner Constant is
    not counted separately."""
    int_inverse = inverse in ("off_by_one", "wrong_constant")
    n = 0
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop(0)
        if int_inverse and _is_neg_int_literal(node):
            n += 1
            continue  # do not descend into the inner Constant(1)
        if inverse == "off_by_one" and _is_int_constant(node):
            n += 1
        elif inverse == "swapped_operands" and isinstance(node, ast.BinOp):
            n += 1
        elif (
            inverse == "wrong_comparator"
            and isinstance(node, ast.Compare)
            and node.ops
            and type(node.ops[0]) in _CMP_SWAP
        ):
            n += 1
        elif inverse == "swapped_boolean" and isinstance(node, ast.Constant) and type(node.value) is bool:
            n += 1
        elif inverse == "wrong_constant" and _is_int_constant(node):
            n += 1
        stack[0:0] = list(ast.iter_child_nodes(node))
    return n


def _recon_candidates(source: str) -> list[dict]:
    """Removed-return reconstruction guesses (NOT inverses).

    For each bare ``return`` (value None), one candidate: return the
    last single-Name assignment target textually before it in the
    same function. Returns a list of dicts with the candidate source
    and a description; empty when no such guess exists.
    """
    tree = ast.parse(source)
    out: list[dict] = []
    for node in _iter_dfs(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigns: list[tuple[int, str]] = []  # (lineno, name)
        bare_returns: list[ast.Return] = []
        for sub in _iter_dfs(node):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                assigns.append((sub.lineno, sub.targets[0].id))
            elif isinstance(sub, ast.Return) and sub.value is None:
                bare_returns.append(sub)
        for ret in bare_returns:
            prior = [name for ln, name in assigns if ln < ret.lineno]
            if not prior:
                continue
            name = prior[-1]
            new_tree = copy.deepcopy(tree)
            for sub in _iter_dfs(new_tree):
                if isinstance(sub, ast.Return) and sub.value is None and sub.lineno == ret.lineno:
                    sub.value = ast.Name(id=name, ctx=ast.Load())
                    break
            ast.fix_missing_locations(new_tree)
            cand = ast.unparse(new_tree)
            ast.parse(cand)
            out.append({
                "source": cand,
                "detail": f"return {name} (last assigned local before line {ret.lineno})",
                "return_lineno": ret.lineno,
                "name": name,
            })
    return out


def generate_candidates(source: str) -> list[dict]:
    """All inverse candidates for one buggy source, deduplicated by
    patched source. Each dict: inverse, site (DFS index), param,
    source, detail. Reconstruction guesses carry inverse=RECON_KEY.
    Order: inverse, then site, then param — deterministic."""
    tree = ast.parse(source)
    by_source: dict[str, dict] = {}
    for inverse in TRUE_INVERSES:
        n_sites = _count_sites(tree, inverse)
        for site in range(n_sites):
            for param in _inverse_params(inverse):
                inverter = _Inverter(inverse, site, param)
                new_tree = inverter.visit(copy.deepcopy(tree))
                ast.fix_missing_locations(new_tree)
                cand = ast.unparse(new_tree)
                ast.parse(cand)
                if cand == source:
                    continue  # no-op candidate (e.g. swapping x*x)
                if cand in by_source:
                    by_source[cand].setdefault("also_via", []).append(
                        {"inverse": inverse, "site": site, "param": param})
                    continue
                by_source[cand] = {
                    "inverse": inverse, "site": site, "param": param,
                    "source": cand,
                    "detail": f"{inverse} site {site} param {param}",
                }
    for recon in _recon_candidates(source):
        if recon["source"] == source or recon["source"] in by_source:
            continue
        by_source[recon["source"]] = {
            "inverse": RECON_KEY, "site": recon["return_lineno"],
            "param": recon["name"], "source": recon["source"],
            "detail": recon["detail"],
        }
    return list(by_source.values())


# ---------------------------------------------------------------------------
# Inverse validation (local synthetic examples only, never held-out).
# ---------------------------------------------------------------------------
def validate_inverses() -> int:
    from generator import generate_single  # noqa: E402
    from build_heldout_synthetic import TEMPLATES  # noqa: E402

    lo, hi = wilson(0, 10)
    assert abs(lo - 0.0) < 1e-9 and abs(hi - 0.28) < 0.005, (lo, hi)
    lo, hi = wilson(0, 8)
    assert abs(lo - 0.0) < 1e-9 and abs(hi - 0.32) < 0.005, (lo, hi)
    print("[validate] wilson self-check passed "
          "(0/10 -> 0.00-0.28, 0/8 -> 0.00-0.32)")

    failures = []
    checked = 0
    for _name, source, _mf, _mp, _hf, _hp in TEMPLATES:
        for op in OPERATORS:
            for seed in (0, 1, 3):
                try:
                    rec = generate_single(source, op, seed)
                except Exception:
                    continue
                buggy, fixed = rec["buggy_code"], rec["fixed_code"]
                if buggy == fixed:
                    continue
                checked += 1
                fixed_dump = ast.dump(ast.parse(fixed))
                cands = generate_candidates(buggy)
                recovered = any(
                    ast.dump(ast.parse(c["source"])) == fixed_dump
                    for c in cands
                )
                matched = [c for c in cands
                           if ast.dump(ast.parse(c["source"])) == fixed_dump]
                if op in TRUE_INVERSES:
                    if not recovered:
                        failures.append((op, seed, _name, "inverse set missed fixed source"))
                    elif matched and matched[0]["inverse"] != op:
                        failures.append((op, seed, _name,
                                         f"recovered via {matched[0]['inverse']} not {op}"))
                else:
                    # Destructive operators: the true inverse does not
                    # exist; record whether anything recovered it.
                    if recovered:
                        print(f"[validate] note: {op} seed {seed} ({_name}) "
                              f"recovered via {matched[0]['inverse']} "
                              f"({matched[0]['detail']})")
    print(f"[validate] checked {checked} generated (operator, seed) cases")
    if failures:
        for f in failures:
            print("[validate] FAIL:", f)
        return 1
    print("[validate] all invertible operators recover the fixed source; "
          "destructive operators have no true inverse (as designed)")
    return 0


# ---------------------------------------------------------------------------
# Frozen set + search
# ---------------------------------------------------------------------------
def load_frozen_set() -> list[dict]:
    records = [json.loads(line) for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    singles = [r for r in records if not r["task_id"].startswith("heldout-compound")]
    seen: Counter = Counter()
    frozen: list[dict] = []
    for rec in singles:
        op = rec["bug_class"]
        if seen[op] < FROZEN_PER_OPERATOR:
            frozen.append(rec)
            seen[op] += 1
    assert len(frozen) == 32, f"frozen set has {len(frozen)} tasks, expected 32"
    assert all(seen[op] == 4 for op in OPERATORS), dict(seen)
    return frozen


def load_all116() -> list[dict]:
    """All 116 held-out tasks in file order (107 singles + 9 compounds).

    Compounds carry a single-operator bug_class (the labelled operator);
    the record carries no second-operator field, so "matched inverse"
    keeps the frozen-32 rule (inverse == bug_class) and compounds are
    also reported separately under by_kind."""
    records = [json.loads(line) for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 116, f"held-out file has {len(records)} tasks, expected 116"
    return records


def _write_task_files(tmpdir: Path, files: dict[str, str], task_source: str) -> None:
    for rel, content in files.items():
        full = tmpdir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(task_source if rel == "task.py" else content, encoding="utf-8")


def _clear_pycache(tmpdir: Path) -> None:
    """Remove every __pycache__ under tmpdir before a test run.

    CPython validates a .pyc by (source mtime in whole seconds,
    source size). Candidates are rewritten many times per second at
    identical byte sizes, so a stale .pyc from the previous candidate
    loads as valid and the new candidate is never actually tested —
    verified experimentally: the true fixed source FAILED its own
    failing test after a same-size, same-second rewrite, and passed
    only once the cache was cleared. PYTHONDONTWRITEBYTECODE (set in
    main) stops new .pyc creation; clearing removes any already
    written. Without this, the search silently tests stale code.
    NOTE: the same hazard applies to harness/test_runner.run_one_test
    generally — eval_dev.py reuses one tmpdir per task across rungs,
    so its pilot/gate results may carry the same flaw. Flagged, not
    fixed here (out of scope for this baseline).
    """
    import shutil

    for pyc in tmpdir.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)


def run_search(dry_run: bool = False, task_set: str = "frozen32") -> dict:
    if task_set == "frozen32":
        tasks = load_frozen_set()
        set_label = "frozen set"
    elif task_set == "all116":
        tasks = load_all116()
        set_label = "all 116 held-out tasks"
    else:
        raise ValueError(f"unknown task set: {task_set!r}")
    print(f"[search] {set_label}: {len(tasks)} tasks "
          f"({', '.join(f'{op}={sum(1 for r in tasks if r['bug_class']==op)}' for op in OPERATORS)})")
    t_start = time.monotonic()
    per_task: list[dict] = []
    cand_counts: Counter = Counter()
    for rec in tasks:
        task_id, op = rec["task_id"], rec["bug_class"]
        buggy = rec["files"]["task.py"]
        cands = generate_candidates(buggy)
        cand_counts.update(c["inverse"] for c in cands)
        entry: dict = {
            "task_id": task_id,
            "bug_class": op,
            "candidates_tried": len(cands),
            "candidates_by_inverse": dict(Counter(c["inverse"] for c in cands)),
            "baseline_failing_passed_on_buggy": None,
            "fixing": [],  # one record per fixing candidate
            "fixed_by_any_inverse": False,
            "fixed_by_matched_inverse": False,
            "fixed_by_reconstruction_guess": False,
            "gate_accepted": False,
        }
        if dry_run:
            per_task.append(entry)
            print(f"[search] {task_id} ({op}): {len(cands)} candidates (dry run)")
            continue
        with tempfile.TemporaryDirectory(prefix="search_baseline_") as td:
            tmpdir = Path(td)
            _write_task_files(tmpdir, rec["files"], buggy)
            _clear_pycache(tmpdir)
            base = run_one_test(rec["failing_test_id"], str(tmpdir))
            entry["baseline_failing_passed_on_buggy"] = base.passed
            if base.passed:
                entry["note"] = ("BUG: failing test passes on unpatched "
                                 "buggy code; task is not discriminating")
            for cand in cands:
                _write_task_files(tmpdir, rec["files"], cand["source"])
                _clear_pycache(tmpdir)
                res = run_one_test(rec["failing_test_id"], str(tmpdir))
                if not res.passed:
                    continue
                _clear_pycache(tmpdir)
                reg = run_regression_suite(rec["regression_test_ids"], str(tmpdir))
                fix = {
                    "inverse": cand["inverse"],
                    "site": cand["site"],
                    "param": cand["param"],
                    "detail": cand["detail"],
                    "regressions_passed": reg.all_passed,
                    "regressions_run": len(reg.results),
                    "regressions_failed": reg.failed_tests,
                }
                entry["fixing"].append(fix)
                if cand["inverse"] == op:
                    entry["fixed_by_matched_inverse"] = True
                if cand["inverse"] == RECON_KEY:
                    entry["fixed_by_reconstruction_guess"] = True
                if cand["inverse"] in TRUE_INVERSES:
                    entry["fixed_by_any_inverse"] = True
            entry["gate_accepted"] = any(
                f["inverse"] in TRUE_INVERSES and f["regressions_passed"]
                for f in entry["fixing"])
        per_task.append(entry)
        n_fix = len(entry["fixing"])
        print(f"[search] {task_id} ({op}): {len(cands)} candidates, "
              f"{n_fix} fixing, gate_accepted={entry['gate_accepted']}")
    total_secs = time.monotonic() - t_start

    # Aggregates: primary = fixed by ANY true inverse (failing test
    # passes); gate-accepted = failing passes AND regressions green.
    def _aggregate(rows: list[dict]) -> dict:
        n = len(rows)
        fixed_any = sum(1 for e in rows if e["fixed_by_any_inverse"])
        fixed_matched = sum(1 for e in rows if e["fixed_by_matched_inverse"])
        gate = sum(1 for e in rows if e["gate_accepted"])
        recon = sum(1 for e in rows if e["fixed_by_reconstruction_guess"])
        lo, hi = wilson(fixed_any, n)
        mlo, mhi = wilson(fixed_matched, n)
        return {
            "n": n, "fixed_any_inverse": fixed_any,
            "fixed_matched_inverse": fixed_matched,
            "gate_accepted": gate,
            "fixed_by_reconstruction_guess": recon,
            "wilson_95_any_inverse": [round(lo, 2), round(hi, 2)],
            "wilson_95_matched_inverse": [round(mlo, 2), round(mhi, 2)],
        }

    by_op: dict[str, dict] = {}
    for op in OPERATORS:
        rows = [e for e in per_task if e["bug_class"] == op]
        if rows:
            by_op[op] = _aggregate(rows)
    # Singles vs compounds (compounds carry two bugs but a single
    # bug_class label; a single inverse can fix at most one of them).
    by_kind: dict[str, dict] = {}
    for kind, rows in (
        ("singles", [e for e in per_task if not e["task_id"].startswith("heldout-compound")]),
        ("compounds", [e for e in per_task if e["task_id"].startswith("heldout-compound")]),
    ):
        if rows:
            by_kind[kind] = _aggregate(rows)
    total_fixed = sum(e["fixed_by_any_inverse"] for e in per_task)
    total_matched = sum(e["fixed_by_matched_inverse"] for e in per_task)
    total_gate = sum(e["gate_accepted"] for e in per_task)
    lo, hi = wilson(total_fixed, len(per_task))
    mlo, mhi = wilson(total_matched, len(per_task))
    if task_set == "frozen32":
        method_n = "each of the 32 frozen held-out singles"
        set_rule = ("first 4 singles per operator in file order, "
                    "per preregistration/KILL_TEST_READINESS.md")
    else:
        method_n = ("each of the 116 held-out tasks (107 singles + "
                    "9 compounds, in file order)")
        set_rule = "all 116 tasks in file order from dataset/dev_tasks_heldout.jsonl"
    report = {
        "method": ("inverse-operator search baseline, NOT learned repair: "
                   f"for {method_n}, every "
                   "single-operator inverse is tried exhaustively over "
                   "every candidate site in task.py (seed-blind); a "
                   "candidate counts as a fix only if the failing test "
                   "passes under harness/test_runner.run_one_test. "
                   "dropped_condition and removed_assignment have no "
                   "deterministic inverse (information destroyed) and "
                   "were not attempted; removed_return has no true "
                   "inverse — a labelled reconstruction guess "
                   "(return last assigned local) was tried separately "
                   "and is never counted as an inverse fix."),
        "heldout_file": "dataset/dev_tasks_heldout.jsonl",
        "task_set": task_set,
        "task_set_rule": set_rule,
        "task_ids": [e["task_id"] for e in per_task],
        "n_tasks": len(per_task),
        "total_candidates_tried": sum(e["candidates_tried"] for e in per_task),
        "candidates_by_inverse": dict(cand_counts),
        "total_fixed_any_inverse": total_fixed,
        "total_fixed_matched_inverse": total_matched,
        "total_gate_accepted": total_gate,
        "wilson_95_pooled_any_inverse": [round(lo, 2), round(hi, 2)],
        "wilson_95_pooled_matched_inverse": [round(mlo, 2), round(mhi, 2)],
        "by_operator": by_op,
        "by_kind": by_kind,
        "elapsed_secs": round(total_secs, 1),
        "dry_run": dry_run,
        "per_task": per_task,
    }
    if task_set == "frozen32":
        # Back-compat keys for the original frozen-32 report format.
        report["frozen_set_rule"] = set_rule
        report["frozen_task_ids"] = report["task_ids"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-set", choices=("frozen32", "all116"),
                        default="frozen32")
    parser.add_argument("--validate-inverses", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=HERE / "search_baseline_results.json")
    args = parser.parse_args(argv)
    # Belt-and-braces with _clear_pycache: no .pyc is ever written by
    # the child test processes, so no stale bytecode can be read.
    import os

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.validate_inverses:
        return validate_inverses()
    report = run_search(dry_run=args.dry_run, task_set=args.task_set)
    if not args.dry_run:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[search] wrote {args.out}")
    n = report["n_tasks"]
    print(f"[search] total fixed (any inverse): "
          f"{report['total_fixed_any_inverse']}/{n} "
          f"(matched {report['total_fixed_matched_inverse']}/{n}, "
          f"gate-accepted {report['total_gate_accepted']}/{n}), "
          f"Wilson pooled any {report['wilson_95_pooled_any_inverse']}, "
          f"matched {report['wilson_95_pooled_matched_inverse']}, "
          f"{report['total_candidates_tried']} candidates, "
          f"{report['elapsed_secs']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
