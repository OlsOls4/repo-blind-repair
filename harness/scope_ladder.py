"""Scope ladder controller: try the narrowest context first, widen on failure.

Rung 1 — trace files only (files named by the failing traceback / issue).
Rung 2 — imports and callers of rung-1 files (via AST: import statements
         plus references to traced symbols), and files rung-1 files import.
Rung 3 — wider search: every other Python file in the repo.

Each rung is time-boxed (default 30s / 60s / 120s, from
:mod:`context_packer`). A rung that exceeds its budget is treated as a
failure even if its patch passed the gate, and the patch is rolled back —
a slow fix that eats the 12-hour scoring budget is not a win.

``patch_fn`` has the signature ``(context_text, rung) -> list[FileEdit]``
(or ``None`` for "no patch at this rung"). By default it is the
:mod:`patcher_stub` placeholder, which raises NotImplementedError until
the Gemma LoRA generator exists. Demos/tests pass their own patch_fn.
"""
from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field

from context_packer import (
    DEFAULT_RUNG_TIME_BUDGETS,
    DEFAULT_TOKEN_BUDGET,
    ContextPack,
    pack_context,
)
from patcher_stub import FileEdit, apply_edits, generate_patch
from regression_gate import GateResult, evaluate_patch
from trace_extractor import TraceInfo


@dataclass
class LadderResult:
    accepted: bool
    rung: int | None  # rung that produced the accepted patch, if any
    gate: GateResult | None = None
    rung_files: dict[int, list[str]] = field(default_factory=dict)
    packs: dict[int, ContextPack] = field(default_factory=dict)
    rung_secs: dict[int, float] = field(default_factory=dict)
    reason: str = ""


def _read_repo_py_files(repo_root: str) -> dict[str, str]:
    """Return {relative_path: content} for every .py file under repo_root."""
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, repo_root)
            try:
                with open(full, encoding="utf-8") as fh:
                    files[rel] = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
    return files


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def discover_rungs(repo_root: str, trace: TraceInfo) -> dict[int, list[str]]:
    """Compute rung file lists for a repo + trace."""
    all_files = _read_repo_py_files(repo_root)

    # Rung 1: implicated files that actually exist in the repo (match by
    # relative path, by basename, or by absolute path inside the repo).
    rung1: list[str] = []
    for implicated in trace.implicated_files:
        candidates = [implicated]
        if os.path.isabs(implicated):
            candidates.append(os.path.relpath(implicated, repo_root))
        candidates.append(os.path.basename(implicated))
        for cand in candidates:
            if cand in all_files and cand not in rung1:
                rung1.append(cand)
    # Also: the failing test's own module is never a fix target, but the
    # source file it exercises usually shares its stem (test_x -> x).
    for test_id in trace.failing_tests:
        # unittest id: tests.test_calculator.TestAdd.test_add
        parts = test_id.split(".")
        for part in parts:
            if part.startswith("test_"):
                src = part[5:] + ".py"
                for path in all_files:
                    if os.path.basename(path) == src and path not in rung1:
                        rung1.append(path)

    rung1_set = set(rung1)
    rung1_stems = {_stem(p) for p in rung1}
    symbols = set(trace.implicated_symbols)

    # Rung 2: files that import a rung-1 module, are imported by a rung-1
    # file, or reference a traced symbol.
    rung2: list[str] = []
    parsed: dict[str, ast.AST] = {}
    for path, content in all_files.items():
        try:
            parsed[path] = ast.parse(content)
        except SyntaxError:
            continue
    for path, tree in parsed.items():
        if path in rung1_set:
            continue
        imported_names: set[str] = set()
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[0])
            elif isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
        if imported_names & rung1_stems or referenced & symbols:
            rung2.append(path)
    # Files imported BY rung-1 files.
    for path in rung1:
        tree = parsed.get(path)
        if tree is None:
            continue
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for other in all_files:
            if other not in rung1_set and _stem(other) in imported and other not in rung2:
                rung2.append(other)

    rung2_set = set(rung2)
    rung3 = sorted(p for p in all_files if p not in rung1_set and p not in rung2_set)
    return {1: sorted(rung1), 2: sorted(rung2), 3: rung3}


def run_ladder(
    repo_root: str,
    trace: TraceInfo,
    failing_test_id: str,
    regression_test_ids: list[str],
    patch_fn=generate_patch,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    rung_time_budgets: dict[int, int] | None = None,
) -> LadderResult:
    """Run the ladder until a patch passes the regression gate."""
    budgets = dict(DEFAULT_RUNG_TIME_BUDGETS)
    if rung_time_budgets:
        budgets.update(rung_time_budgets)

    all_files = _read_repo_py_files(repo_root)
    rungs = discover_rungs(repo_root, trace)
    result = LadderResult(accepted=False, rung=None, rung_files=rungs)

    for rung in (1, 2, 3):
        start = time.monotonic()
        pack = pack_context(
            {r: {p: all_files[p] for p in rungs[r] if p in all_files} for r in (rung,)},
            token_budget=token_budget,
            rung_time_budgets=budgets,
        )
        result.packs[rung] = pack
        edits = patch_fn(pack.text, rung)
        if not edits:
            result.rung_secs[rung] = time.monotonic() - start
            continue
        gate = evaluate_patch(repo_root, edits, failing_test_id, regression_test_ids)
        elapsed = time.monotonic() - start
        result.rung_secs[rung] = elapsed
        if gate.accepted and elapsed > budgets[rung]:
            # Out-of-budget win is not a win: roll back and keep climbing.
            reverse = [FileEdit(e.path, e.new_text, e.old_text) for e in edits]
            try:
                apply_edits(repo_root, reverse)
            except ValueError:
                pass
            result.reason = f"rung {rung} patch passed but exceeded {budgets[rung]}s budget"
            continue
        if gate.accepted:
            result.accepted = True
            result.rung = rung
            result.gate = gate
            result.reason = f"accepted at rung {rung}: {gate.reason}"
            return result
        result.gate = gate
        result.reason = f"rung {rung} rejected: {gate.reason}"

    return result
