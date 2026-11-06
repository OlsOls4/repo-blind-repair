"""Inference-only repair agent: scope ladder + regression gate, no adapter.

Entry point for the inference-only variant of Repo-Blind Repair. It
wires together the existing harness — no new control flow is invented
here:

  trace_extractor -> scope_ladder.discover_rungs -> context_packer
  -> patcher_stub.generate_candidates (trained-format model path if
  a model callable is supplied: one full fixed file generated per
  implicated source file, diffed into a patch — else deterministic
  one-line heuristics)
  -> regression_gate.evaluate_patch (failing test must pass AND the
  capped regression suite, max 15 tests / 90 s, must stay green)

A candidate is kept ONLY if the gate accepts it. Candidates are tried
in order (model candidates first, then heuristics in source order);
the gate reports the accepted patch's diff size (lines added +
removed). The model path uses the TRAINED format only (see
``patcher_stub`` "Format alignment"): the legacy FILE:/OLD:/NEW:
parser is retained for backwards compatibility but is no longer
used for model output. The dev-set evaluation (``kaggle/eval_dev.py``)
does not use this agent's heuristic fallback at all — its fix rate
is the adapter's alone.

No training, no adapter, no network, no teacher model, no paid API.
The base model is used only if the caller supplies ``model_fn``; with
no model supplied, the agent runs on heuristics alone and is honest
about that (see ``patcher_stub`` limits). Nothing here claims a repair
rate, a score, or a timing: those are UNMEASURED for this variant.

Usage (from harness/, against a repo copy — the agent patches in place
only on gate acceptance, and rolls back every rejected candidate):

    python3 inference_agent.py <repo_root> \\
        --failing-test-id tests.test_x.TestX.test_fails \\
        --regression tests.test_x.TestX.test_ok \\
        [--issue-text "add() returns ..."] \\
        [--test-output failing-output.txt]

If --test-output is omitted, the failing test is run once to capture
its output for the trace extractor. Exit 0 if a patch is accepted,
1 otherwise. The repo is left unchanged unless a patch is accepted.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context_packer import DEFAULT_RUNG_TIME_BUDGETS, DEFAULT_TOKEN_BUDGET, pack_context  # noqa: E402
from patcher_stub import FileEdit, apply_edits, generate_candidates  # noqa: E402
from regression_gate import GateResult, evaluate_patch  # noqa: E402
from scope_ladder import discover_rungs  # noqa: E402
from test_runner import run_one_test  # noqa: E402
from trace_extractor import TraceInfo, extract_trace  # noqa: E402


@dataclass
class InferenceResult:
    accepted: bool
    rung: int | None = None
    candidate: list[FileEdit] | None = None
    gate: GateResult | None = None
    candidates_tried: int = 0
    rung_secs: dict[int, float] = field(default_factory=dict)
    reason: str = ""


def run_inference_only(
    repo_root: str,
    trace: TraceInfo,
    failing_test_id: str,
    regression_test_ids: list[str],
    model_fn=None,
    max_candidates_per_rung: int = 50,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    rung_time_budgets: dict[int, int] | None = None,
) -> InferenceResult:
    """Try inference-only candidates rung by rung until the gate accepts.

    Rung budgets default to 30/60/120 s (rung 1/2/3). A patch that
    passes the gate but exceeds its rung budget is rolled back and
    the ladder climbs, matching ``scope_ladder.run_ladder`` semantics.
    Rejected candidates are already rolled back by the gate; the repo
    is therefore unchanged unless a patch is accepted (or a rollback
    itself fails, which is reported in ``reason`` and never hidden).
    """
    budgets = dict(DEFAULT_RUNG_TIME_BUDGETS)
    if rung_time_budgets:
        budgets.update(rung_time_budgets)

    # Read repo .py files the same way the ladder does (via discover_rungs
    # inputs): reuse the ladder's file discovery by re-reading here.
    all_files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, repo_root)
            try:
                with open(full, encoding="utf-8") as fh:
                    all_files[rel] = fh.read()
            except (OSError, UnicodeDecodeError):
                continue

    rungs = discover_rungs(repo_root, trace)
    result = InferenceResult(accepted=False, reason="no candidate passed the gate")
    tried = 0

    for rung in (1, 2, 3):
        start = time.monotonic()
        pack = pack_context(
            {r: {p: all_files[p] for p in rungs[rung] if p in all_files} for r in (rung,)},
            token_budget=token_budget,
            rung_time_budgets=budgets,
        )
        candidates = generate_candidates(
            pack.text, rung, model_fn=model_fn,
            max_candidates=max_candidates_per_rung,
        )
        for candidate in candidates:
            if time.monotonic() - start > budgets[rung]:
                result.reason = f"rung {rung} time budget ({budgets[rung]}s) exceeded"
                break
            tried += 1
            gate = evaluate_patch(repo_root, candidate, failing_test_id, regression_test_ids)
            elapsed = time.monotonic() - start
            if gate.accepted and elapsed > budgets[rung]:
                # Out-of-budget win is not a win: roll back, keep climbing.
                reverse = [FileEdit(e.path, e.new_text, e.old_text) for e in candidate]
                try:
                    apply_edits(repo_root, reverse)
                except ValueError:
                    pass
                result.reason = f"rung {rung} patch passed but exceeded {budgets[rung]}s budget"
                continue
            if gate.accepted:
                result.accepted = True
                result.rung = rung
                result.candidate = candidate
                result.gate = gate
                result.candidates_tried = tried
                result.rung_secs[rung] = elapsed
                result.reason = f"accepted at rung {rung}: {gate.reason}"
                return result
            result.gate = gate
            result.reason = f"rung {rung} rejected: {gate.reason}"
        result.rung_secs[rung] = time.monotonic() - start
    result.candidates_tried = tried
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", help="repo to repair (patched in place only on acceptance)")
    parser.add_argument("--failing-test-id", required=True)
    parser.add_argument("--regression", action="append", default=[],
                        help="regression test id (repeatable)")
    parser.add_argument("--issue-text", default="")
    parser.add_argument("--test-output", default="",
                        help="file with failing-test output; if omitted, the failing test is run once")
    args = parser.parse_args(argv)

    if args.test_output:
        with open(args.test_output, encoding="utf-8") as fh:
            output = fh.read()
    else:
        output = run_one_test(args.failing_test_id, args.repo_root).output
    trace = extract_trace(output, args.issue_text)
    result = run_inference_only(
        args.repo_root, trace, args.failing_test_id, args.regression,
    )
    print(f"Accepted: {result.accepted} at rung {result.rung} — {result.reason}")
    print(f"Candidates tried: {result.candidates_tried}")
    print(f"Rung seconds: {result.rung_secs}")
    if result.candidate is not None:
        for edit in result.candidate:
            print(f"Patch: {edit.path}: {edit.old_text!r} -> {edit.new_text!r}")
    if result.gate is not None:
        print(f"Gate: diff +{result.gate.lines_added}/-{result.gate.lines_removed} "
              f"lines, {result.gate.files_changed} file(s)")
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
