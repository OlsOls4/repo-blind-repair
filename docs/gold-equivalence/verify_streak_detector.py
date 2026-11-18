#!/usr/bin/env python3
"""Offline check of the V8 streak detector against attempt-7 data.

Replicates loop.py's V8 logic EXACTLY (same call_key serialisation,
same no-op-edit rule) over the stored k2-kill32-summary.json turns —
the per-task trace files were wiped with the Kaggle session, but the
summary JSON carries every turn's parsed_calls + tool_results.

Acceptance (wave-11 step 1c, as corrected by the wave-12 Diagnosis
panel): a streak >= 3 must flag exactly the 5 strict identical-call
tasks (off_by_one-0, swapped_operands-1, wrong_constant-0,
off_by_one-2, swapped_operands-6) and ZERO of the 25 resolved tasks.
isbn (in-generation repetition only) and removed_return-0 (varied
probes + comment edits) must NOT flag on the call-streak rule.

Usage: python3 verify_streak_detector.py
Exit 0 iff the acceptance holds; prints the per-task table either way.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "k2-kill32-summary.json"


def call_key(calls):
    # Identical to loop.py V8.
    return json.dumps(
        [(c.get("name"), c.get("arguments") or {}) for c in calls],
        sort_keys=True, default=str)


def analyse(task):
    max_streak = 0
    streak = 0
    prev_key = None
    noop_turns = 0
    for turn in task["turns"]:
        calls = turn.get("parsed_calls") or []
        if not calls:
            prev_key = None
            streak = 0
            continue
        key = call_key(calls)
        streak = streak + 1 if key == prev_key else 1
        prev_key = key
        max_streak = max(max_streak, streak)
        for resp in turn.get("tool_results") or []:
            result = resp.get("result") or {}
            if resp.get("name") == "edit_file" and \
                    result.get("status") == "ok" and \
                    not result.get("diff"):
                noop_turns += 1
    return max_streak, noop_turns


def main():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    expected_strict = {
        "heldout-loan-off_by_one-0",
        "heldout-loan-swapped_operands-1",
        "heldout-loan-wrong_constant-0",
        "heldout-loan-off_by_one-2",
        "heldout-loan-swapped_operands-6",
    }
    flagged = set()
    false_positives = []
    rows = []
    for task in tasks:
        max_streak, noop_turns = analyse(task)
        resolved = (task.get("verify") or {}).get("resolved", False)
        if max_streak >= 3:
            flagged.add(task["task_id"])
            if resolved:
                false_positives.append(task["task_id"])
        rows.append((task["task_id"], max_streak, noop_turns, resolved))
    print(f"{'task':42s} {'streak':>6s} {'noop':>4s}  resolved")
    for task_id, streak, noop, resolved in sorted(rows):
        mark = " <-- FLAGGED" if streak >= 3 else ""
        print(f"{task_id:42s} {streak:6d} {noop:4d}  {str(resolved):5s}{mark}")
    print()
    print(f"flagged (>=3): {len(flagged)} — expected exactly the 5 strict")
    print(f"  missing: {sorted(expected_strict - flagged) or 'none'}")
    print(f"  extra:   {sorted(flagged - expected_strict) or 'none'}")
    print(f"false positives on resolved tasks: "
          f"{false_positives or 'none'}")
    ok = flagged == expected_strict and not false_positives
    print("ACCEPTANCE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
