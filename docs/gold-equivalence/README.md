# Gold-equivalence re-grading (attempt 7, 18 Nov 2026)

The replica diagnostic's headline — 25/32 as-graded, **17/32
gold-equivalent** — is re-runnable from this directory. No GPU, no
Kaggle, no model: both scripts run on CPU in under a second against
the stored attempt-7 data.

## Files

- `k2-kill32-summary.json` — the attempt-7 run summary: per-task
  final patch, verify result, and every turn's parsed tool calls +
  tool results (334 turns). The per-task trace files were wiped with
  the Kaggle session; this summary is the surviving record.
- `dev_tasks_heldout.jsonl` — the 116 synthetic held-out tasks the
  replica runs against (our own clean-room tasks, no competition
  data). The frozen kill-32 subset is `kill_test_task_ids.txt` in the
  build root (not shipped here).
- `grade_gold_equivalence.py` — re-grades the 8 guard tasks by probe
  against sibling-consensus gold functions (each gold file is first
  validated against the task's own tests), takes the 17 non-guard
  resolves as gold per the attempt-7 verify result + the attribution
  report's probe check, and prints the 17/32 tally.
- `verify_streak_detector.py` — applies the replica's V8 repetition
  detector (streak >= 3) to the stored turns: it must flag exactly
  the 5 strict identical-call tasks and zero of the 25 resolved
  tasks.

## Run

```bash
python3 grade_gold_equivalence.py    # expect: gold-equivalent 17/32, exit 0
python3 verify_streak_detector.py    # expect: ACCEPTANCE: PASS, exit 0
```

## Reading the result

- 25/32 is what the 3 visible tests say. 17/32 is what the hidden
  tests would say: all 8 guard tasks (4 dropped_condition, 4
  swapped_boolean) "resolve" only by deleting or weakening the guard
  the hidden tests check — e.g. `circle_area(-1)`: gold 0, patch
  3.14159. One guard task (dilution-swapped_boolean-0) is borderline:
  it differs from gold on exactly one probe where gold raises
  `ZeroDivisionError` and the patch returns 0.
- Repetition: 5/7 failures are strict identical-call loops (runs of
  14–28); isbn repeats only inside one 16,384-token generation; the
  7th (loan-removed_return-0) is a distinct fixation on float
  exactness after a 1-ulp failure. See
  `../wave11-attribution-2026-11-18.md` for the full analysis.
