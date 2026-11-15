# Pre-registration record — kill test (frozen 5 Nov 2026)

The kill test for the LoRA route was **designed 1 Nov 2026** and
**frozen 5 Nov 2026** (task list, thresholds, and runbook fixed
before the GPU quota reset on 7 Nov, so before any retraining
result existed).

- `kill_test_task_ids.txt` — the frozen n=32 task set (4 singles
  per operator, first 4 per operator in file order from the
  held-out set). Header records the freeze date.
- `KILL_TEST_READINESS.md` — the pre-registered rule: KILL if
  <4/32 fixed, or fixes in ≤1 operator, or parsable non-identical
  output <30%; CONTINUE 4–9/32 with ≥3 operators; STRONG ≥10/32.
  Plus the EOS falsifier and diagnostic arbitration.
- `EVAL_PLAN.md` — evaluation plan, gates, budgets, and the
  stop-sequence probe rule (LoRA-only adoption, 5 conditions).
- `FIRST_GPU_RUNBOOK.md` — the session runbook (baseline → Stage 1
  retrain epoch 1 → kill pilot → Gate 2 probe), quota budget, and
  stop conditions.

## Outcome (7 Nov 2026)

Baseline base model 0/32; Stage 1 v19 epoch-1 adapter kill pilot
**0/32** (parsable 4/32, 12.5%, 0 operators) — all three kill
conditions met. **Verdict: KILL the LoRA route.** No epoch 2, no
Gate 2 probe. The EOS fix works (kill-pilot mean 22.4 s/task vs
baseline 130.9 s/task) — the adapter simply fixes nothing. Notebook:
https://www.kaggle.com/code/olsols4/first-gpu-session-gemma-4-baseline-stage-1-v19/edit
(3 saved versions; link resolves for the notebook owner only —
the notebook is private, so it returns 404 to anyone else).

Caveat (also in the paper): the bytecode path is verified — the
kill pilot ran the fixed harness (cache-clearing verified, Kaggle
copy byte-identical to the fixed file; see MEASUREMENT_PLAN.md §1
in the workspace) — but no per-task kill log was retained in the
repo, so 0/32 rests on the notebook record, not a checked-in
results file. The parsable rate (12.5%) is generation-side and
unaffected.

What is missing, exactly (15 Nov 2026): the per-task log files
(`baseline_n32.json`, `kill_pilot_n32.json`) and the per-task
printed lines — which of the 32 frozen tasks (see
`kill_test_task_ids.txt`) each outcome belongs to, and per-task
timings. Why: the run's outputs lived in the Kaggle session's
`/kaggle/working`, which is wiped when an interactive session stops
("No persistence"), and the JSON files were never committed to a
notebook version; an exhaustive search of this machine (repo,
workspace, memory, transcripts, `/tmp`) on 15 Nov 2026 found no
copy, and no Kaggle CLI/API is available here to fetch notebook
outputs. What is retained instead: the aggregates above, the frozen
task IDs, the checkpoint reference
(`full_train_output/run-20261107T113316Z/checkpoint-opt-5000/`),
and the per-task lines as printed cell output inside the notebook's
saved versions — recoverable only by authenticated access to the
notebook (owner browser read, or `kaggle kernels output` with an
API token). Nothing was invented to fill the gap.
