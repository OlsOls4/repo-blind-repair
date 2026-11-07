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
Gate 2 probe. The EOS fix works (pilot median 13.8 s/task vs
baseline 130.9 s) — the adapter simply fixes nothing. Notebook:
https://www.kaggle.com/code/olsols4/first-gpu-session-gemma-4-baseline-stage-1-v19/edit
(3 saved versions).

Caveat (also in the paper): the saved run log does not prove the
kill pilot ran with the cache fix, so 0/32 may include false
negatives. The parsable rate (12.5%) is generation-side and
unaffected.
