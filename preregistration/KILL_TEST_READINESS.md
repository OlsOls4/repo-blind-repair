# Kill-test readiness — CPU-only check (2026-11-05)

Pre-registered kill test: retrain with end-of-sequence appended
and the bug-class label dropped (v19), then a 32-task stratified
held-out singles pilot (4 per operator). KILL the LoRA route if
fewer than 4/32 are fixed, if fixes fall in at most one operator,
or if parsable (syntactically valid Python) non-identical output
is below 30%. CONTINUE 4–9/32 with ≥3 operators. STRONG ≥10/32.

## Retrain dry-run — PASSED

Command (sizing only, no training, no model load):

```
python3 kaggle/full_train_manual_lora_v19prep.py \
  --dataset kaggle/train_records_20k.jsonl \
  --dry-run --epochs 2 --max-chunk-optimizer-steps 5000
```

Exit 0. Verbatim output:

```
=== FULL-RUN DRY-RUN PLAN (no training, no model load) ===
dataset: kaggle/train_records_20k.jsonl (N=20000, sha256=1786411c21103026...)
base: google/gemma-4-12b-it-qat-w4a16-ct (unchanged); LoRA r=8 alpha=16 lr=0.0002
epochs=2, micro-batch=1, accumulation=4, B_eff=4
total optimiser steps=10000, micro-steps (upper)=40000
sizing at 0.339s/micro-step + 60s load: stepping=13560.0s, total=13620.0s (227.0 min, 3.78 h)
quota: remaining=96300s (26.75 h); chunk budget=57780s (16.05 h, 60% of remaining; 70% cap=67410s)
chunk: budget-derived max=42566 optimiser steps; this chunk limit=5000 optimiser steps
resume: fresh run (no --resume-from). If the output run directory already exists and is non-empty, main() aborts rather than overwriting it.
DRY-RUN PASSED: argument parsing, dataset validation, sizing, chunk budget, and resume metadata (if any) all check out. This proves NOTHING about GPU step time, memory, resume in a fresh Kaggle session, or serving.
```

Caveat (do not misread the output): the dry-run's quota figures
(remaining 26.75 h) are **script defaults, not live Kaggle
state**. Live quota is 0 h available until the weekly reset (no
reset date shown). The real Stage-1 chunk (5,000 optimiser steps)
needs the reset first. The dry-run proves argument parsing,
dataset validation, sizing, and chunk budgeting only — nothing
about GPU step time, memory, resume, or serving (the script says
so itself).

## Frozen n=32 set — VERIFIED LOADABLE

File: `kaggle/dev_tasks_heldout.jsonl` — 116 records, all with
`failing_test_id` and `files` (including `tests/test_task.py`); 0
records missing either. Split by `task_id` prefix:

- Singles: **107** (every record not prefixed `heldout-compound`)
- Compounds: **9** (`heldout-compound-*`)

Singles per operator (`bug_class`):

| Operator | Singles available | Needed for kill set |
|---|---|---|
| off_by_one | 19 | 4 |
| swapped_operands | 20 | 4 |
| wrong_comparator | 6 | 4 |
| dropped_condition | 5 | 4 |
| swapped_boolean | 5 | 4 |
| wrong_constant | 15 | 4 |
| removed_return | 19 | 4 |
| removed_assignment | 18 | 4 |

Minimum per operator is 5 — a 4-per-operator draw (n=32) is
feasible from singles alone, with 1 spare in each of the two
thinnest operators (dropped_condition, swapped_boolean). Draw
deterministically (seed 42, first 4 per operator in file order)
and record the 32 task IDs before running, so the pilot is
auditable against this file.

## Verdict: READY, pending GPU quota reset

- Retrain script: dry-run passed; dataset hash matches
  (sha256 1786411c…); Stage-1 chunk (5,000 steps) planned.
- Kill set: frozen, loadable, 4/operator feasible.
- Blockers (all GPU-side, none CPU-side): Kaggle GPU quota 0 h
  until weekly reset (no date shown); Gate 2 live vLLM probe
  also waits for reset. Nothing in this file can be run until
  then. When quota returns, run Stage 1 first and stop for the
  kill pilot before epoch 2 — do not run the full 10,000 steps
  in one go.
