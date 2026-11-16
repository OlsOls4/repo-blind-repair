# Clean-clone verification log — 15 Nov 2026

Fresh copy of the staged tree (`publish/repo/` as it will be pushed,
63 files + the edits in this sprint) copied to a temp dir with no
`__pycache__`/`.pyc`, dataset placed alongside as a sibling `dataset/`
(the layout `reproduce.py` expects for a real clone). CPU only, stdlib
only, no network, no GPU. Run Sun 15 Nov 2026, ~11:00–11:20 GMT.

## Commands

```bash
cp -r publish/repo /tmp/cleanclone/repo-blind-repair
cp -r publish/dataset /tmp/cleanclone/dataset
cd /tmp/cleanclone/repo-blind-repair
python3 reproduce.py
python3 data_generation/validate_scaled_dataset.py --data-dir /tmp/cleanclone/dataset
python3 reproduce_search.py
```

## Results

### `reproduce.py` — FAILED (sample regeneration)

- Records: 20,000 (provenance entries: 20,000). Sources rebuilt: 2,126.
- Singles per operator: exactly 1,750 each (all 8 operators).
  Composites: 6,000. Schema/parse/buggy≠fixed checks: pass.
- **Sample regeneration: 998/1,000 exact (2 mismatches)** —
  record 4634 (`wrong_comparator`, source 553, seed 7) and record
  17855 (`composite:off_by_one+wrong_comparator`, source 573,
  seed 5000).
- Unit tests: mutation **10/10 OK**, harness **33/33 OK**.

### `validate_scaled_dataset.py` (full 20,000-record regeneration) — FAILED

- `AssertionError: 107 regeneration mismatches` (of 20,000; 0.5%).
- Breakdown by bug class: `wrong_comparator` 34;
  `composite:off_by_one+wrong_comparator` 23;
  `composite:wrong_comparator+wrong_constant` 23;
  `composite:removed_assignment+wrong_comparator` 23;
  `composite:swapped_boolean+wrong_comparator` 4.
- **Every mismatch involves `wrong_comparator`; no other operator
  mismatches.**

### `reproduce_search.py` — PASSED

- **77/116 any-inverse, 71/116 matched-inverse, 77 gate-accepted**,
  6,713 candidates, 770.1 s; Wilson any [0.57, 0.74], matched
  [0.52, 0.70]. Matches the recorded 8 Nov 2026 results exactly.

## Cause (traced, not guessed)

`apply_mutation` selects `target_index = seed % len(candidates)`.
The 8 Nov D1 fix corrected the comparator inverter in the mutator
(`mutation/operators.py`) after the dataset was generated, changing
the `wrong_comparator` candidate enumeration/order: the same
(source, operator, seed) now selects a different candidate than at
generation time (e.g. record 4634: 8 candidates, seed 7 picks index 7
`level >= needed`; the record holds index 6 `role == 'admin'`).
The 31 Oct "0 mismatches" verification predates the D1 fix and is
no longer reproducible with the current pipeline.

## What this means

- The byte-exact regeneration claim is narrowed, not destroyed:
  19,893/20,000 records (99.5%) regenerate exactly; the 107
  affected records remain valid (buggy ≠ fixed, both parse, schema
  and counts pass). The search-baseline headline (77/116) reproduces
  exactly, and both unit suites pass.
- The README and `data/train_records_20k_REPORT.md` now disclose
  the gap (see the 15 Nov 2026 notes there) instead of claiming
  0 mismatches with the current code.
- Open decision for the parent: disclose as-is (done here) vs
  regenerate the 20,000-record dataset with the fixed pipeline
  (new dataset version, re-validation, Kaggle re-upload, paper
  numbers stale — a material change, not taken here).

## What was NOT verified in this run

- Kill-test per-task log: NOT RECOVERABLE (verdict 15 Nov 2026).
  Exhaustive local search found no copy of `baseline_n32.json` /
  `kill_pilot_n32.json` or the per-task printed lines; they survive
  only as printed cell output in the notebook's saved versions,
  retrievable solely by authenticated access. Full disclosure is in
  `preregistration/README.md` (and, in shorter form, in the paper).
  No log was invented.
- Day J daily outcome (SUCCEEDED, Public Score 0.00, checked Mon
  16 Nov ~09:27; see the `[Day J outcome, checked Mon 16 Nov ~09:27]`
  line in `README.md`).
