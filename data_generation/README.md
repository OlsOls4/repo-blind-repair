# Data generation — regenerate the corpus from the sidecar

Stdlib only, CPU only, no network. These scripts are the exact
generator and validator used for dataset v18 (20,000 records), with
two mechanical patches for publication (see `../PATCHES.md`):
the hardcoded local `mutation/` path is now repo-relative, and the
validator accepts `--data-dir`.

## Files

- `generate_scaled_dataset.py` — builds the corpus: 82 parameterised
  template shapes (62 general + 20 boolean-rich) rebuilding 2,126
  sources, of which 1,166 distinct sources appear in the 20,000
  records, mutated into 20,000 records (14,000 singles at exactly
  1,750 per operator + 6,000 composites across 14 pairs), deduplicated
  by `buggy_code` (also against the pilot file). Writes
  `train_records_20k.jsonl` + `train_records_20k_provenance.jsonl`.
  Run: `python3 generate_scaled_dataset.py [--out DIR]` (default: this
  directory). A few minutes on CPU.
- `validate_scaled_dataset.py` — independent validator: record count
  and schema, every file parses, buggy != fixed, `failing_test` empty,
  bug-class consistency, exact singles/composites counts, distinctness
  (including against the pilot), every `fixed_code` one of the rebuilt
  sources, and FULL determinism (every record regenerated from the
  provenance sidecar and compared field-for-field: 0 mismatches).
  Run: `python3 validate_scaled_dataset.py [--data-dir DIR]` (default:
  this directory). Writes `train_records_20k_REPORT.md` next to the
  dataset. A few minutes on CPU.
- `build_heldout_synthetic.py` — builds the 116-task held-out
  synthetic evaluation set from NEW domains/shapes not present in the
  training generator, with derived tests verified fail-on-buggy /
  pass-on-fixed. Run:
  `python3 build_heldout_synthetic.py --out dev_tasks_heldout.jsonl
  [--per-operator 20] [--train-file train_records_20k.jsonl]`.
  Label every number from this file "held-out synthetic" — never a
  dev-set or leaderboard result.
- `audit_near_dup.py` + `audit_near_dup_results.json` — near-duplicate
  audit (5 Nov 2026): token-trigram containment of each held-out task
  against the training sources. Result: max containment 0.39, 0/116
  above 0.50, 0 exact matches. Run it from a directory holding
  `train_records_20k.jsonl` and `dev_tasks_heldout.jsonl` next to the
  script; it rewrites the results JSON. (The generator rebuilds 2,126
  sources; 1,166 distinct sources appear in the records — the audit
  compares against the 1,166 actually present.) The shipped results
  JSON has its two input-path fields scrubbed to bare filenames; the
  script writes absolute paths when re-run, so scrub any fresh
  results file before committing.
- `pilot/pilot_records.jsonl` — the 225-record pilot set, used by the
  generator/validator only to prove the scaled set does not repeat
  pilot records. Also a handy small sample.

## Provenance model

Each record is reproducible from `(source_index, operators, seed)`:
`generate_record(sources[source_index], operators, seed)` through the
unchanged `mutation/` pipeline reproduces the stored record exactly.
The sidecar is the corpus's audit trail — the records file is a
cache of it, not the other way round.
