# Scaled training dataset — validation report

Generated: deterministic, local only (no Kaggle, no network, no cost).
Generator: `generate_scaled_dataset.py` (re-run reproduces the file;
provenance sidecar regenerates every record exactly — see check 8).

## Licence / provenance basis — route (a) ONLY
All 20,000 records are deterministic AST mutations of toy programs
written for this task (82 parameterised template shapes: 62 general
+ 20 boolean-rich, plus composed 3-function modules). No real human
bug fixes, no teacher model output, no third-party code. The
permissively-licensed route (b) was NOT used, so no licence
verification was needed and no external licence is claimed.

## File
- `train_records_20k.jsonl`: 20000 records, 24055389 bytes
- `train_records_20k_provenance.jsonl`: 20000 entries
  (source_index, operators, seed per record)
- Pilot file untouched: `pilot/pilot_records.jsonl` (225 records)

## Checks — ALL PASSED
1. Record count 20,000; schema keys exactly
   {buggy_code, fixed_code, mutations, failing_test, bug_class}.
2. Every buggy_code and fixed_code parses (ast.parse).
3. buggy_code != fixed_code on all 20,000 records.
4. failing_test == "" on all records (no test invented).
5. bug_class matches mutations on all records.
6. Singles: exactly 1,750 per operator:
   {"dropped_condition": 1750, "off_by_one": 1750, "removed_assignment": 1750, "removed_return": 1750, "swapped_boolean": 1750, "swapped_operands": 1750, "wrong_comparator": 1750, "wrong_constant": 1750}.
7. Composites: 6,000 (30.0% of total). Pair counts (uneven — pairs
   differ in applicability, not a requirement):
   {"composite:dropped_condition+off_by_one": 363, "composite:dropped_condition+swapped_operands": 211, "composite:dropped_condition+wrong_constant": 367, "composite:off_by_one+wrong_comparator": 707, "composite:removed_assignment+off_by_one": 594, "composite:removed_assignment+wrong_comparator": 592, "composite:removed_return+dropped_condition": 360, "composite:removed_return+wrong_constant": 702, "composite:swapped_boolean+off_by_one": 165, "composite:swapped_boolean+wrong_comparator": 177, "composite:swapped_operands+removed_return": 494, "composite:swapped_operands+wrong_constant": 534, "composite:wrong_comparator+wrong_constant": 709, "composite:wrong_constant+swapped_operands": 25}.
8. buggy_code distinct across the file and distinct from all 225
   pilot buggy_codes (no padding with repeats).
9. Every fixed_code is exactly one of the 2126 deterministically
   rebuilt sources.
10. FULL determinism: all 20,000 records regenerated from provenance
    through the unchanged mutation pipeline match the file exactly
    (0 mismatches).

## Token-length survey (ESTIMATE — read the caveat)
Method: chars/4 on the training-text format
(`Fix this Python bug (bug class: ...)` + buggy + `Fixed code:` +
fixed), because no Gemma tokenizer exists in this VM.
- Scaled set estimate: median 178, p90 422, max 669.
- Same estimator on the pilot file: median 181, p90 275, max 317,
  vs its known REAL-tokenizer survey median 243, p90 317, max 383
  (ratios ~1.34 / ~1.15 / ~1.21). Calibrated real-tokenizer guess
  for the scaled set: median ~240, p90 ~490, max ~810.
- Flag: the scaled median matches the pilot's, but the p90/max tail
  is longer (composed 3-function modules are longer than pilot
  sources). The pilot step-time sizing (0.332 s/micro-step) was
  measured on short sequences, so the full-train run MUST re-survey
  with the real tokenizer and re-measure step time on the first
  chunk before later chunks commit to a step budget. Max stays
  well under the 1024-token warning threshold and the 32K context.

## What this does NOT claim
- No Kaggle timings and no quality results are claimed for this
  dataset. Whether it transfers to real bugs is measured later on
  the dev set, not asserted here.
- Scorer vLLM serving of a trained adapter remains UNPROVEN
  (Gate 2 Stage 3 UNTESTED); this dataset does not change that.
