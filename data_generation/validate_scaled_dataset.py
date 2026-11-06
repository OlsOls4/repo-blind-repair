"""Independent validator for train_records_20k.jsonl (stdlib only).

Checks, all of which must pass (exit non-zero otherwise):
  1. Exactly 20,000 records; schema keys exactly
     {buggy_code, fixed_code, mutations, failing_test, bug_class}.
  2. Every buggy_code and fixed_code parses (ast.parse).
  3. buggy_code != fixed_code; failing_test == "" (never invented).
  4. bug_class matches mutations (single operator, or
     "composite:op1+op2"); mutation operators are all known.
  5. Singles: exactly 1,750 per operator; composites: 6,000 (30%).
  6. buggy_code distinct across the file AND distinct from every
     pilot_records.jsonl buggy_code.
  7. Every fixed_code is exactly one of the deterministically rebuilt
     sources (generate_scaled_dataset.build_sources()).
  8. FULL determinism: every record is regenerated from
     train_records_20k_provenance.jsonl (source_index, operators,
     seed) through the unchanged mutation pipeline and compared
     field-for-field with the file record.
  9. Token-length survey (chars/4 estimate on the training-text
     format, calibrated against the pilot's known real-tokenizer
     survey) — reported, not gated, except max must stay < 1024
     estimated (the full-train warning threshold).

Writes train_records_20k_REPORT.md next to the dataset and prints
a summary. No network, no training, no Kaggle.

Publication patch (vs. the competition original): main() accepts an
optional --data-dir (default: this file's directory) so the validator
can run against a published dataset directory without moving files.
The checks themselves are unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTATION_DIR = HERE.parent / "mutation"  # repo-relative (patched for publication; was a hardcoded local path)
sys.path.insert(0, str(MUTATION_DIR))
sys.path.insert(0, str(HERE))

from generator import generate_record  # noqa: E402
from operators import OPERATORS  # noqa: E402
import generate_scaled_dataset as gen  # noqa: E402

EXPECTED_KEYS = {"buggy_code", "fixed_code", "mutations", "failing_test", "bug_class"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE,
                        help="directory holding train_records_20k.jsonl and "
                             "train_records_20k_provenance.jsonl "
                             "(default: this script's directory)")
    args = parser.parse_args(argv)
    data_path = args.data_dir / "train_records_20k.jsonl"
    prov_path = args.data_dir / "train_records_20k_provenance.jsonl"
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    provs = [json.loads(line) for line in prov_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 20_000, f"record count {len(records)}"
    assert len(provs) == 20_000, f"provenance count {len(provs)}"

    sources = gen.build_sources()
    source_set = set(sources)
    pilot_buggy = gen._pilot_buggy_codes()

    counts: Counter = Counter()
    seen_buggy: set[str] = set()
    for rec in records:
        assert set(rec) == EXPECTED_KEYS, sorted(rec)
        ast.parse(rec["buggy_code"])
        ast.parse(rec["fixed_code"])
        assert rec["buggy_code"] != rec["fixed_code"], "buggy == fixed"
        assert rec["failing_test"] == "", "failing_test must stay empty"
        ops = [m["operator"] for m in rec["mutations"]]
        assert all(op in OPERATORS for op in ops), ops
        expected = ops[0] if len(ops) == 1 else f"composite:{'+'.join(ops)}"
        assert rec["bug_class"] == expected, rec["bug_class"]
        assert rec["fixed_code"] in source_set, "fixed_code is not a generated source"
        assert rec["buggy_code"] not in seen_buggy, "duplicate buggy_code in file"
        assert rec["buggy_code"] not in pilot_buggy, "buggy_code repeats a pilot record"
        seen_buggy.add(rec["buggy_code"])
        counts[rec["bug_class"]] += 1

    singles = {op: counts[op] for op in OPERATORS}
    assert all(v == 1_750 for v in singles.values()), singles
    n_comp = sum(v for k, v in counts.items() if k.startswith("composite:"))
    assert n_comp == 6_000, n_comp

    # Full determinism: regenerate every record from provenance.
    regen_mismatch = 0
    for rec, prov in zip(records, provs):
        again = generate_record(
            sources[prov["source_index"]], list(prov["operators"]), prov["seed"],
            filename=f"scaled_source_{prov['source_index']}.py",
        )
        if again != rec:
            regen_mismatch += 1
    assert regen_mismatch == 0, f"{regen_mismatch} regeneration mismatches"

    est = sorted((len(gen.to_text_example(r)) + 3) // 4 for r in records)
    n = len(est)
    survey = {"median": est[n // 2], "p90": est[int(0.9 * (n - 1))], "max": est[-1]}
    assert survey["max"] < 1024, survey

    pair_counts = {k: v for k, v in sorted(counts.items()) if k.startswith("composite:")}
    report = f"""# Scaled training dataset — validation report

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
- `train_records_20k.jsonl`: {len(records)} records, {data_path.stat().st_size} bytes
- `train_records_20k_provenance.jsonl`: {len(provs)} entries
  (source_index, operators, seed per record)
- Pilot file untouched: `pilot/pilot_records.jsonl` (225 records)

## Checks — ALL PASSED
1. Record count 20,000; schema keys exactly
   {{buggy_code, fixed_code, mutations, failing_test, bug_class}}.
2. Every buggy_code and fixed_code parses (ast.parse).
3. buggy_code != fixed_code on all 20,000 records.
4. failing_test == "" on all records (no test invented).
5. bug_class matches mutations on all records.
6. Singles: exactly 1,750 per operator:
   {json.dumps(singles, sort_keys=True)}.
7. Composites: 6,000 (30.0% of total). Pair counts (uneven — pairs
   differ in applicability, not a requirement):
   {json.dumps(pair_counts, sort_keys=True)}.
8. buggy_code distinct across the file and distinct from all 225
   pilot buggy_codes (no padding with repeats).
9. Every fixed_code is exactly one of the {len(sources)} deterministically
   rebuilt sources.
10. FULL determinism: all 20,000 records regenerated from provenance
    through the unchanged mutation pipeline match the file exactly
    (0 mismatches).

## Token-length survey (ESTIMATE — read the caveat)
Method: chars/4 on the training-text format
(`Fix this Python bug (bug class: ...)` + buggy + `Fixed code:` +
fixed), because no Gemma tokenizer exists in this VM.
- Scaled set estimate: median {survey['median']}, p90 {survey['p90']}, max {survey['max']}.
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
"""
    (args.data_dir / "train_records_20k_REPORT.md").write_text(report, encoding="utf-8")
    print(f"records: {len(records)} (14,000 single, 6,000 composite)")
    print(f"singles per operator: {json.dumps(singles, sort_keys=True)}")
    print(f"sources rebuilt: {len(sources)}")
    print(f"determinism: 20,000/20,000 regenerated exactly (0 mismatches)")
    print(f"token survey (chars/4 estimate): {json.dumps(survey)}")
    print("validation: ALL PASSED; report: train_records_20k_REPORT.md")


if __name__ == "__main__":
    main()
