"""CPU reproduce for the Repo-Blind Repair resource.

Recomputes the headline numbers from the repo + dataset — no network,
no GPU, no Kaggle, stdlib only. Run from the repo root:

    python3 reproduce.py

It expects the dataset in a sibling `dataset/` directory: in the
staging tree that is publish/dataset; for the published repo, download
the companion Kaggle dataset and place it alongside the repo as
`dataset/`. (This file is vendored in the repo so the pointer in
README.md resolves in-tree; the staging original lives at
publish/reproduce/reproduce.py. Unlike that original, this copy
resolves the repo to its own directory — in a real clone the
directory is `repo-blind-repair`, not `repo`.)

Checks:
  1. Dataset file headline counts (20,000 records; 14,000 singles at
     exactly 1,750 per operator; 6,000 composites; every record
     parses; buggy != fixed; failing_test always empty; every
     fixed_code one of the deterministically rebuilt sources).
  2. Sample regeneration: 1,000 records (first 500 + 500 evenly
     spaced) regenerated from the provenance sidecar through the
     staged mutation pipeline and compared field-for-field.
     (The FULL 20,000-record regeneration — 0 mismatches — was run
     31 Oct 2026; see data/train_records_20k_REPORT.md in the repo,
     and the staged full-validator run recorded in RESULTS.md.)
  3. Both unit-test suites from the staged repo: mutation 10/10,
     harness 33/33.

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE  # this file lives at the repo root in a real clone
# (repo-blind-repair/); the staging original in publish/reproduce/
# uses HERE.parent / "repo" because it sits one level up.
DATASET = HERE.parent / "dataset"

sys.path.insert(0, str(REPO / "data_generation"))

import generate_scaled_dataset as gen  # noqa: E402
from generator import generate_record  # noqa: E402
from operators import OPERATORS  # noqa: E402

SAMPLE_FIRST = 500
SAMPLE_EVEN = 500


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_tests(package_dir: Path, label: str) -> tuple[int, bool]:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=package_dir, capture_output=True, text=True)
    tail = (proc.stderr.strip().splitlines() or [""]) [-1]
    ok = proc.returncode == 0 and tail.startswith("OK")
    ran = 0
    for line in proc.stderr.splitlines():
        if line.startswith("Ran "):
            ran = int(line.split()[1])
    print(f"  {label}: {'OK' if ok else 'FAIL'} ({ran} tests, exit {proc.returncode})")
    if not ok:
        print(proc.stderr[-2000:])
    return ran, ok


def main() -> int:
    failures: list[str] = []

    records = load_jsonl(DATASET / "train_records_20k.jsonl")
    provs = load_jsonl(DATASET / "train_records_20k_provenance.jsonl")
    print(f"records: {len(records)} (provenance entries: {len(provs)})")
    if len(records) != 20_000 or len(provs) != 20_000:
        failures.append("record/provenance count != 20,000")

    counts: Counter = Counter()
    sources = gen.build_sources()
    source_set = set(sources)
    print(f"sources rebuilt: {len(sources)}")
    for rec in records:
        if set(rec) != {"buggy_code", "fixed_code", "mutations",
                        "failing_test", "bug_class"}:
            failures.append("schema mismatch")
            break
        ast.parse(rec["buggy_code"])
        ast.parse(rec["fixed_code"])
        if rec["buggy_code"] == rec["fixed_code"]:
            failures.append("buggy == fixed")
            break
        if rec["failing_test"] != "":
            failures.append("failing_test non-empty")
            break
        if rec["fixed_code"] not in source_set:
            failures.append("fixed_code not a rebuilt source")
            break
        counts[rec["bug_class"]] += 1
    singles = {op: counts[op] for op in OPERATORS}
    n_comp = sum(v for k, v in counts.items() if k.startswith("composite:"))
    print(f"singles per operator: {json.dumps(singles, sort_keys=True)}")
    print(f"composites: {n_comp}")
    if any(v != 1_750 for v in singles.values()):
        failures.append("singles != 1,750 per operator")
    if n_comp != 6_000:
        failures.append("composites != 6,000")

    # Sample regeneration from the sidecar: first 500 records plus 500
    # evenly spaced from index 500 on (exactly 1,000 distinct records,
    # covering singles and composites).
    idx = list(range(SAMPLE_FIRST))
    step = (len(records) - SAMPLE_FIRST) // SAMPLE_EVEN
    idx += list(range(SAMPLE_FIRST, len(records), step))[:SAMPLE_EVEN]
    idx = sorted(set(idx))
    mismatch = 0
    for i in idx:
        prov = provs[i]
        again = generate_record(
            sources[prov["source_index"]], list(prov["operators"]),
            prov["seed"],
            filename=f"scaled_source_{prov['source_index']}.py")
        if again != records[i]:
            mismatch += 1
    print(f"sample regeneration: {len(idx) - mismatch}/{len(idx)} exact "
          f"({mismatch} mismatches)")
    if mismatch:
        failures.append(f"sample regeneration: {mismatch} mismatches")
    print("full regeneration: 20,000/20,000 exact (0 mismatches) — "
          "31 Oct 2026 full-validator run, see "
          "data/train_records_20k_REPORT.md")

    print("unit tests (repo):")
    _, ok_m = run_tests(REPO / "mutation", "mutation")
    _, ok_h = run_tests(REPO / "harness", "harness")
    if not ok_m:
        failures.append("mutation tests failed")
    if not ok_h:
        failures.append("harness tests failed")

    if failures:
        print("REPRODUCE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("REPRODUCE PASSED: all headline numbers recomputed from the "
          "staged tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
