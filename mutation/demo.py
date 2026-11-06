"""Demo: mutate a toy file, print 5 example training records, verify.

Checks for every record:
  1. The mutated file still parses (ast.parse).
  2. The mutated code differs from the original.
  3. The record's fixed_code restores the original exactly.
  4. Re-running the same (operator, seed) reproduces the same record
     (determinism).

No test is invented: failing_test is empty in every record, because
no test was derived or run for these mutations.

Run from this directory:
    python demo.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generator import generate_record  # noqa: E402

EXAMPLES = [
    (["off_by_one"], 0),
    (["wrong_comparator"], 1),
    (["swapped_boolean"], 0),
    (["removed_return"], 0),
    (["off_by_one", "wrong_constant"], 2),  # composite example
]


def main() -> int:
    toy_path = Path(__file__).resolve().parent / "demo_repo" / "toy.py"
    source = toy_path.read_text()
    ast.parse(source)
    records = []
    for operators, seed in EXAMPLES:
        record = generate_record(source, operators, seed, filename="toy.py")
        # Verify: parses, differs, restores, deterministic.
        ast.parse(record["buggy_code"])
        assert record["buggy_code"] != record["fixed_code"], operators
        assert record["fixed_code"] == source, operators
        repeat = generate_record(source, operators, seed, filename="toy.py")
        assert repeat == record, f"non-deterministic: {operators}"
        assert record["failing_test"] == "", "no test may be invented"
        records.append(record)
        print(f"--- bug_class={record['bug_class']} ---")
        for m in record["mutations"]:
            print(
                f"  {m['operator']} (seed {m['seed']}, line {m['line']}): "
                f"{m['original_code']!r} -> {m['mutated_code']!r}"
            )
    print(f"\nOK: {len(records)} records, all parse, differ, restore, and "
          "reproduce deterministically.")
    print("\nFull records (buggy_code truncated to 200 chars):")
    for record in records:
        slim = dict(record)
        slim["buggy_code"] = record["buggy_code"][:200] + "..."
        slim["fixed_code"] = "<original toy.py, restored exactly>"
        print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
