#!/usr/bin/env python3
"""Load and validate the Repo-Blind Repair corpus (CPU only, stdlib only).

Short reproducible check that the dataset next to this repo is intact:
counts, record schema, JSON parse, buggy != fixed, and provenance
coverage. It does NOT regenerate the corpus (see reproduce.py for the
sample-regeneration check) and it does NOT train or score anything.

Usage, from the repo root, with the dataset as a sibling directory:

    python3 examples/load_and_validate.py            # uses ../dataset
    python3 examples/load_and_validate.py --data-dir /path/to/dataset

Expected output ends with: LOAD AND VALIDATE: PASS
"""

import argparse
import json
import sys
from pathlib import Path

EXPECTED = {
    "train_records_20k.jsonl": 20_000,
    "train_records_20k_provenance.jsonl": 20_000,
    "dev_tasks_heldout.jsonl": 116,
    "pilot_records.jsonl": 225,
}

TRAIN_KEYS = {"buggy_code", "fixed_code", "mutations", "failing_test", "bug_class"}
PROV_KEYS = {"source_index", "operators", "seed"}


def load_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="dataset directory (default: sibling 'dataset/' next to the repo)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent.parent / "dataset"
    errors = []

    for name, expected_count in EXPECTED.items():
        path = data_dir / name
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        try:
            records = load_lines(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: unreadable ({exc})")
            continue
        if len(records) != expected_count:
            errors.append(f"{name}: {len(records)} records, expected {expected_count}")
        print(f"{name}: {len(records)} records")
        if name == "train_records_20k.jsonl":
            for i, rec in enumerate(records):
                if not TRAIN_KEYS <= rec.keys():
                    errors.append(f"train record {i}: missing keys {TRAIN_KEYS - rec.keys()}")
                    break
                if rec["buggy_code"] == rec["fixed_code"]:
                    errors.append(f"train record {i}: buggy == fixed")
                    break
        if name == "train_records_20k_provenance.jsonl":
            for i, rec in enumerate(records):
                if not PROV_KEYS <= rec.keys():
                    errors.append(f"provenance record {i}: missing keys {PROV_KEYS - rec.keys()}")
                    break

    print("Honest limits: full regeneration with the current pipeline matches "
          "19,893/20,000 records (99.5%); the 107 mismatches are all "
          "wrong_comparator records, which remain valid bugs. See "
          "VERIFICATION_LOG.md. Licence: corpus CC BY 4.0; code Apache-2.0.")

    if errors:
        print("LOAD AND VALIDATE: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("LOAD AND VALIDATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
