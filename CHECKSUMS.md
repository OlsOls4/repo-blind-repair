# Checksums — Repo-Blind Repair resource (20 Nov 2026)

SHA-256 checksums for the resource as staged on 20 Nov 2026. Verify a
download with `sha256sum <file>` and compare each hash with the tables
below. The dataset is version 3 on
Kaggle; the data files below are the files as staged for that version.
The corpus itself is unchanged by the 20 Nov scorer result — Day P
(0.15) is a property of a submitted agent package, not of this data.

## Kaggle dataset files (version 3)

https://www.kaggle.com/datasets/olsols4/repo-blind-repair-synthetic-corpus-20k
— licence CC BY 4.0.

| File | Bytes | SHA-256 |
|---|---|---|
| `train_records_20k.jsonl` | 24,055,389 | `1786411c211030261fca9b784f82e2507a9ec6a21cdf80e72c957741794c5ddd` |
| `train_records_20k_provenance.jsonl` | 1,465,747 | `3e848a122e859a96dd4bc8ccc45703d95cf4b3baea44a2362da4dfe4881eafe2` |
| `dev_tasks_heldout.jsonl` | 128,517 | `2699746acb8cbb764cbe43ce2f03389833b1c388100953d19cf9010afb379e05` |
| `pilot_records.jsonl` | 240,323 | `b26dd691881d3691cff02b54ee737f196fa3c0f56b114fb919fe357487c142f7` |
| `train_records_20k_REPORT.md` | 4,390 | `1ac33e0f434cd3057e0db4ba0a6b7e07b72efda55c8659849cbfb993c61c65a8` |

## Day P submission archive (the scored package — NOT part of this repo)

The 0.15 scorer result comes from a faithful clone of the public
Patchsmith E010/V6 recipe (Apache 2.0), byte-identical to the published
archive. The submission package is a competition entry, not part of
this resource, and is recorded here only so the result is checkable.

- Local package: `submission-patchsmith-clone.zip`
  (md5 `fb62f7732c052cd661f17de25baf2408`, 4 files, 5,451 bytes unpacked)
- Archive SHA-256: `e07535bdf469e399c80f0c7d846b55fcd763186760f0ac5072aa4ee61a8277c5`
- Model: `gemma-4-31b-it-qat-w4a16-ct`; 6 tools; temp 0.2; budgets
  60 s / 28 calls / 8 min / 80 turns
- Submitted 19 Nov 2026 14:49:15 GMT; SUCCEEDED; Public Score **0.15**
  (checked 20 Nov 2026 18:00–18:01 GMT) — first nonzero of 15 scored
  dailies; the other 14 are all 0.00. 0.15 is below the 0.24 leaderboard
  top: it joins the block, it does not lead.

## Repo state

- Staged tree: `publish/repo/` as prepared 20 Nov 2026 (73 files: 70 at
  the last push, head `61d62bd2`, 18 Nov 2026, + the 3 new files in
  this polish — this polish is staged locally and NOT yet pushed;
  see `docs/release-2026-11-20.md`).
- Paper: FINAL and frozen (submitted 5 Nov, re-submitted 6 + 11 Nov,
  pin `1bf60f4b`; re-scored 34/50). Nothing in this release changes it.
- Code licence: Apache-2.0 (`LICENSE`). Corpus licence: CC BY 4.0
  (`DATA_LICENSE`).

## How to verify

```bash
# from a download of the Kaggle dataset:
sha256sum train_records_20k.jsonl train_records_20k_provenance.jsonl \
  dev_tasks_heldout.jsonl pilot_records.jsonl train_records_20k_REPORT.md
# compare each hash with the table above

# from a clone of the repo, with the dataset as a sibling directory:
python3 examples/load_and_validate.py   # expect: LOAD AND VALIDATE: PASS
python3 reproduce.py                    # expect the documented 998/1,000
                                        # sample-regeneration result (see
                                        # VERIFICATION_LOG.md); counts,
                                        # schema, parse and both unit
                                        # suites still pass
```
