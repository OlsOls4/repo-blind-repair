# Repo-Blind Repair

A synthetic, byte-exactly regenerable corpus for contamination-controlled
Python bug repair — plus the harness and dev-set converter from the
paper *"Repo-Blind Repair: Fixing Python Bugs Without Ever Seeing a Real
Fix"* (McBride, 2026; see `paper/paper.md`).

**The guarantee, stated precisely:** the fine-tuning data contains no
real human bug fix, no teacher-model output, and no third-party code.
Every record is a deterministic AST mutation of a toy Python program
written for this task, so the fixed code is exactly the original code
by construction. The guarantee covers the fine-tuning data only — not
the base model's pre-training. A provenance sidecar
(`data/train_records_20k_provenance.jsonl`) records
`(source_index, operators, seed)` per record, so anyone can regenerate
all 20,000 records byte-identically (0 mismatches, verified 31 Oct 2026;
see `data/train_records_20k_REPORT.md`).

**What this resource is not:** it does not claim synthetic training
transfers to real bugs. The paper's headline finding is a diagnosed
failure: the trained model never emits end-of-sequence and loops
corrupted output to the token cap (held-out pilots 0/10 and 0/8). Use
this corpus to measure that gap, not to assume it away.

## Contents

| Path | What it is |
|---|---|
| `data/` | Provenance sidecar (20,000 entries) + validation report. The 20,000-record file itself lives in the companion Kaggle dataset (see below) — this repo stays small and regenerates it. |
| `mutation/` | The 8-operator deterministic AST mutation pipeline (stdlib only) + licence checker + demo. |
| `harness/` | Offline repair harness: scope ladder, context packer, test runner, regression gate (stdlib only). `32/32` tests pass. |
| `data_generation/` | Corpus generator + validator + held-out-set builder. Regenerates the 20,000 records from the sidecar. |
| `converter/` | Dev-set converter: derives validated failing tests + regressions from the competition's public dev set (needs that data; see its README). |
| `training/` | Manual-LoRA training (v19 prep) + held-out eval scripts. Kaggle GPU + Gemma weights required; see its README. |
| `paper/` | The submitted paper text. |
| `LICENSE` / `DATA_LICENSE` | Code: Apache-2.0. Corpus data: CC BY 4.0. |

## Quickstart (CPU only, stdlib only, no network)

```bash
python3 -m unittest discover -s mutation/tests -v   # 9/9
python3 -m unittest discover -s harness/tests -v    # 32/32
python3 mutation/demo.py
python3 harness/demo.py
```

## Regenerate the corpus (one command, CPU only)

```bash
python3 data_generation/generate_scaled_dataset.py --out /tmp/regen
python3 data_generation/validate_scaled_dataset.py --data-dir /tmp/regen
```

Expected: 20,000 records (14,000 singles at exactly 1,750 per operator,
6,000 composites), every record regenerating exactly from the sidecar
(0 mismatches). Full run takes a few minutes on CPU. See
`data_generation/README.md`. A lighter check is
`reproduce.py` in this repo (sample regeneration +
headline counts + both test suites) — run it from the repo root; it
expects the dataset in a sibling `dataset/` directory (the staging
layout, or a Kaggle dataset download placed alongside the repo).

## Get the records without regenerating

Companion Kaggle dataset (CC BY 4.0) — **STAGING PLACEHOLDER, not yet live**:
`https://www.kaggle.com/datasets/olsols4/repo-blind-repair-corpus`
— `train_records_20k.jsonl` (20,000 records),
`train_records_20k_provenance.jsonl`,
`dev_tasks_heldout.jsonl` (116 tasks, **evaluation-only: never train on it**),
`pilot_records.jsonl` (225-record sample).

## How to cite

See `CITATION.cff`. Paper: McBride, O. (2026). *Repo-Blind Repair:
Fixing Python Bugs Without Ever Seeing a Real Fix.* Gemma 4 Developer
Agent Competition, Paper Track.

## Honest limits (read before using)

- **Synthetic only.** Single-operator mutations are shallow: no missing
  logic, wrong API use, or multi-file faults. Composites narrow the gap;
  they do not close it.
- **Held-out is synthetic transfer only.** The 116-task held-out set
  measures repair of deterministic mutations of unseen toy programs. It
  is not a real-bug fix rate and must never be presented as one.
- **The trained adapter is NOT in this repo.** The v18 LoRA adapter
  (62.6 MiB) is a failed artefact (0/18 held-out) and is excluded by
  design. If a future adapter passes the pre-registered kill test, it
  will ship as a GitHub Release asset with its evaluation numbers.
- **Scorer serving unproven.** The inference-only submission package
  compiled and ran at the scorer (E4B daily, 5 Nov 2026, status
  SUCCEEDED, Public Score 0.00 — it fixed nothing, as expected for
  inference-only). vLLM serving of a trained adapter remains untested.
- **Dev-set converter output is not publishable as-is.** The derived
  dev-set tasks embed competition data by construction (full repository
  trees + problem statements). The converter script is published; its
  output must not be uploaded publicly without stripping. See
  `converter/README.md`.

## Licence

Code: Apache License 2.0 (`LICENSE`). Corpus data: CC BY 4.0
(`DATA_LICENSE`). Copyright 2026 Olly McBride.
