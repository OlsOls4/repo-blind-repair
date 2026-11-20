# Repo-Blind Repair

A synthetic, 99.5%-exactly regenerable corpus (19,893/20,000; see Honest limits) for contamination-controlled
Python bug repair — plus the harness and dev-set converter from the
paper *"Repo-Blind Repair: Fixing Python Bugs Without Ever Seeing a Real
Fix"* (McBride, 2026; see `paper/paper.md`).

> **Current status — 20 Nov 2026.** First nonzero Kaggle score: Day P, a faithful clone of the public Patchsmith E010/V6 recipe (31B, Apache 2.0), scored **Public Score 0.15** (SUCCEEDED, 20 Nov) — the only nonzero of 15 scored dailies; the other 14 are all 0.00. 0.15 is below the 0.24 leaderboard top: it joins the block, it does not lead. The corpus did not produce this score — Day P is an inference-only clone of a public recipe run on real dev tasks, with no training on this corpus. The local replica headline is corrected: attempt 7's 25/32 as-graded (17/32 gold-equivalent) is **retired** — the replica parser unescaped literal backslash-n in tool-call strings, which the scorer's parsers keep raw; replaying the same outputs through the corrected parser gives **11/32 as-graded, 9/32 gold-equivalent** (a fixed-output replay, not a fresh run and not a lower bound). Real-issue serve traces (20 Nov) confirmed the repetition attractor on a published issue and showed the agent never emitted a source edit at all — a policy choice, not parser loss. Full analysis: `docs/wave11-attribution-2026-11-18.md`.

**The guarantee, stated precisely:** the fine-tuning data contains no
real human bug fix, no teacher-model output, and no third-party code.
Every record is a deterministic AST mutation of a toy Python program
written for this task, so the fixed code is exactly the original code
by construction. The guarantee covers the fine-tuning data only — not
the base model's pre-training. A provenance sidecar
(`data/train_records_20k_provenance.jsonl`) records
`(source_index, operators, seed)` per record. The 31 Oct 2026 full
validation regenerated all 20,000 records byte-identically
(0 mismatches) with the pipeline as it then stood; a 15 Nov 2026
re-check with the current pipeline finds 107 mismatches (0.5%),
all involving the `wrong_comparator` operator — the 8 Nov D1 fix
corrected the comparator inverter after the dataset was generated,
changing which candidate a seed selects. The affected records
remain valid (buggy ≠ fixed, both parse); only the byte-exact
claim is narrowed. See "Regenerate the corpus" below and
`VERIFICATION_LOG.md`.

## At a glance

- **Claim:** repair without recall.
- **Corpus:** 20,000 records (19,893/20,000 exactly regenerable with the current pipeline).
- **Harness (unit tests, not repair):** 33/33, 10/10.
- **Headline (scorer):** 0/18 pilots; kill test 7 Nov — LoRA 0/32, route KILLED; scorer dailies: 14 × 0.00, then Day P **0.15** (first nonzero, 20 Nov).
- **Repair (simulator, upper bound):** skill-script 29/32 kill, 103/116 overall.
- **Replica diagnostic (19 Nov, not a scorer result):** attempt-7 headline 25/32 retired; corrected-parser replay 11/32 as-graded, 9/32 gold-equivalent.
- **Resource:** https://github.com/OlsOls4/repo-blind-repair; data https://www.kaggle.com/datasets/olsols4/repo-blind-repair-synthetic-corpus-20k

**Measurement kinds:** Search = exhaustive model-free enumeration, measured CPU. Simulator = perfect enumeration, no model in the loop, upper bound. Scorer = Kaggle Public Score, the only leaderboard number.

**How this differs from Toeda's scaffold study:** Toeda asks whether scaffolding helps (25.2% on real tasks with a minimal prompt, llama.cpp); this entry asks whether provenance — synthetic-only training — transfers. K1/K2 show the Toeda-minimal prompt itself scores 0.00 under the Kaggle scorer, so the differentiation is empirical, not just conceptual.

**Who should use this:** if you train repair models and want a contamination-free benchmark, regenerate the corpus and run the search baseline.

**What this resource is not:** it does not claim synthetic training
transfers to real bugs. The paper's headline finding is a diagnosed
failure, settled by a pre-registered kill test on 7 Nov 2026: the
end-of-sequence fix works — the retrained (v19) adapter now emits
end-of-sequence — but it still fixes 0/32 held-out tasks (4/32
parsable), so the LoRA route was killed and the failure is in the
training, not the tasks. The earlier v1 pilots (0/10 and 0/8) showed
the original failure mode, in which the model never emitted
end-of-sequence and looped corrupted output to the token cap. Use
this corpus to measure that gap, not to assume it away. A model-free
inverse-operator search on the same 116 held-out tasks fixes 77/116
(71/116 by the matched inverse) — see `search_baseline/`; the tasks
are not hard, the trained model is.

## Contents

| Path | What it is |
|---|---|
| `data/` | Provenance sidecar (20,000 entries) + validation report. The 20,000-record file itself lives in the companion Kaggle dataset (see below) — this repo stays small and regenerates it. |
| `mutation/` | The 8-operator deterministic AST mutation pipeline (stdlib only) + licence checker + demo. |
| `harness/` | Offline repair harness: scope ladder, context packer, test runner, regression gate (stdlib only). `33/33` tests pass. |
| `data_generation/` | Corpus generator + validator + held-out-set builder. Regenerates the 20,000 records from the sidecar. |
| `search_baseline/` | Model-free inverse-operator search baseline on the 116 held-out tasks (CPU only): script + results (77/116 any-inverse, 71/116 matched, 77 gate-accepted) + method note. Re-run with `python3 reproduce_search.py` (~12 min CPU). |
| `converter/` | Dev-set converter: derives validated failing tests + regressions from the competition's public dev set (needs that data; see its README). |
| `training/` | Manual-LoRA training (v19 prep) + held-out eval scripts. Kaggle GPU + Gemma weights required; see its README. |
| `paper/` | The submitted paper text. |
| `search_guided/` | Search-guided simulator arms + executable skill script (simulator upper bounds, not scorer results). |
| `reproduce.py` | CPU reproduce: headline counts + 1,000-record sample regeneration + both unit suites (expects the dataset in a sibling `dataset/` directory). |
| `reproduce_search.py` | CPU re-run of the 116-task search baseline (~12 min). |
| `VERIFICATION_LOG.md` | Clean-clone verification record (15 Nov 2026) + the regeneration-gap decision. |
| `CHECKSUMS.md` | SHA-256 checksums for the dataset files + the Day P archive, and the release state (20 Nov 2026). |
| `examples/load_and_validate.py` | Short reproducible load/validation example (stdlib only, CPU): loads the sibling `dataset/`, checks counts, schema and provenance. |
| `docs/release-2026-11-20.md` | Release notes for dataset v3 + repo state at 20 Nov 2026. |
| `PATCHES.md` | The mechanical code changes vs the originals (repo-relative paths, `--data-dir`). |
| `LICENSE` / `DATA_LICENSE` | Code: Apache-2.0. Corpus data: CC BY 4.0. |

## Quickstart (CPU only, stdlib only, no network)

```bash
python3 -m unittest discover -s mutation/tests -v   # 10/10
python3 -m unittest discover -s harness/tests -v    # 33/33
python3 mutation/demo.py
python3 harness/demo.py
python3 reproduce.py   # expect REPRODUCE FAILED on the sample check: 998/1,000 (see VERIFICATION_LOG.md); needs the dataset in a sibling dataset/ directory
```

## Get the records without regenerating

Companion Kaggle dataset (CC BY 4.0):
`https://www.kaggle.com/datasets/olsols4/repo-blind-repair-synthetic-corpus-20k`
— `train_records_20k.jsonl` (20,000 records),
`train_records_20k_provenance.jsonl`,
`dev_tasks_heldout.jsonl` (116 tasks, **evaluation-only: never train on it**),
`pilot_records.jsonl` (225-record sample).

## Regenerate the corpus (one command, CPU only)

```bash
python3 data_generation/generate_scaled_dataset.py --out /tmp/regen
python3 data_generation/validate_scaled_dataset.py --data-dir /tmp/regen
```

Expected: 20,000 records (14,000 singles at exactly 1,750 per operator,
6,000 composites). NOTE (15 Nov 2026): the full regeneration no longer
matches byte-exactly — 107/20,000 records differ, all involving
`wrong_comparator` (34 singles, 73 composites), because the 8 Nov D1
fix corrected the comparator inverter after the dataset was generated.
`reproduce.py` reports this (sample 998/1,000). Counts, schema, and
parse checks still pass. See `VERIFICATION_LOG.md` for the full
verification record and the decision (disclose as-is; regeneration
deferred). Full run takes a
few minutes on CPU. See `data_generation/README.md`. A lighter check is
`reproduce.py` in this repo (sample regeneration +
headline counts + both test suites) — run it from the repo root; it
expects the dataset in a sibling `dataset/` directory (the staging
layout, or a Kaggle dataset download placed alongside the repo).

## Versions (naming — do not confuse them)

- **Corpus v18** — the 20,000-record dataset ("Dataset v18" in the
  paper): templates + mutation pipeline, frozen.
- **Kaggle dataset v3** — the versioned upload of that corpus at the
  dataset URL above (current; the v3 update is the honest 19,893/20,000
  regeneration disclosure — see `CHECKSUMS.md`).
- **Adapter v1** — the 1 Nov training run (checkpoint-opt-10000,
  10,000 steps); 0/18 held-out, never emitted end-of-sequence.
- **Adapter v19** — the 7 Nov retrain (end-of-sequence appended,
  bug-class label dropped); 0/32 on the pre-registered kill test
  (4/32 parsable). The LoRA route was killed.
- **Test status (current, 18 Nov 2026 — not a version):** harness suite 33/33 passing; mutation suite 10/10 passing.

## How to cite

See `CITATION.cff`. Paper: McBride, O. (2026). *Repo-Blind Repair:
Fixing Python Bugs Without Ever Seeing a Real Fix.* Gemma 4 Developer
Agent Competition, Paper Track.

## Honest limits (read before using)

- **Regeneration gap (15 Nov 2026).** The staged pipeline
  regenerates 19,893/20,000 records exactly; 107 (0.5%) differ, all
  involving `wrong_comparator`, after the 8 Nov D1 comparator-inverter
  fix. The 31 Oct "0 mismatches" verification predates that fix.
  See `VERIFICATION_LOG.md`.
- **Net effect for a judge:** reproduce.py exits 1 on the sample check (998/1,000); counts, schema, parse, both unit suites, and reproduce_search.py (77/116) all still pass — the gap is confined to regeneration exactness, not validity.
- **Replica diagnostic (19 Nov 2026, not a scorer result).** Local replica, real 12B, frozen 32 synthetic tasks. Attempt 7's headline — 25/32 as-graded, 17/32 gold-equivalent — is **retired**: the replica parser unescaped literal backslash-n in tool-call strings, which the scorer's parsers keep raw. Replaying the same outputs through the corrected parser gives **11/32 as-graded, 9/32 gold-equivalent** (a fixed-output replay, not a fresh run and not a lower bound). All 8 guard tasks (4 dropped_condition, 4 swapped_boolean) pass only by deleting/weakening the guard the hidden tests would check. Replica ≠ scorer: transformers, not vLLM; synthetic single-file tasks, not real repos. Dominant failure: repetition attractor — 5/7 failures are strict identical-call loops (runs of 14–28) until the 5-min budget died; a 6th (isbn) repeats only inside one 16,384-token generation and a 7th is a distinct fixation mode. Gold-equivalence re-graded by script (`docs/gold-equivalence/grade_gold_equivalence.py`, exit 0). Full analysis: `docs/wave11-attribution-2026-11-18.md`.
- **Synthetic only.** Single-operator mutations are shallow: no missing
  logic, wrong API use, or multi-file faults. Composites narrow the gap;
  they do not close it.
- **Held-out is synthetic transfer only.** The 116-task held-out set
  measures repair of deterministic mutations of unseen toy programs. It
  is not a real-bug fix rate and must never be presented as one.
- **The trained adapter is NOT in this repo.** The v1 LoRA adapter
  (62.6 MiB) is a failed artefact (0/18 held-out) and is excluded by
  design. The v19 retrain (7 Nov 2026, end-of-sequence fix applied)
  also failed the pre-registered kill test (0/32 fixed, 4/32
  parsable), so no adapter ships as a release asset either — a future
  adapter would ship with its evaluation numbers only if it passes
  the kill test.
- **Scorer serving unproven.** The inference-only submission package
  compiled and ran at the scorer (E4B daily, 5 Nov 2026, status
  SUCCEEDED, Public Score 0.00 — it fixed nothing, as expected for
  inference-only). vLLM serving of a trained adapter remains untested.
  Scorer-daily tally, current as of 20 Nov 2026: fifteen scored dailies
  (the 5 Nov E4B daily + Days A–J + K1 + K2 + Day S + Day P). Fourteen
  read Public Score 0.00; Day P reads 0.15 (first nonzero).
  [Day J outcome, checked Mon 16 Nov ~09:27: E4B skill-script daily
  submitted 15 Nov 09:19 GMT, status SUCCEEDED, Public Score 0.00 —
  the eleventh consecutive 0.00. Per the pre-set rule, Pick 2 locks to
  E4B v2, not the E4B skill-script.]
  [K1 outcome, checked Tue 17 Nov ~09:30: Toeda-exact minimal 12B
  daily submitted 16 Nov 09:20 GMT, status SUCCEEDED, Public Score
  0.00 — the twelfth consecutive 0.00. The loop hypothesis is not
  confirmed; next daily is K2 pure Toeda min.]
  [K2 outcome, checked Wed 18 Nov ~09:30: pure Toeda minimal 12B
  daily submitted 17 Nov 11:00 GMT, status SUCCEEDED, Public Score
  0.00 — the thirteenth consecutive 0.00. K1 = 0 and K2 = 0 triggers
  the ladder stop rule: prompt variants stop. Next daily is the fixed
  skill arm.]
  [Day S outcome, checked Thu 19 Nov ~10:00: fixed skill-script daily
  submitted 18 Nov ~11:00 GMT, status SUCCEEDED, Public Score 0.00 —
  the fourteenth consecutive 0.00. The skill mechanism is NOT the
  breaker: this package declares the skills: field the earlier
  skill-script dailies lacked, and still fixes nothing publicly.]
  [Day P outcome, checked Fri 20 Nov 18:00–18:01 GMT: 31B Patchsmith
  clone daily submitted 19 Nov 14:49:15 GMT, status SUCCEEDED, Public
  Score 0.15 — the first nonzero, and the only nonzero of 15. The
  package is a faithful clone of the public Patchsmith E010/V6 recipe
  (Apache 2.0), byte-identical to the published archive (archive
  SHA-256 e07535bdf469e399c80f0c7d846b55fcd763186760f0ac5072aa4ee61a8277c5;
  6 tools, temp 0.2, 28 calls). 0.15 is below the 0.24 leaderboard top:
  it joins the block, it does not lead. The corpus did not produce this
  score — no training on it; Day P is the Pick 1 candidate ahead of the
  25 Nov selection.]
- **Three kinds of results — do not mix them.** (1) *Synthetic*
  (simulator / replica): model-free search and perfect-enumeration
  upper bounds on toy tasks, plus the local replica diagnostic —
  attempt 7's 25/32 headline is retired; the corrected-parser replay
  is 11/32 as-graded, 9/32 gold-equivalent. None of these is a
  leaderboard number. (2) *Kaggle scorer*: the only leaderboard
  numbers — fourteen 0.00s and one 0.15 (Day P). (3) *Real-issue
  serve traces*: the exact K2 package run against a published FastAPI
  issue under the scorer's serving stack (a proxy verifier, not the
  scorer) — the repetition attractor is confirmed on a real issue,
  and the agent never emitted a source edit in any of four runs. No
  scored package was trained on this corpus: the 12B dailies are
  inference-only prompts and Day P is an inference-only clone of a
  public recipe. Nothing here shows synthetic training transferring
  to real bugs.
- **Dev-set converter output is not publishable as-is.** The derived
  dev-set tasks embed competition data by construction (full repository
  trees + problem statements). The converter script is published; its
  output must not be uploaded publicly without stripping. See
  `converter/README.md`.

## Licence

Code: Apache License 2.0 (`LICENSE`). Corpus data: CC BY 4.0
(`DATA_LICENSE`). Copyright 2026 Olly McBride.
