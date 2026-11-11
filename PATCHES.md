# Patches vs. the competition originals

This repo is staged from the competition working tree
(`~/workspace/gemma-repo-blind/`). Everything is copied verbatim
EXCEPT the mechanical patches below and this file. No results,
numbers, or behaviour were changed.

## 1. Hardcoded local path → repo-relative (2 files)

- `data_generation/generate_scaled_dataset.py:48`
- `data_generation/validate_scaled_dataset.py:37`

Was:

```python
MUTATION_DIR = Path.home() / "workspace" / "gemma-repo-blind" / "mutation"
```

Now:

```python
MUTATION_DIR = HERE.parent / "mutation"  # repo-relative (patched for publication; was a hardcoded local path)
```

Why: the originals pointed at the author's local workspace. The
patched form resolves `mutation/` relative to the script, so the
repo works from any checkout. Behaviour is identical: same module,
same operators, same seeds.

## 2. Validator accepts `--data-dir` (1 file)

- `data_generation/validate_scaled_dataset.py`: `main()` now takes an
  optional `--data-dir` (default: the script's directory) and reads /
  writes the dataset files there. Previously it only read from its own
  directory. The checks themselves are unchanged.

Why: so the validator can run against a published dataset directory
(e.g. the Kaggle dataset download) without moving files.

## 3. Search baseline path-hygienised for the published layout (new files, 8 Nov 2026)

- `search_baseline/search_baseline.py` — adapted from
  `kaggle/search_baseline.py` (the script behind the 116-task
  headline: 77/116 any-inverse, 71/116 matched, 77 gate-accepted).
  Path changes only: it lives in `search_baseline/` instead of
  `kaggle/`, imports `mutation/`, `harness/`, and `data_generation/`
  from the repo root, and finds the held-out tasks in the sibling
  `dataset/` directory (`dataset/dev_tasks_heldout.jsonl`, with
  in-tree fallbacks) instead of `kaggle/dev_tasks_heldout.jsonl`.
  Search logic, candidate generation, test invocation, and
  aggregation are unchanged — verified by `reproduce_search.py`,
  which re-runs all 116 tasks from this tree and checks the headline
  figures against the recorded 8 Nov 2026 results.
- `search_baseline/SEARCH_BASELINE_116.md` — same adaptation of
  `kaggle/SEARCH_BASELINE_116.md` (path references re-pointed to the
  published layout; results text unchanged).
- `search_baseline/search_baseline_116_results.json` — byte copy of
  the 8 Nov 2026 run record, unedited.

## 4. `reproduce.py` resolves the repo to its own directory (8 Nov 2026)

- `reproduce.py`: `REPO` was `HERE.parent / "repo"`, which only works
  in the staging tree. In a real clone the directory is
  `repo-blind-repair`, so this copy uses `REPO = HERE` (the file
  lives at the repo root) and `DATASET = HERE.parent / "dataset"`.
  The staging original at `publish/reproduce/reproduce.py` keeps the
  old form, which is correct there. Behaviour is otherwise identical.

## Verification

- `../reproduce/reproduce.py` (staging tree) runs a sample
  regeneration (1,000 records from the sidecar through the staged
  pipeline, compared field-for-field) plus both test suites.
- The staged validator was run in full against the staged dataset
  directory; see `../reproduce/RESULTS.md` for the recorded output.
- The 31 Oct 2026 full validation of the source files (0 mismatches)
  is preserved verbatim in `data/train_records_20k_REPORT.md`.
