# Patches vs. the competition originals

This repo is staged from the competition working tree
(`~/workspace/gemma-repo-blind/`). Everything is copied verbatim
EXCEPT the two mechanical patches below and this file. No results,
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

## Verification

- `../reproduce/reproduce.py` (staging tree) runs a sample
  regeneration (1,000 records from the sidecar through the staged
  pipeline, compared field-for-field) plus both test suites.
- The staged validator was run in full against the staged dataset
  directory; see `../reproduce/RESULTS.md` for the recorded output.
- The 31 Oct 2026 full validation of the source files (0 mismatches)
  is preserved verbatim in `data/train_records_20k_REPORT.md`.
