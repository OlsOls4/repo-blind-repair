# Dev-set converter — derive validated tasks from the public dev set

`build_dev_tasks.py` converts the competition's public dev set
(129 real tasks) into evaluation tasks with one validated failing
test plus up to 15 regression tests each, by running the real test
suite against the buggy and gold-patched code. The gold patch is
never written to the output. 30/30 converter tests pass:
`python3 -m pytest test_build_dev_tasks.py` (pytest required).

## What you need (not in this repo)

The public dev set itself: `tasks.jsonl` + `snapshots/` (129 task
archives, 22.42 GB), from the Gemma 4 Developer Agent Competition
Data tab on Kaggle (login-gated). Run:

```bash
python3 build_dev_tasks.py --inspect /kaggle/input/<data-dir>
python3 build_dev_tasks.py --convert ...   # see --help
```

## What it produced (for the paper)

3 of the first 20 public dev tasks converted (fastapi_14962,
fastapi_15280, fastapi_14978; 15 regressions each). That is a
derivation count, not a fix rate — no model has run on these tasks.
Converter work stopped at the pre-agreed hard stop (3/20 < 10).

## DO NOT publish the converter's output as-is

The derived output (`dev_subset20_r5.jsonl`) embeds competition data
by construction: full repository trees and full problem statements /
hints for the converted tasks. A 5 Nov 2026 audit found it NOT CLEAN
for public release. The converter script is published here; its
output must NOT be uploaded to a public repo or dataset without
stripping the embedded competition data first (mechanism under
investigation). The paper's preservation claim for that output may
need rewording — see the paper's Data availability section.
