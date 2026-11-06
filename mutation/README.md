# Repo-Blind Repair — mutation pipeline

Local, deterministic, zero-cost. No Kaggle, no training, no paid
anything, no teacher models, no real human bug-fixes as training data.
Bugs are produced only by deterministic AST mutation of clean code,
so the fixed code is exactly the original code by construction.

## Files

- `operators.py` — 8 deterministic AST operators: off_by_one,
  swapped_operands, wrong_comparator, dropped_condition,
  swapped_boolean, wrong_constant, removed_return,
  removed_assignment. Each mutation is reproducible from
  (operator, seed) and records file, line, operator, and
  original vs mutated snippet. Inapplicable operators raise
  `MutationNotApplicable` instead of producing broken code.
- `generator.py` — synthetic-bug generator. Emits
  `{buggy_code, fixed_code, mutations, failing_test, bug_class}`
  for 1 mutation (single) or 2 mutations (composite).
  `failing_test` is left empty unless a real test is actually
  derived — no test is ever invented here.
- `licence_checker.py` — permissive-only audit (MIT, Apache-2.0,
  BSD-2/3-Clause). Copyleft / non-commercial / private / unknown
  licences are rejected; a licence only counts as verified if its
  file is present locally. Remote URLs are UNVERIFIED by design.
- `LICENCES.md` — audit-list template + demo entries.
- `demo.py` + `demo_repo/toy.py` — demo (see below).

## Run

From this directory (requires only Python 3, stdlib only):

```bash
python3 demo.py
python3 -m unittest discover -s tests -v
python3 licence_checker.py --demo   # exit 1 is EXPECTED: demo entries
                                    # are named-only and UNVERIFIED
```

## Demo verification

`demo.py` prints 5 example training records (4 single-operator,
1 composite) and asserts for each: the mutated file still parses
(`ast.parse`), the mutated code differs from the original, the
record's `fixed_code` restores the original exactly, and re-running
the same (operator, seed) reproduces the identical record.

## Licence demo — honest status

Demo entries (requests, flask, click) are **named only, not
downloaded and not verified**. Their published licences are listed
in `LICENCES.md`, but the checker marks them UNVERIFIED because no
licence file is present locally. Verify each repo's own LICENCE
file before any training use; if a later download attempt fails,
record the failure in `LICENCES.md` verbatim instead of faking it.

## Known limits

- Mutation realism: single-operator mutations (off-by-one, swapped
  operands, etc.) do not cover real human bugs such as missing
  logic, wrong API use, or multi-file faults. Composite (2-operator)
  mutations narrow the gap but do not close it — this is the
  pipeline's main limitation, recorded for the training phase.
- `ast.unparse` reformats source, so mutated files differ from the
  original in formatting as well as in the mutated node; the
  recorded snippet (not a diff) is the exact change to trust.
