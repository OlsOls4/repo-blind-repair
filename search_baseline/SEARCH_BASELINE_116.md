# Search baseline — inverse-operator repair on all 116 held-out tasks

> Published copy (8 Nov 2026): staged in the public repo under
> `search_baseline/`; paths are repo-relative. The original run
> (8 Nov 2026, CPU-only, 697.2 s) is recorded verbatim below;
> `reproduce_search.py` re-runs it and checks the headline
> figures against `search_baseline_116_results.json`.


**Status: measured, CPU-only, 8 Nov 2026. Not learned repair.**

- Script: `search_baseline/search_baseline.py` (`--task-set all116`; the frozen-32
  path is unchanged — a dry-run reproduces the original 1,292 candidates
  exactly). Re-run the whole 116-task check with `python3 reproduce_search.py`
  from the repo root (CPU-only, ~12 min)
- Results: `search_baseline/search_baseline_116_results.json` (per-task records +
  aggregates; same format as `search_baseline_results.json`, plus
  `total_fixed_matched_inverse`, pooled matched Wilson, and a
  singles/compounds split under `by_kind`)
- Set: all 116 tasks in file order from `dataset/dev_tasks_heldout.jsonl`
  (companion Kaggle dataset, placed alongside the repo as `dataset/`)
  (107 singles + 9 compounds). Compounds carry a single-operator
  `bug_class` label; the record carries no second-operator field, so
  "matched inverse" keeps the frozen-32 rule (inverse == `bug_class`).
- Candidates tried: 6,713 (~58/task) in 697.2 s (~11.6 min). Baseline
  sanity: the failing test failed on the unpatched buggy code for all
  116 tasks.
- Harness: the FIXED `harness/test_runner.py` was verified before the
  run (md5 `cd2aeb109ec2d275a5fa20abb944971b`; `__pycache__` deletion
  before every run + `PYTHONDONTWRITEBYTECODE=1`), and the script
  itself clears `__pycache__` before each test run. No stale-bytecode
  exposure: the frozen-32 subset inside this run reproduces the
  5 Nov result exactly (24/20/24, below).

## Method — search baseline, not learned repair

Identical to the frozen-32 baseline (method documented in full in the
docstring of `search_baseline.py`): for each task, every
single-operator inverse is tried exhaustively over every candidate
site in `task.py`, and a candidate counts as a fix only if the
task's failing test passes under the harness runner
(`harness/test_runner.py::run_one_test`). Regression behaviour is
reported separately with the same capped suite the gate uses.
The search is seed-blind. Inverses, the no-inverse operators, and
the labelled reconstruction guess are defined in the docstring of
`search_baseline.py` and unchanged here.

## Results

Headline: **failing test passes on 77/116 (66.4%), all 77
gate-accepted** (regressions green). Conservative "true repair"
figure — fixed by the **matched** inverse — is **71/116 (61.2%)**.
Wilson 95%: any-inverse **0.57–0.74**, matched-inverse **0.52–0.70**.

| Bug class | n | Fixed, any inverse | Fixed, matched inverse | Gate-accepted | Wilson 95% (any / matched) |
|---|---:|---:|---:|---:|---|
| off_by_one | 20 | 20 | 20 | 20 | 0.84–1.00 / 0.84–1.00 |
| swapped_operands | 20 | 20 | 20 | 20 | 0.84–1.00 / 0.84–1.00 |
| wrong_comparator | 6 | 6 | 6 | 6 | 0.61–1.00 / 0.61–1.00 |
| dropped_condition | 5 | 5 | 0 (no inverse exists) | 5 | 0.57–1.00 / 0.00–0.43 |
| swapped_boolean | 5 | 5 | 5 | 5 | 0.57–1.00 / 0.57–1.00 |
| wrong_constant | 20 | 20 | 20 | 20 | 0.84–1.00 / 0.84–1.00 |
| removed_return | 20 | 0 | 0 | 0 | 0.00–0.16 / 0.00–0.16 |
| removed_assignment | 20 | 1 | 0 | 1 | 0.01–0.24 / 0.00–0.16 |
| **Pooled (n=116)** | **116** | **77** | **71** | **77** | **0.57–0.74 / 0.52–0.70** |

| Kind | n | Fixed, any inverse | Fixed, matched inverse | Gate-accepted | Wilson 95% (any / matched) |
|---|---:|---:|---:|---:|---|
| Singles | 107 | 71 | 65 | 71 | 0.57–0.75 / 0.51–0.69 |
| Compounds | 9 | 6 | 6 | 6 | 0.35–0.88 / 0.35–0.88 |

- 196 fixing candidates in total across the 116 tasks; **0** of them
  failed a regression.
- The frozen-32 subset inside this run gives **24/32 any, 20/32
  matched, 24/32 gate-accepted** — identical to the 5 Nov frozen-32
  run, so the extension and the refactored script reproduce the
  original measurement exactly.
- Every invertible single-operator class is at its ceiling on the
  full set: off_by_one, swapped_operands, wrong_constant 20/20
  matched; wrong_comparator 6/6; swapped_boolean 5/5.
- `dropped_condition` 5/5 any-inverse, 0 matched — same mechanism
  as on the frozen 32: all five fixes flip a guard boolean default
  (`swapped_boolean`) so the broken guard never fires and the single
  failing probe passes. The dropped operand is never recovered.
- `removed_assignment` 1/20: the single fix is a `wrong_constant`
  probe-weakness pass (`heldout-isbn-removed_assignment-2`), not a
  repair.
- `removed_return` 0/20 by any inverse — but the labelled
  reconstruction guess (`return <last assigned local>`) fixes **4
  of the 20** (`heldout-days-removed_return-0`,
  `heldout-roman-removed_return-0`, `heldout-taxi-removed_return-0`,
  plus the compound `heldout-compound-removed_return-0`). These are
  NOT counted as inverse fixes (gate-accepted stays 0 for the
  class), and on the frozen 32 the guess had produced 1 candidate
  and fixed 0. On the full set it produces 8 candidates and fixes 4
  — all accumulator-style returns (`return total` / `return out`).
- Compounds 6/9 fixed, all by the labelled operator's inverse at
  site 0. A single inverse fixes the failing test even though the
  task carries two bugs — the failing probe evidently exercises only
  one of them (same single-probe weakness, extended to compounds).
  The 3 unfixed compounds include the 2 with zero fixing candidates
  of any kind.
- The any-vs-matched gap is **6 tasks**, all probe-weakness, not
  repairs: 5 `dropped_condition` (boolean-default flips) + 1
  `removed_assignment` (coincidental constant).
- **35 tasks have zero fixing candidates of any kind** (19
  `removed_assignment`, 16 `removed_return`) — the genuinely hard
  core on the full set, where repair must reconstruct missing code.

## Interpretation

The frozen-32 headline survives the extension almost exactly:
75% → 66% any-inverse, 62.5% → 61% matched. The tasks are still
not hard in the invertible classes — every one of them is at its
ceiling on the full set — and the LoRA's 0/32 is still a format
failure, not task hardness. What the full set adds is precision
about the hard core: it is 35 tasks (30%), concentrated entirely
in the two information-destroying operators, plus the 5
`dropped_condition` tasks that only a probe-weakness pass can
"fix". The conservative true-repair ceiling for any
single-edit, test-guided agent on this set is 71/116 (61%) by
matched inverse, 77/116 (66%) if probe-weakness passes are
counted; the reconstruction guess adds 4 more failing-test passes
on `removed_return` without changing the inverse figures.

## Implications for the search-guided agent prototype

(Ceiling numbers below are this measurement; prototype numbers are
subagent A's reported CPU-simulator results, 8 Nov 2026.)

- The exhaustive ceiling on singles is **71/107 (66%)** any-inverse,
  **65/107 (61%)** matched. The prototype's reported 60/107 (56%)
  captures ~85% of the exhaustive ceiling within the 60-call budget
  — the remaining 11 tasks are enumeration depth/budget, not missing
  inverses. On the kill set the same comparison is 22/32 (prototype)
  vs 24/32 (exhaustive).
- The prototype's enumeration should include the reconstruction
  guess as a first-class candidate type, not an afterthought: it
  fixes 4/20 `removed_return` tasks here (0/4 on the frozen 32,
  where it looked useless). Without it, `removed_return` is a
  guaranteed 0 for any inverse-only enumeration.
- `dropped_condition` and `removed_assignment` cannot be repaired
  by inverse enumeration at all (0 matched on 25 tasks combined);
  their only "fixes" are probe-weakness passes. A prototype that
  chases them spends calls on passes a judge would not count as
  repairs — better to skip them early and spend the budget where a
  matched inverse exists.
- Compounds are in scope for single-edit enumeration: 6/9 fix with
  one inverse, because the failing probe sees only one bug. The
  prototype does not need a two-edit mechanism to cover them.
- The gate is not the binding constraint: 196/196 fixing
  candidates passed regressions, and every any-inverse fix was
  gate-accepted. The binding constraints are enumeration coverage
  (finding the right site/param within 60 calls) and the 5-minute
  per-task wall clock at the scorer — both unproven at the scorer,
  and neither is measured by this baseline.
