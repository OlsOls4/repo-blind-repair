<!-- Staged copy of paper/SUBMISSION.md body (Abstract through References),
     regenerated 2026-11-11 from the 2,798/2,846-word body (recount.py methods
     1/2; verified against SUBMISSION.md). Source of truth is paper/SUBMISSION.md
     — do not edit this copy directly. -->
## Abstract

**KILL RESULT (7 Nov): retrained LoRA (low-rank adaptation) fixes 0/32; end-of-sequence fix works but repairs nothing. LoRA route killed.**

**Offline coding agents** are usually trained on real human bug fixes, so repair is hard to tell from recall on hidden tasks.

**Repo-Blind Repair** removes that ambiguity for its fine-tuning data by construction: the model is trained only on synthetic bugs from deterministic AST mutation of toy Python programs written for this task — never a real human fix, a real repository, or the competition's hidden tasks. The guarantee covers the fine-tuning data only, not base pre-training. Every record's fixed code is exactly the original, so ground truth is known.

**The headline finding** is a diagnosed failure mechanism, not a score: training never taught end-of-sequence (EOS) after the fixed file, so at evaluation the model runs to the token cap looping corrupted "Fixed code:" fragments. Held-out pilots fix 0/10 (Wilson 95% 0.00–0.28) and 0/8 on stratified singles (0.00–0.32). The corpus, harness, and dev-set converter are offered as a public, contamination-controlled resource (see Data and code availability).

### At a glance

- **Claim:** repair without recall.
- **Corpus:** 20,000 records.
- **Harness:** 33/33, 10/10.
- **Headline:** 0/18 pilots; kill test 7 Nov — LoRA 0/32, route KILLED.
- **Repair (simulator):** skill-script 29/32 kill, 103/116 overall.
- **Resource:** https://github.com/OlsOls4/repo-blind-repair; data https://www.kaggle.com/datasets/olsols4/repo-blind-repair-synthetic-corpus-20k

## Introduction

Coding agents increasingly work where code cannot leave the machine — private codebases, regulated teams, repositories barred from cloud APIs. The standard recipe — fine-tuning on real issues and human-written patches — makes two claims hard to audit: generalisation, because a high score may reflect memorisation; and provenance, because an adopting team cannot easily verify that their code, a competitor's, or the benchmark was excluded.

Repo-Blind Repair makes both claims auditable; the corpus inventory in Methods is the audit trail. The training data contains no real bug fixes at all: clean code is deliberately broken by deterministic transformations, and the model learns to restore it. Because each bug is a known transformation of known clean code, every training example carries its own ground truth — fix, bug class, changed node. Nothing in fine-tuning depends on a human patch, a teacher model, or a private repository.

The scientific question is therefore quantitative: how far does synthetic training transfer, where does it fail, and which additions close the gap? That question is the contribution; the learned-vs-search comparison is measured (see Results), while a real-fix baseline and dev-set rate remain **TO BE MEASURED**.

### In plain English

We break clean toy programs in known ways and teach the model to fix them, so a good score cannot come from memorised fixes. The model sees only broken code and must restore the original.

One real training record (off_by_one, record 344) divides by 3, not 2.

## Methods

### Clean-room training data: deterministic mutation

Training records are produced locally (Python standard library only) by a deterministic AST mutation pipeline: a clean Python file, one or two operators, and a record with five fields: `buggy_code` (full mutated file), `fixed_code` (full original file), `mutations` (file, line, operator, original versus mutated snippet), `failing_test` (empty — none invented), and `bug_class` (the operator, or `composite:op1+op2`).

Eight operators, each reproducible from an (operator, seed) pair:

| Operator | Effect |
|---|---|
| off_by_one | integer ±1 |
| swapped_operands | swap operands |
| wrong_comparator | `<` to `<=` |
| dropped_condition | drop one conjunct |
| swapped_boolean | flip boolean |
| wrong_constant | shift by ≥2 |
| removed_return | drop return |
| removed_assignment | replace with `pass` | Candidates are collected in deterministic AST walk order; the target is `seed % len(candidates)`, and only that node changes; every mutated file still parses.

Singles teach localisation of one fault; composites hypothesise a narrower realism gap.

Corpus inventory (offline, from the validation report). Dataset v18: 82 parameterised shapes (62 general + 20 boolean-rich), rebuilt into 2,126 sources, 1,166 in the 20,000 records — 14,000 singles at 1,750 per operator, plus 6,000 composites across 14 pairs (25–709 per pair). No real human bug fix, no teacher-model output, and no third-party code was used. A provenance sidecar records (source_index, operators, seed) per record; regenerating all 20,000 records matches exactly (0 mismatches). Every record parses, buggy differs from fixed, and `failing_test` is empty. A token-trigram containment audit of the 116 held-out tasks against the 1,166 unique clean sources finds no near-duplicates: max 0.39, none above 0.50 (kaggle/audit_near_dup.py, sha256 6192d2f1). Above 0.50, a held-out task shares over half its token trigrams with a training source.

### Repair harness: scope ladder, context packer, regression gate

The repair agent runs inside an offline, standard-library-only harness with four components.

First, a trace extractor parses failing test output and issue text into failing tests, traceback frames, and implicated files. Second, a scope ladder controls where the agent looks:

| Rung | Looks at | Time-box |
|---|---|---|
| 1 | Files named by the failing trace | 30 s |
| 2 | Imports and callers (AST analysis) | 60 s |
| 3 | Wider repository search | 120 s |

Third, a context packer assembles rung-ordered file contents under a token budget (characters divided by four, cap 32,768), truncating with markers. Fourth, a test runner executes unittest identifiers with hard timeouts, and a regression gate accepts a patch only if the failing test passes and a capped suite (max 15 tests, 90 seconds) stays green.

Patch application is conservative: an edit applies only if its old text occurs exactly once, otherwise rejected atomically. The harness never touches the network, calls a model, or writes outside its repository. Patch generation is a marked placeholder; a missing model raises an error, never silent "no patch needed".

The intended model is a Gemma 4 12B LoRA (quantisation-aware training (QAT) checkpoint) trained on the records above.

## Results

| Check | Result |
|---|---|
| Unit tests — not repair scores | Harness 33/33; mutation 10/10 |
| Held-out pilots (1 Nov) | 0/10 compound-heavy (0.00–0.28); 0/8 stratified singles (0.00–0.32) |
| Kill pilot (7 Nov) | 0/32; parsable 4/32 (12.5%); 0 operators |
| Base-model baseline | 0/32; parsable 1/32 |
| Dev-set derivation | 3 of 20 tasks converted, 15 regressions each (not a fix rate); output deleted 5 Nov 2026 |
| Search baseline (model-free) | 24/32 frozen; 77/116 full |
| Search-guided v3 (simulator) | 28/32 kill; 87/107 singles; 95/116 overall |
| Skill-script arm (simulator) | 29/32 kill; 95/107 singles; 103/116 overall |
| Scorer dailies | 0.00 ×7 |

*Numbers: 0/18 = LoRA pilots (v1), 0/32 = kill pilot (v19), 24/32 = model-free search (not the model), 33/33 = harness unit tests (not repair).*

### Harness

The harness suite passes **33/33 tests** (one-line rung-1 demo patch; failing test passes, regressions green); the mutation suite passes **10/10 tests** (five records parse, differ, reproduce identically, plus a comparator-inverter regression test).

Training-path verification (31 Oct 2026): Gate 1 (adapter attaches to all 328 target modules) **PASSED**; the 20-step pilot **SUCCEEDED** in 65.19s. Timing only — not repair quality. Gate 2 (the live-vLLM stop-sequence probe) was **OVERALL INCONCLUSIVE**.

### Pilots

Held-out synthetic evaluation, 1 Nov 2026: 116 tasks from 18 new domains, audited overlap-free (see Methods), every derived test verified fail-on-buggy, pass-on-fixed. Pilot slices fix 0/10 on a compound-heavy file-order slice (5 gate_rejected — candidate produced but failing test still failed — and 5 no_patch — no candidate produced) and 0/8 on a stratified single-bug slice (7 no_patch, 1 gate_rejected) — failure is not confined to two-fault tasks. Intervals are per slice, not pooled: 0/10 gives Wilson 95% 0.00–0.28 and 0/8 gives 0.00–0.32; neither proves a true rate of zero. Only 17 distinct tasks were piloted (18 evaluations) before the GPU quota was exhausted; the kill test has since run 32 tasks (8–9 overlap the pilots, most likely 8).

Caveat: the first baseline run scored 14/32 — CPython loaded stale bytecode when same-size candidates were rewritten within the same second. The harness now clears __pycache__ before every run and sets PYTHONDONTWRITEBYTECODE=1; the 6 gate-rejected pilot results may include false negatives.

Adapter evaluated: all 1 Nov slices ran on checkpoint-opt-10000 (Version 1, 1 Nov 04:57 GMT; 10,000 steps; training loss 8.5253 to 0.0191, **TRAINING LOSS ONLY**) — not the packaged inference-only variant, and not v19.

Models identify buggy files from issue text alone at up to 76% on SWE-Bench Verified versus 53% off-benchmark [9], and score three times better on Verified than on fresh benchmarks [10].

### Search baseline

A model-free inverse-operator search on all 116 held-out tasks fixes 77/116 (66%, Wilson 95% 0.57–0.74) by any inverse and 71/116 (61%, 0.52–0.70) by the matched inverse; on the frozen 32, 24/32 (75%, 0.58–0.87), 20/32 matched (62.5%, 0.45–0.77). The any-vs-matched gap is exactly 6 probe-weakness passes (5 dropped_condition, 1 removed_assignment), not repairs. The hard core is 35 tasks with zero fixing candidates (19 removed_assignment, 16 removed_return); a reconstruction guess fixes 4 of 20 removed_return. Note the 81-vs-77 trap: fixing lists are non-empty for 81 tasks, but 4 are reconstruction guesses, excluded from the count. Search baseline, not learned repair: the tasks are not hard. The LoRA's 0/18 was a format failure — it never emitted end-of-sequence — but with end-of-sequence fixed, the retrained adapter still fixes 0/32. The kill test measures format learning: learned hard core 32/32, search hard core 35/116.

### What repairs instead: test-guided inverse enumeration

A model-free simulator shows what a test-guided agent could do with the same inverses: enumerate single-operator inverses only inside the function the failing test calls, test each with the fixed harness, and stop at the first gate-accepted fix. This is a SIMULATOR UPPER BOUND — perfect enumeration, no wasted calls — not a real agent run. Under the prompt's own inverse order and a 60-call budget (4 to orient, 3 per failed candidate), it fixes 22/32 kill tasks (69%, Wilson 0.51–0.82) and 60/107 singles (56%, 0.47–0.65). Folding each revert into the next edit allows 27 candidates in the same budget, lifting this to 24/32 (75%) and 69/107 (64%). Median 17 calls when fixed (max 59). A reconstruction phase for the 35 destructive tasks — reinserting the last assigned local before a removed return, or the last computed value before a removed assignment — fixes 18 more: 28/32 kill, 87/107 singles (95/116 overall). The same enumeration as an executable skill, run by one script call, fixes 29/32 kill and 95/107 singles (103/116 overall) in simulation, median 0.8 s script time per task; the unproven link is whether the scorer attaches the skill-toolset. With reconstruction, the exhaustive ceiling is 29/32 and 95/107 (103/116 overall; 71/107 by any single inverse, 65/107 matched). The gap to the LoRA's 0/32 is a search problem with a known ceiling, not task hardness.

### Kill test

The pre-registered kill test ran 7 Nov 2026 (v19: end-of-sequence appended, label dropped). The retrained adapter fixes 0/32 (Wilson 95% 0.00–0.11); 4/32 are parsable (12.5%, Wilson ~0.05–0.28), in 0 operators; the base model also fixes 0/32 (1/32 parsable). All three kill conditions were met. End-of-sequence now works (22.4 vs 130.9 s/task, means; kaggle/MEASUREMENT_PLAN.md §3) but repair does not follow.

### Worked example

*Historical — 1 Nov v1 adapter; superseded by v19.*

*Model output — verbatim, corrupted, do not read as code.*

Task `heldout-compound-off_by_one-0` (1 Nov 2026): the generation re-emits the buggy function, then the cue, then corrupted variants — verbatim head, trimmed with `[…]`:

```
def compound_interest(principal, rate, years):
    return principal * rate + principal
Fixed code:
defious(principal, rate, years):
    if principal < 0 or years < 0:
        return 0
    return principal * rateions // 100
[…]
```

Then it loops, never emitting end-of-sequence; the cleaned output is neither identical to the input nor parsable (raw 1,715, cleaned 1,716). Source: "Gemma 4 held-out synthetic eval pilot", Version 5, 1 Nov 2026.

Three path failures fixed (rung-2 empty context, format mismatch, missing end-of-sequence); v19 ends the looping, fixes 0/32.

## What remains unmeasured and why

Quota: 18h30m remains after the 7 Nov session. Measured now: base-model baseline 0/32, kill pilot 0/32, scorer dailies 0.00 ×7. Still **TO BE MEASURED**: a real-fix baseline, the dev-set fix rate, and full-run training sizing.

**Kill test — what falsified this.** Designed 1 Nov, frozen 5 Nov (see preregistration/ in the repo): retrain with end-of-sequence appended, then 32 held-out singles. It killed the LoRA route — 0/32 fixed, 0 operators, parsable (syntactically valid Python) 12.5% — so the failure is in the training, not the tasks. The kill pilot ran the fixed harness (MEASUREMENT_PLAN.md §1: cache-clearing verified, Kaggle copy byte-identical), but no per-task kill log was retained, so 0/32 rests on the notebook record, not a checked-in results file.

## Related work

First, benchmark-driven repair evaluation (SWE-bench [1]): patches pass held-out tests on real repositories; this work adopts that discipline with synthetic-only training. Second, agentic repair scaffolds (SWE-agent [2]; Agentless [3]): the loop matters as much as the model. Third, mutation-based training and testing: small code changes have long judged test-suite quality [4], and recent work has used synthetic bugs to generate training data with known ground truth [5][7]. All comparisons here are qualitative except the search-versus-LoRA contrast (24/32 versus 0/32).

| Approach | Real code? | Teacher model? | Deterministic? | Byte-exact regeneration? |
|---|---|---|---|---|
| This work | No | No | Yes | Yes |
| SWE-smith [5] | Yes | Yes | No | No |
| BugLab [6] | Yes | No | No | No |
| BugPilot [7] | Yes | Yes | No | No |
| SWE-Gym [8] (teacher SFT) | Yes | Yes | No | No |

The difference is the constraint, not the mechanism: in every other row the bugs live in real code or the training signal comes from a teacher model, and none is deterministic or byte-exactly regenerable. SWE-smith's procedural branch applies fixed transformations chosen at random, so it is not deterministic end-to-end. Here the synthetic corpus is the *only* training data, localised by failing-trace scope.

Toeda [11] finds scaffolding helps only the smallest model; we vary provenance instead.

## Failure modes and limitations

First, mutation realism: single-operator mutations are shallow — no missing logic, wrong API choices, misunderstood requirements, or multi-file faults; composites narrow, not close, this gap.

Second, formatting: AST unparsing reformats source, so evaluation must compare snippets, not whole-file text.

Third, context and time: the scope ladder helps only if the failing trace implicates the faulty code, and the gate only if the suite is meaningful.

Fourth, licence and corpus diversity: no external code is used, so no licence audit list exists; toy programs may be less diverse than a real-fix corpus.

Fifth, cost: zero — free Kaggle notebooks, open weights, CPU-only data preparation.

## Conclusion

Repo-Blind Repair asks whether an agent can learn to fix bugs from bugs it made itself, with provenance holding by construction. The harness is built and locally verified (33/33 harness tests, 10/10 mutation tests); the training path passed its timing pilot (20/20 steps, 65.19s; Gate 1 328/328), but Gate 2 is INCONCLUSIVE, and scorer serving is proven for inference-only packages. Pilots fix 0/18. The kill test settles the gap question for this recipe: clean-room training as run here does not transfer to repair; the dev-set fix rate and any leaderboard outcome are still **TO BE MEASURED**. The durable contribution is the resource (public; see Data and code availability): a fully synthetic, byte-exactly regenerable corpus with harness and dev-set converter. The harness is itself the application — a scope-ladder agent with a minimal-diff regression gate for any Python repository with tests.

## Data and code availability

Data and code availability: code, corpus, and scripts are at https://github.com/OlsOls4/repo-blind-repair (commit 486744a, Apache-2.0); the corpus is also on Kaggle at https://www.kaggle.com/datasets/olsols4/repo-blind-repair-synthetic-corpus-20k (v2, CC BY 4.0); the corpus regenerates byte-identically from its provenance sidecar. Reuse: `python3 data_generation/generate_scaled_dataset.py` regenerates the 20,000 records; `python3 search_baseline/search_baseline.py --task-set all116` reruns the 116-task search. It does NOT cover: toy programs only; no multi-file, API, or requirements bugs; failing_test is empty.

## References

[1] Jimenez et al., ICLR 2024 — SWE-bench: Can Language Models Resolve Real-World GitHub Issues? arXiv:2310.06770.
[2] Yang et al., NeurIPS 2024 — SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv:2405.15793.
[3] Xia et al., FSE 2025 — Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
[4] Jia & Harman, IEEE TSE 37(5), 2011 — An Analysis and Survey of the Development of Mutation Testing.
[5] Yang et al., 2025 — SWE-smith: Scaling Data for Software Engineering Agents. arXiv:2504.21798.
[6] Allamanis et al., NeurIPS 2021 — Self-Supervised Bug Detection and Repair (BugLab).
[7] Sonwane et al., 2025 — BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills. arXiv:2510.19898.
[8] Pan et al., ICML 2025 — Training Software Engineering Agents and Verifiers with SWE-Gym. arXiv:2412.21139.
[9] Liang et al., 2025 — The SWE-Bench Illusion. arXiv:2506.12286.
[10] Prathifkumar et al., 2025 — Does SWE-Bench-Verified Test Agent Ability or Model Memory? arXiv:2512.10218.
[11] Toeda, 2026 — Scaffolding Helps the Smallest Gemma 4 Agent, Not the Larger Ones. Medium, 31 Oct 2026. https://medium.com/@toeda/scaffolding-helps-the-smallest-gemma-4-agent-not-the-larger-ones-63da369962b0

