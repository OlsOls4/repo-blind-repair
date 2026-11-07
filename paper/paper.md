<!-- Staged copy of paper/SUBMISSION.md body (Abstract through References), regenerated 2026-11-07 from the 2,831/2,845-word base (wave-6 kill-test patch). Source of truth is paper/SUBMISSION.md — do not edit this copy directly. -->

## Abstract

**KILL RESULT (7 Nov): retrained LoRA (low-rank adaptation) fixes 0/32; end-of-sequence fix works but repairs nothing. LoRA route killed.**

**Offline coding agents** are usually trained on real human bug fixes, so repair is hard to tell from recall on hidden tasks.

**Repo-Blind Repair** removes that ambiguity for its fine-tuning data by construction: the model is trained only on synthetic bugs from deterministic AST mutation of toy Python programs written for this task — never a real human fix, a real repository, or the competition's hidden tasks. The guarantee covers the fine-tuning data only, not base pre-training. Every record's fixed code is exactly the original, so ground truth is known. The agent repairs bugs inside a scope ladder with a minimal-diff regression gate.

**The headline finding** is a diagnosed failure mechanism, not a score: training never taught end-of-sequence (EOS) after the fixed file, so at evaluation the model runs to the token cap looping corrupted "Fixed code:" fragments. Held-out pilots fix 0/10 on a compound-heavy slice (Wilson 95% 0.00–0.28) and 0/8 on stratified singles (0.00–0.32). The corpus, harness, and dev-set converter are offered as a public, contamination-controlled resource for the competition's "Tasks & Benchmarks" topic (see Data and code availability).

### At a glance

- **Claim:** repair without recall.
- **Corpus:** 20,000 records.
- **Harness:** 33/33, 9/9.
- **Headline:** 0/18 pilots; kill test run 7 Nov — LoRA 0/32, route KILLED.
- **Resource:** public — see Data and code availability.

## Introduction

Coding agents increasingly work where code cannot leave the machine — private codebases, regulated teams, repositories barred from cloud APIs. The standard recipe — fine-tuning on real issues and human-written patches — makes two claims hard to audit: generalisation, because a high score may reflect memorisation; and provenance, because an adopting team cannot easily verify that their code, a competitor's, or the benchmark itself was excluded.

Repo-Blind Repair makes both claims auditable; the corpus inventory in Methods is the audit trail. The training data contains no real bug fixes at all: clean code is deliberately broken by deterministic transformations, and the model learns to restore it. Because each bug is a known transformation of known clean code, every training example carries its own ground truth — fix, bug class, changed node. Nothing in fine-tuning depends on a human patch, a teacher model, or a private repository. The clean-room claim covers the fine-tuning data and its LoRA, not the base checkpoint's pre-training.

The scientific question is therefore quantitative: how far does synthetic training transfer, where does it fail, and which additions close the gap? That question is the contribution; the baseline comparison remains **TO BE MEASURED**; the failure audit is now measured (see Results).

### In plain English

We break clean toy programs in known ways and teach the model to fix them, so a good score cannot come from memorised fixes. The model sees only broken code and must restore the original.

One real training record (off_by_one, record 344) divides by 3, not 2.

## Methods

### Clean-room training data: deterministic mutation

Training records are produced locally, with only the Python standard library, by a deterministic AST mutation pipeline: a clean Python file, one or two operators, and a record with five fields: `buggy_code` (full mutated file), `fixed_code` (full original file), `mutations` (file, line, operator, original versus mutated snippet), `failing_test` (empty — none invented), and `bug_class` (the operator, or `composite:op1+op2`).

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
| removed_assignment | replace with `pass` | Candidates are collected in deterministic AST walk order; the target is `seed % len(candidates)`, and only that node changes. Every mutated file still parses.

Single-operator records teach localisation of one fault; composites apply two operators, hypothesising a narrower realism gap, with no pilot support yet.

Corpus inventory (offline, from the validation report). Dataset v18 uses only templates written for this task: 82 parameterised shapes (62 general + 20 boolean-rich), rebuilt into 2,126 sources, 1,166 of which appear in the 20,000 records — 14,000 singles at 1,750 per operator, plus 6,000 composites across 14 pairs (25–709 per pair, uneven because pairs differ in applicability). No real human bug fix, no teacher-model output, and no third-party code was used. A provenance sidecar records (source_index, operators, seed) per record; regenerating all 20,000 records from it matches exactly (0 mismatches). Every record parses, buggy differs from fixed throughout, and `failing_test` is empty. A token-trigram containment audit of the 116 held-out tasks against the 1,166 unique clean sources finds no near-duplicates: max 0.39, none above 0.50 (kaggle/audit_near_dup.py, sha256 6192d2f1). The 0.50 threshold marks substantive overlap: above it, a held-out task shares over half its token trigrams with a training source.

### Repair harness: scope ladder, context packer, regression gate

The repair agent runs inside an offline, standard-library-only harness with four components.

First, a trace extractor parses failing test output and issue text into failing tests, traceback frames, and implicated files. Second, a scope ladder controls where the agent looks, advancing only on failure:

| Rung | Looks at | Time-box |
|---|---|---|
| 1 | Files named by the failing trace | 30 s |
| 2 | Imports and callers (AST analysis) | 60 s |
| 3 | Wider repository search | 120 s |

Third, a context packer assembles rung-ordered file contents under a token budget (characters divided by four, cap 32,768), truncating with markers. Fourth, a test runner executes unittest identifiers with hard timeouts, and a regression gate accepts a patch only if the failing test passes and a capped suite (max 15 tests, 90 seconds, abort on first failure) stays green.

Patch application is conservative: an edit applies only if its old text occurs exactly once; otherwise it is rejected atomically, nothing half-applied. The harness never touches the network, calls a model, or writes outside its repository. Patch generation is a marked placeholder until the LoRA generator exists; a missing model raises an error, never silent "no patch needed".

The intended model is a Gemma 4 12B LoRA (quantisation-aware training (QAT) checkpoint format) trained on the records above.

## Results

| Check | Result |
|---|---|
| Harness unit tests (not repair) | 33/33 tests pass |
| Mutation unit tests (not repair) | 9/9 tests pass |
| Held-out pilots (pre-retrain, 1 Nov) | 0/10 compound-heavy (0.00–0.28); 0/8 stratified singles (0.00–0.32) |
| Kill pilot (7 Nov) | 0/32; parsable 4/32 (12.5%); 0 operators |
| Base-model baseline | 0/32; parsable 1/32 |
| Dev-set derivation | 3 of 20 tasks converted, 15 regressions each (not a fix rate); output deleted 5 Nov 2026 |

*Numbers: 0/18 = LoRA pilots (v1), 0/32 = kill pilot (v19), 24/32 = model-free search (not the model), 33/33 = harness unit tests (not repair).*

### Harness

The harness suite passes **33/33 tests**; its demo applies a one-line rung-1 patch; the failing test passes and regressions stay green. The mutation suite passes **9/9 tests**; its demo checks five example records: each parses, differs from the original, and reproduces identically.

Training-path verification (31 Oct 2026): 328 packed-weight target modules, no plain `.weight`; manual LoRA (PEFT could not wrap the base). Gate 1 (adapter attaches to all target modules) **PASSED 328/328**. The pilot **SUCCEEDED 20/20 steps in 65.19s**; reload **PASSED**. Timing only — not a repair-quality result. Gate 2 was **OVERALL INCONCLUSIVE**.

### Pilots

Held-out synthetic evaluation, 1 Nov 2026: 116 tasks from 18 new domains, audited overlap-free (see Methods), every derived test verified fail-on-buggy, pass-on-fixed. Pilot slices fix 0/10 on a compound-heavy file-order slice (5 gate_rejected — candidate produced but failing test still failed — and 5 no_patch — no candidate produced) and 0/8 on a stratified single-bug slice (7 no_patch, 1 gate_rejected) — failure is not confined to two-fault tasks. Intervals are per slice, not pooled: 0/10 gives Wilson 95% 0.00–0.28 and 0/8 gives 0.00–0.32; neither proves a true rate of zero (a Wilson interval is the plausible range for the true rate). Only 18 of 116 tasks were piloted before the GPU quota was exhausted; the kill test has since run 32 more (overlap unknown).

Caveat: the first baseline run scored 14/32 — CPython loaded stale bytecode when same-size candidates were rewritten within the same second. The harness now clears __pycache__ before every run and sets PYTHONDONTWRITEBYTECODE=1; the 6 gate-rejected pilot results may include false negatives, but the 12 no_patch failures and looping diagnostics are unaffected.

Adapter evaluated (identity, not a result): all 1 Nov slices ran via local eval on checkpoint-opt-10000 (Version 1, 1 Nov 04:57 GMT; 10,000 steps; training loss 8.5253 to 0.0191, **TRAINING LOSS ONLY**) — not the packaged inference-only variant. The 1 Nov slices used v1 checkpoint-opt-10000, not v19.

The repair-vs-recall question is now measured: models identify buggy files from issue text alone at up to 76% on SWE-Bench Verified versus 53% off-benchmark [9], and score three times better on Verified localisation than on fresh benchmarks [10].

### Search baseline

A model-free inverse-operator search on the frozen 32-task kill set fixes 24/32 (75%, Wilson 95% 0.58–0.87); 20/32 (62.5%, 0.45–0.77) by the matched inverse. Five invertible classes fix 4/4 matched; dropped_condition 4/4 any-inverse, 0 matched (probe-weakness passes); removed_return and removed_assignment 0/4 (no inverse). Search baseline, not learned repair: the tasks are not hard. The LoRA's 0/18 was a format failure — it never emitted end-of-sequence — but with end-of-sequence fixed, the retrained adapter still fixes 0/32, none of the 24 search fixes. The kill test measures format learning: learned hard core 32/32; search hard core 8.

### Kill test

The pre-registered kill test ran 7 Nov 2026 (v19: end-of-sequence appended, label dropped). The retrained adapter fixes 0/32 (Wilson 95% 0.00–0.11); 4/32 are parsable (12.5%, Wilson ~0.05–0.28), in 0 operators; the base model also fixes 0/32 (1/32 parsable). All three kill conditions were met: no epoch 2, no Gate 2. End-of-sequence now works (13.8 vs 130.9 s/task) but repair does not follow.

### Diagnostics

Gate diagnostics on 3 gate_rejected tasks: patches apply cleanly, but the failing test still fails — plausible but incorrect fixes. A 2-task generation diagnostic found degenerate looping output.

### Worked example

*Historical — 1 Nov v1 adapter; superseded by v19.*

*Model output — verbatim, corrupted, do not read as code.*

Task `heldout-compound-off_by_one-0` (1 Nov 2026). The generation re-emits the buggy function, then the cue, then corrupted variants — verbatim head, trimmed with `[…]`:

```
def compound_interest(principal, rate, years):
    return principal * rate + principal
Fixed code:
defious(principal, rate, years):
    if principal < 0 or years < 0:
        return 0
    return principal * rateions // 100
[…]
Fixed code:
def compound(principal, rate, years):
```

Then it loops: the second `Fixed code:` repeats the earlier body. The model never emits end-of-sequence, runs to the 512-token cap, and the cleaned output is neither identical to the input nor parsable (raw 1,715, cleaned 1,716). Source: "Gemma 4 held-out synthetic eval pilot", Version 5, 1 Nov 2026.

Three evaluation-path failure modes were fixed: a rung-2 empty-context crash; a train/eval format mismatch (trained on "Fixed code:", evaluated on FILE:/OLD:/NEW: blocks); and missing end-of-sequence after fixed files, so generation ran to the 512-token cap looping "Fixed code:" fragments, mitigated by truncating at a bare cue line. Retraining with end-of-sequence appended — the principled fix — ran 7 Nov as v19, ends the looping, fixes 0/32.

### Derivation

Dev-set derivation (not a fix rate): a CPU-only converter derived one validated failing test plus 15 regressions for 3 of the first 20 dev tasks; its notebook was deleted 5 Nov 2026 (audit: files embedded dev-set data). The sibling E4B zip has run at the scorer (SUCCEEDED, Public Score 0.00, inference-only).

## What remains unmeasured and why

Quota reset; the first GPU session ran 7 Nov, 18h30m remains. Measured now: base-model baseline 0/32, kill pilot 0/32, scorer dailies 0.00 ×3. Still **TO BE MEASURED**: a real-fix baseline, the dev-set fix rate, and full-run training sizing. No further epoch.

**Kill test — what falsified this.** Designed 1 Nov, frozen 5 Nov: retrain with end-of-sequence appended, then 32 held-out singles. It killed the LoRA route — 0/32 fixed, 0 operators, parsable (syntactically valid Python) 12.5% — so the failure is in the training, not the tasks. Caveat: cache-fix status of the kill pilot unverified — 0/32 may include false negatives.

## Related work

First, benchmark-driven repair evaluation (SWE-bench [1]): patches pass held-out tests on real repositories; this work adopts that discipline with synthetic-only training. Second, agentic repair scaffolds (SWE-agent [2]; Agentless [3]): the loop matters as much as the model, and the scope ladder and regression gate follow it. Third, mutation-based training and testing: small code changes have long judged test-suite quality [4], and recent work has used synthetic bugs to generate training data with known ground truth [5][7]. All comparisons here are qualitative except the search-versus-LoRA contrast on the frozen 32 (24/32 versus 0/32).

| Approach | Real code? | Teacher model? | Deterministic? | Byte-exact regeneration? |
|---|---|---|---|---|
| This work | No | No | Yes | Yes |
| SWE-smith [5] | Yes | Yes | No | No |
| BugLab [6] | Yes | No | No | No |
| BugPilot [7] | Yes | Yes | No | No |
| SWE-Gym [8] (teacher SFT) | Yes | Yes | No | No |

The difference is the constraint, not the mechanism: in every other row the bugs live in real code or the training signal comes from a teacher model, and none is deterministic end-to-end or byte-exactly regenerable. SWE-smith's procedural branch applies fixed transformations chosen at random, so it is not deterministic end-to-end. Here the synthetic corpus is the *only* training data, localised by failing-trace scope.

Toeda [11] varies scaffold by size, pre-registered (104 tasks, 4,056 runs): scaffolding helps only E4B (+5.4 points); the 12B minimal scaffold resolves 25.2%. We vary provenance instead.

## Failure modes and limitations

First, mutation realism: single-operator mutations are shallow — no missing logic, wrong API choices, misunderstood requirements, or multi-file faults; composites narrow but do not close this gap.

Second, formatting: AST unparsing reformats source, so diff-based evaluation must compare snippets, not whole-file text.

Third, context and time: the scope ladder helps only if the failing trace implicates the faulty code, and the gate only if the test suite is meaningful.

Fourth, licence and corpus diversity: no external code is used, so no licence audit list exists; toy-program sources may be less diverse than a real-fix corpus.

Fifth, cost: zero-cost — free Kaggle notebooks, open weights, CPU-only data preparation.

## Conclusion

Repo-Blind Repair asks whether an agent can learn to fix bugs from bugs it made itself, with provenance holding by construction for its fine-tuning data. The harness is built and locally verified (33/33 harness tests, 9/9 mutation tests); the training path has passed its timing pilot (20/20 steps, 65.19s; Gate 1 328/328), but Gate 2 is INCONCLUSIVE, and scorer serving is proven for inference-only packages. Held-out synthetic pilots fix 0/10 on a compound-heavy slice (Wilson 95% 0.00–0.28) and 0/8 on stratified singles (0.00–0.32); neither proves a true rate of zero. The kill test settles the gap question for this recipe: clean-room training as run here does not transfer to repair; the dev-set fix rate and any leaderboard outcome are still **TO BE MEASURED**. The durable contribution is the resource (public; see Data and code availability): a fully synthetic, byte-exactly regenerable corpus with harness and dev-set converter, built for contamination-controlled measurement. The harness is itself the application — a scope-ladder agent with a minimal-diff regression gate for any Python repository with tests.

## Data and code availability

Data and code availability: code, corpus, and scripts are public at https://github.com/OlsOls4/repo-blind-repair (commit f796a284, Apache-2.0); the corpus is also on Kaggle at https://www.kaggle.com/datasets/olsols4/repo-blind-repair-synthetic-corpus-20k (v2, CC BY 4.0); the corpus regenerates byte-identically from its provenance sidecar. It does NOT cover: toy programs only; no multi-file, API, or requirements bugs; failing_test is empty; harness patch generation is a placeholder.

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
