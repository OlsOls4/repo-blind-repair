<!-- Staged copy of paper/SUBMISSION.md body (Abstract through References), regenerated 2026-11-05 from the 2,820/2,839-word base. Source of truth is paper/SUBMISSION.md — do not edit this copy directly. -->

## Abstract

**Offline coding agents** are usually trained on real human bug fixes, so repair is hard to tell from recall on hidden tasks.

**Repo-Blind Repair** removes that ambiguity for its fine-tuning data by construction: the model is trained only on synthetic bugs from deterministic AST mutation of toy Python programs written for this task — never a real human fix, a real repository, or the competition's hidden tasks. The guarantee covers the fine-tuning data only, not base pre-training. The fixed code in every record is exactly the original code, so ground truth is known. The agent repairs bugs inside a scope ladder with a minimal-diff regression gate.

**The headline finding** is a diagnosed failure mechanism, not a score: training never taught end-of-sequence after the fixed file, so at evaluation the model runs to the token cap looping corrupted "Fixed code:" fragments instead of stopping. Held-out pilots fix 0/10 on a compound-heavy slice (Wilson 95% 0.00–0.28) and 0/8 on stratified singles (0.00–0.32). The corpus, harness, and dev-set converter are offered as a contamination-controlled resource for the competition's "Tasks & Benchmarks" topic. Fix rates, gap measurements, and leaderboard outcomes remain **TO BE MEASURED**; the results below are preliminary evidence.

### At a glance

- **Claim:** repair without recall.
- **Corpus:** 20,000 records.
- **Harness:** 32/32, 9/9.
- **Headline:** 0/18 pilots; diagnosed end-of-sequence failure; kill test pending.
- **Resource:** link to follow.

## Introduction

Coding agents are increasingly asked to work where code cannot leave the machine — private codebases, regulated teams, repositories barred from cloud APIs. The standard recipe — fine-tuning on real issues and human-written patches — makes two claims hard to audit: generalisation, because a high score may reflect memorisation; and provenance, because an adopting team cannot easily verify that their code, a competitor's, or the benchmark itself was excluded.

Repo-Blind Repair makes both claims auditable; the corpus inventory in Methods is the audit trail. The training data contains no real bug fixes at all. Instead, clean code is deliberately broken by deterministic transformations, and the model learns to restore it. Because each bug is a known transformation of known clean code, every training example carries its own ground truth: fix, bug class, changed node. Nothing in fine-tuning depends on a human patch, a teacher model, or a private repository. One scope note: the clean-room claim covers the fine-tuning data and its LoRA, not the base checkpoint's pre-training.

Real human bugs include missing logic, wrong API use, and multi-file faults, which single-operator mutations do not cover. The scientific question is therefore quantitative: how far does synthetic training transfer, where does it fail, and which additions close the gap? That question is the contribution; the baseline comparison, bug-class breakdown, and failure audit remain **TO BE MEASURED**.

### In plain English

We break clean toy programs in known ways and teach the model to fix them, so a good score cannot come from memorised fixes. The model sees only broken code and must restore the original. If it learns the trick, it should fix bugs it has never seen.

One real training record (off_by_one):

Buggy (mutated function only):
```
def loyalty_points(spent):
    points = spent // 3
    if spent >= 403:
        points = points + 25
    return points
```

Fixed:
```
def loyalty_points(spent):
    points = spent // 2
    if spent >= 403:
        points = points + 25
    return points
```

## Methods

### Clean-room training data: deterministic mutation

Training records are produced locally, with only the Python standard library, by a deterministic AST mutation pipeline: a clean Python file, one or two operators, and a record with five fields: `buggy_code` (full mutated file), `fixed_code` (full original file), `mutations` (file, line, operator, original versus mutated snippet), `failing_test` (empty — none invented), and `bug_class` (the operator, or `composite:op1+op2`).

Eight operators, each reproducible from an (operator, seed) pair: **off_by_one** (integer ±1), **swapped_operands**, **wrong_comparator** (`<` to `<=`), **dropped_condition** (one conjunct removed), **swapped_boolean**, **wrong_constant** (shift ≥2), **removed_return**, and **removed_assignment** (replaced by `pass`). Candidates are collected in deterministic AST walk order, the target selected by `seed % len(candidates)`; only that node changes. Every mutated file is verified to still parse.

Single-operator records teach localisation of one fault; composites apply two in sequence and hypothesise narrowing the realism gap — no pilot support exists yet.

Corpus and provenance inventory (offline, from the validation report). The evaluated corpus, dataset v18, uses only templates written for this task: 82 parameterised shapes (62 general + 20 boolean-rich), rebuilt into 2,126 sources, 1,166 of which appear in the 20,000 records — 14,000 singles at 1,750 per operator, plus 6,000 composites across 14 pairs (25–709 per pair, uneven because pairs differ in applicability). No real human bug fix, no teacher-model output, and no third-party code was used. A provenance sidecar records (source_index, operators, seed) per record; regenerating all 20,000 records from it matches exactly (0 mismatches). Every record parses, buggy differs from fixed throughout, and `failing_test` is empty. A token-trigram containment audit of the 116 held-out tasks against the 1,166 unique clean sources finds no near-duplicates: max 0.39, none above 0.50 (kaggle/audit_near_dup.py, sha256 6192d2f1).

### Repair harness: scope ladder, context packer, regression gate

The repair agent runs inside an offline, standard-library-only harness with four components.

First, a trace extractor parses failing test output and issue text into failing tests, traceback frames, and implicated files and symbols. Second, a scope ladder controls where the agent looks: rung 1, files named by the failing trace; rung 2, imports and callers found by AST analysis; rung 3, wider repository search. Each rung is time-boxed (30, 60, 120 seconds), advancing only on failure.

Third, a context packer assembles rung-ordered file contents under a token budget (characters divided by four, cap 32,768), truncating with markers. Fourth, a test runner executes unittest identifiers in subprocesses with hard timeouts, and a regression gate accepts a patch only if the failing test passes and a capped suite (at most 15 tests, 90 seconds, abort on first failure) stays green.

Patch application is conservative: an edit applies only if its old text occurs exactly once in the file; otherwise the patch is rejected atomically, nothing half-applied. The harness never touches the network, calls a model, or writes outside its repository. Patch generation is a marked placeholder until the LoRA generator exists; a missing model raises an error, never a silent "no patch needed".

The intended model is a Gemma 4 12B LoRA (QAT checkpoint format) trained on the records above, with the same repair behaviour in local evaluation and in the scorer's agent tree.

## Results

| Check | Result |
|---|---|
| Harness suite | 32/32 tests pass |
| Mutation suite | 9/9 tests pass |
| Held-out pilots | 0/10 compound-heavy (0.00–0.28); 0/8 stratified singles (0.00–0.32) |
| Dev-set derivation | 3 of 20 tasks converted, 15 regressions each (not a fix rate); output deleted 5 Nov 2026 |

### Harness

The harness suite passes **32/32 tests**; its demo accepts a one-line rung-1 patch, after which the failing test passes and regressions stay green. The mutation suite passes **9/9 tests**; its demo checks five example records: each parses, differs from the original, and reproduces identically from the same (operator, seed).

Training-path verification, measured 31 Oct 2026: all 328 target modules are packed-weight `Linear` with no plain `.weight`; `v_proj` is absent on 8 global layers using K=V attention. PEFT could not wrap the base (it requires torchao >0.16.0; the environment has 0.10.0), so the pilot used manual LoRA in pure torch. Gate 1 (adapter attaches to all target modules) **PASSED 328/328** with `run_compressed=False`. The manual-LoRA pilot then **SUCCEEDED 20/20 steps in 65.19s, median 0.339s/step**; its reload check **PASSED**. Timing only — explicitly **not** a repair-quality result. Gate 2 (adapter discovery and serving) was **OVERALL INCONCLUSIVE**: Stage 1 **Passed**; Stage 2 **not testable here** (the reproduced discovery check failed on the tokenizer's chat template file); Stage 3 **not yet tested** (no vLLM in the Kaggle notebook environment).

### Pilots

Held-out synthetic evaluation, measured 1 Nov 2026 (synthetic transfer only): 116 tasks from 18 new domains, audited overlap-free (see Methods), every derived test verified fail-on-buggy, pass-on-fixed. Pilot slices fix 0/10 on a compound-heavy file-order slice (5 gate_rejected — patch produced but the failing test still failed — and 5 no_patch — no candidate produced) and 0/8 on a stratified single-bug slice (7 no_patch, 1 gate_rejected) — failure is not confined to two-fault tasks. Intervals are per slice, not pooled: 0/10 gives Wilson 95% 0.00–0.28 and 0/8 gives 0.00–0.32; neither proves a true rate of zero. Only 18 of 116 tasks were piloted: the two slices ran on 1 Nov before the GPU quota was exhausted.

Adapter evaluated (identity, not a result). All 1 Nov held-out slices ran via local eval (kaggle/eval_dev.py) on checkpoint-opt-10000 from the committed dataset-v18 retrain (Version 1, 1 Nov 04:57 GMT; 10,000 steps; training loss 8.5253 to 0.0191, **TRAINING LOSS ONLY**) — not the packaged inference-only variant.

The repair-vs-recall question is now measured: models identify buggy files from issue text alone at up to 76% on SWE-Bench Verified versus 53% off-benchmark [9], and score three times better on Verified localisation than on fresh benchmarks [10].

### Diagnostics

Gate diagnostics on 3 gate_rejected tasks: patches apply cleanly, but the failing test still fails on all 3 — plausible but incorrect fixes. A 2-task generation diagnostic found degenerate looping output.

### Worked example

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

def bulk_price(unit, n):
    total = unit * n
    if n > 10::
        total = total * 93 // 10
    return total
[…]
```

Then it loops: a second `Fixed code:` and `def compound(principal, rate,_ years):` repeat the earlier body. The model never emits end-of-sequence, runs to the 512-token cap, and the cleaned output is neither identical to the input nor parsable (raw 1,715, cleaned 1,716, identical `False`, parses `False`). Source: "Gemma 4 held-out synthetic eval pilot", Version 5, 1 Nov 2026.

Three evaluation-path failure modes were fixed: a rung-2 empty-context crash (skip and guard); a train/eval format mismatch (trained on "Fixed code:", evaluated on FILE:/OLD:/NEW: blocks); and missing end-of-sequence after fixed files, so generation ran to the 512-token cap looping "Fixed code:" fragments, mitigated by truncating at a bare cue line. Retraining with end-of-sequence appended, the principled fix, waits for the weekly quota reset (no reset date shown).

### Derivation

Dev-set derivation (not a fix rate): a CPU-only converter derived one validated failing test plus 15 regressions for 3 of the first 20 public dev tasks (10 fastapi, 8 requests, 2 rich) — fastapi_14962, fastapi_15280, fastapi_14978, all fastapi, so the subset is selection-biased — its notebook was deleted 5 Nov 2026 (audit: files embedded public dev-set data); the other 17 skipped for diagnosed reasons — 12 test-collection failures, 3 ambiguous multi-candidate derivations, 1 with no failing-test survivor, 1 patch touching no test function. The submission package is starter-shaped YAML; its inference-only zip verifies locally (4 files, 13,342 bytes, exit 0) — packaging, not scorer validation for this zip; the sibling E4B zip has since run at the scorer (SUCCEEDED, Public Score 0.00, inference-only). Verifiability is by artifact: the provenance sidecar regenerates all 20,000 records byte-identically.

## What remains unmeasured and why

The GPU quota is fully used (30:00/30 hours, 0 available, no reset date as of 5 Nov 2026), so every GPU-dependent measurement waits for the reset. Nothing is promised before the 12 Nov paper deadline: the fix rate, the gap to a real-fix baseline, and any leaderboard outcome remain **TO BE MEASURED**. Once quota returns: base-model baseline, kill test, real-fix baseline comparison, scored dry run. The dev-set fix rate is **TO BE MEASURED** (derivation for 3 of 20 tasks verified above); full-run training sizing is **TO BE MEASURED**. This writeup reports the harness, mutation taxonomy, pilot timing, gate results, and measurements.

**What would falsify this.** A pre-registered kill test, designed 1 Nov and not yet run: retrain with end-of-sequence appended and the bug-class label dropped, then a 32-task stratified held-out singles pilot. KILL the LoRA route if fewer than 4/32 are fixed, if fixes fall in at most one operator, or if parsable (syntactically valid Python) non-identical output is below 30%. The test is prospective: designed before the retrain it governs, which has not run, so nothing here is retroactively pre-registered.

## Related work

First, benchmark-driven repair evaluation (SWE-bench [1]): patches pass held-out tests on real repositories; this work adopts that discipline with synthetic-only training. Second, agentic repair scaffolds (SWE-agent [2]; Agentless [3]): the loop matters as much as the model, and the scope ladder and regression gate follow that tradition. Third, mutation-based training and testing: small code changes have long judged test-suite quality [4], and recent work has used synthetic bugs to generate training data with known ground truth [5][7]. All comparative statements here are qualitative; quantitative comparisons are **TO BE MEASURED**.

| Approach | Real code? | Teacher model? | Deterministic? | Byte-exact regeneration? |
|---|---|---|---|---|
| This work | No | No | Yes | Yes |
| SWE-smith [5] | Yes | Yes | No | No |
| BugLab [6] | Yes | No | No | No |
| BugPilot [7] | Yes | Yes | No | No |
| SWE-Gym [8] (teacher SFT) | Yes | Yes | No | No |

The difference is the constraint, not the mechanism: in every row but ours the bugs live in real code or the training signal comes from a teacher model, and none is deterministic end-to-end or byte-exactly regenerable. Columns record whole-pipeline properties: real code, teacher signal, same inputs giving the same corpus, and byte-identical regeneration. SWE-smith's procedural branch applies fixed transformations chosen at random, so it is not deterministic end-to-end. Here the synthetic corpus is the *only* training data, localised by failing-trace scope.

Toeda [11] asks the complementary question, varying scaffold by model size in a pre-registered study (104 of 129 public tasks, 4,056 runs): scaffolding helps only the smallest model (E4B +5.4 points); the 12B minimal scaffold resolves 25.2%. We vary training provenance instead — different axes, not rival claims.

## Failure modes and limitations

First, mutation realism. Single-operator mutations are shallow: no missing logic, wrong API choices, misunderstood requirements, or multi-file faults; composites narrow but do not close this gap.

Second, formatting. AST unparsing reformats source; diff-based evaluation must compare snippets, not whole-file text.

Third, context and time. The scope ladder helps only if the failing trace implicates the faulty code, and the regression gate only if the test suite is meaningful.

Fourth, licence and corpus diversity. The corpus uses no external code, so no licence audit list exists; toy-program sources may be less diverse than a real-fix corpus.

Fifth, cost and compute. The plan is zero-cost: free Kaggle notebooks, open weights, CPU-only data preparation.

## Conclusion

Repo-Blind Repair asks whether an agent can learn to fix bugs from bugs it made itself, under a provenance property holding by construction for its fine-tuning data. The harness is built and locally verified (32/32 harness tests, 9/9 mutation tests, both demos passing); the training path has passed its timing pilot (20/20 steps, 65.19s; Gate 1 328/328), but Gate 2 is INCONCLUSIVE and scorer serving is UNPROVEN. Held-out synthetic pilots fix 0/10 on a compound-heavy slice (Wilson 95% 0.00–0.28) and 0/8 on stratified singles (0.00–0.32); neither proves a true rate of zero. The dev-set fix rate, the synthetic-to-real gap, and any leaderboard outcome are **TO BE MEASURED**. If the gap proves small, clean-room training is a credible route to offline repair; if large, the measurement shows where it stops. The durable contribution is the resource: a fully synthetic, byte-exactly regenerable corpus with harness and dev-set converter, built for contamination-controlled measurement.

## Data and code availability

Data and code availability: dev-set derivation at the Kaggle link in Results; the corpus regenerates byte-identically from its provenance sidecar; no public repository exists.

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
