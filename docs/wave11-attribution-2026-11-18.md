# Wave 11 follow-up — attribution of the shadow scorer's first successful run

**Date:** 18 Nov 2026. **Coordinator synthesis of four independent panels** (Diagnosis,
Researchers, Principal Software Engineers, Judges), run in parallel with no shared
verdicts before initial assessments. **Status:** all four panels completed; this report
is the coordinator's synthesis, not any single panel's verdict.

## 0. Verdict

**The headline is not 25/32. It is 17/32 gold-equivalent (25/32 as-graded by the visible
3-test suites) — and the loop still works on simple tasks.** The Engineers panel's D7
correction, verified by the coordinator from the patches themselves: 8 of the 25
"resolved" pass the 3 visible tests by deleting or weakening the guard the hidden tests
would check. All 8 guard tasks (4 `dropped_condition`, 4 `swapped_boolean`) "resolved"
and **0/8 are gold-equivalent**; on non-guard tasks the model resolved 17/24.

The dominant failure mode is new and was found independently by all four panels: **a
repetition attractor / failure-to-edit**. In 6 of 7 failures the agent repeated one
identical tool call or one identical no-op edit 13–28 times until the 5-minute budget
died, with zero harness intervention — because nudges only fire when *no* tool call is
emitted, and these turns always emit a parseable call. The K2 prompt (22 words) contains
no anti-repetition rule, no edit discipline, and no probe-variation instruction.

**What the run proves:** the loop + real 12B can fix simple synthetic bugs (17/32 gold,
25/32 as-graded, median 45.8 s for resolved tasks), parse/format is healthy *in the
replica* (333/334 turns standard tier), and the Kaggle zeros are not "the model can't
fix anything", not a serving collapse, not a gate/revert artefact, and not a time
shortage (every failure had 300 s and needed one edit).

**What it does not prove:** anything about the Kaggle scorer's vLLM serving (replica is
transformers), anything about real-repo scale (synthetic tasks are 7-line single files),
or that any prompt tweak will score >0. A replica rate is not a Kaggle rate (DESIGN
§3 red line 4; §9).

**Coordinator's decision — the one recommended next step:** wait for the free Day S
scorer result tomorrow (Thu 19 Nov ~09:59 GMT), then, **if Day S = 0, test an
anti-repetition + edit-first prompt arm in the replica on the frozen 32** — the only
replica-directed, single-variable, scorer-faithful fix with a predicted mechanism the
traces support. It is the only candidate next step that can *produce a daily*; the
scale probe can only produce information. Full sequencing in §7. **If Day S > 0, the
branch rule governs instead: repeat the identical Day S package; replica work pauses.**

## 1. What the run actually showed (coordinator-verified numbers)

Attempt 7, 18 Nov, L4x4, real 12B, dequantized load, K2 package (Toeda min verbatim,
temp 1.0, 6 tools, 150 s / 50 calls / 5 min / 80 turns). Source:
`kaggle/shadow-scorer/runs/attempt7/k2-kill32-summary.json` (334 turns) and
`k2-probe4-summary.json` (46 turns), parsed from saved notebook V3.

- **Probe 4/4 tasks, 3 resolved**, gate PASSED (projection 3.00 h vs 6.4 h budget).
- **Full frozen 32: 25 resolved, 4 budget_exhaustion, 3
  verification_failed:failing_still_fails** (total 5,947.5 s). Zero R0 replica faults.
- **Per-operator (as-graded):** dropped_condition 4/4, off_by_one 2/4,
  removed_assignment 4/4, removed_return 3/4, swapped_boolean 4/4, swapped_operands 2/4,
  wrong_comparator 3/4, wrong_constant 3/4.
- **Per-operator (gold-equivalent, coordinator-verified from patches):** guard tasks
  0/8; non-guard 17/24. The 8 probe-weak resolves: all 4 `dropped_condition` patches
  delete the guard and flip the default to `False` (e.g. `circle_area(radius,
  checked=False)` with no guard; visible test only calls `circle_area(5)`); 3
  `swapped_boolean` patches drop `checked and`/`safe and` from the guard (gold is
  flipping the default back to `True`); `dilution-swapped_boolean-0` adds an
  unconditional `if c2==0` and leaves `safe=False`. A hidden test probing the guard
  (negative input with `checked=True`, or `checked=False`) fails all 8. Confidence:
  high for the 4 dropped_condition, medium-high for the 4 swapped_boolean (gold
  inferred from operator semantics, no stored reference patch — DESIGN §1 notes this
  gap; Engineers' fix #2 would turn this into measurement).
- **Loop health:** 333/334 turns `finish_reason: stop`, parse tier standard 333, empty
  1; nudges sent: 1 (`max_tokens`, isbn); compaction events: 0; `get_status` calls: 0;
  max prompt estimate 5,432 tokens (compaction threshold 14,336 never approached);
  `would_have_timed_out`: 0/334 — including isbn turn 11 (see §5, defect 2).
- **Resolved tasks:** median 45.8 s, 5–9 calls. **Failures:** all `stop_reason:
  budget_time`. Loan 8/14 resolved vs non-loan 17/18.
- **Stochasticity:** `heldout-loan-off_by_one-0` resolved in the probe (54.7 s, 8
  calls) but budget-exhausted in the full 32 (300.1 s, 31 calls) — same package, same
  task, temp 1.0, no seed. Single-run counts carry Toeda-scale noise (7.5–16%
  run-to-run disagreement).

## 2. The 7-failure attribution table (Diagnosis, coordinator-checked)

DESIGN §4 taxonomy; "first decisive event → terminal state". All 7 stop
`budget_time`; `consecutive_nudges = 0` in all 7 (nudges never fired — every turn
emitted a parseable call).

| Task | §4 | Mechanism | Key evidence |
|---|---|---|---|
| `heldout-loan-off_by_one-0` | **8** budget_exhaustion, clean tree | Probe loop | T1 read (`/13` vs `/12` in `monthly_rate`), T2 pytest 1061.01 vs 1066.19, T8–T31 identical probe ×24, 0 edits, 300.1 s |
| `heldout-loan-swapped_operands-1` | **8** | Probe loop | T1 read (`factor = months ** (1 + r)`), T2 pytest 130.62 vs 1066.19, T4–T31 identical probe ×28, 0 edits, 300.7 s |
| `heldout-loan-wrong_constant-0` | **8** | Probe loop | T1 read (`/14`), T6 probed `monthly_rate(0.12)=0.01` (the clue), T7–T32 identical probe ×26, 0 edits, 303.0 s |
| `heldout-isbn-wrong_comparator-0` | **8** (mechanism distinct) | Runaway length generation, identified-but-didn't-act | T4–T10 probes (weighted_sum 123, check_digit 5, expected 7); T11 16,384 tok, 2,594.8 s, `finish_reason=length`, 0 calls; raw text correctly derives "change `!=` to `==`" then re-derives arithmetic in loops; 2,694.0 s total |
| `heldout-loan-off_by_one-2` (kill32) | **7** budget_exhaustion, partial edit in tree | Wrong fix + probe loop | T5 wrong edit (numerator `(months - 1)`), T6–T31 identical probe of 1055.629 ×26, no pytest after T5, 308.7 s. Final patch: `return principal * r * (1 + r) ** (months - 1) / (factor - 1)` |
| `heldout-loan-swapped_operands-6` | **7** | Wrong fix + no-op edit loop | T4 edit → −1× payment; T7 edit → numerator==denominator (payment = 120); T9–T21 identical no-op edits ×13, T22 budget error, 315.4 s. Final patch: `return principal * r * (1 - (1 + r) ** -months) / (1 - (1 + r) ** -months)` |
| `heldout-loan-removed_return-0` | **7** | Right idea, wrongly applied (float regrouping) | T4 inlined return, deleted `factor`; T5 pytest `1066.1854641401 != 1066.1854641401003` (1 ulp); T7–T10 Decimal probes (T10 gives 1066.18546414010048…); T11 no-op comment deletion; T13 budget error, 337.6 s |

**Loan clustering (6/7):** mostly meaningful, with two discounts (Diagnosis). Loan fail
rate 6/14 (43%) vs non-loan 1/18 (6%); hypergeometric p ≈ 0.017 for ≥6 of 7 failures
being loan. But kill32 is loan-heavy by design (14/32 vs 25/116 overall), and all 14
loan tasks share one function, one failing assertion, one expected value — 14
near-identical items, not 14 independent domains. Structural difference verified in
`dev_tasks_heldout.jsonl`: loan is the only kill-32 domain with `**` exponentiation and
a 3-operand return formula sharing an intermediate (`factor`), and the only domain whose
failing test asserts exact equality on a 16-significant-digit float
(`1066.1854641401003`); every other domain's expected values are short/integer. The
loan formula invites numeric probing and punishes regrouping — it gives the repetition
attractor more opportunities. Isbn (the one non-loan failure) shows the same
"identified but didn't act" shape in text form, so the mode is not loan-specific.

**Taxonomy gap (Diagnosis):** §4 has no sub-code separating "budget exhausted after
productive work" from "budget exhausted in a repetition attractor", or for "one
generation consumed the budget". All four cat-8 tasks code identically despite three
distinct mechanisms. Recommend sub-codes 8a (repetition loop) and 8b (runaway
generation) before run 2.

**Evidence limitation (Diagnosis):** the thought channel is empty in all 334 turns
(`thought == ""`; `raw_text` is only the tool call, except isbn T11's ramble). For 6/7
failures "identified vs not" is behavioural inference, not observation — only the
discriminating evidence being in context from T1–T2 and behaviour being pure
re-observation is observed.

## 3. Ranked hypotheses for the Kaggle zeros (synthesis)

| Rank | Hypothesis | Evidence for | Evidence against | Confirm | Kill |
|---|---|---|---|---|---|
| **1** | **Repetition attractor / failure-to-edit (NEW)** — agent re-probes or re-emits no-op edits instead of editing; nudges structurally blind to it (they fire only on no-tool-call turns, HARNESS §5.3); budget dies with an empty/wrong tree | Present in 6/7 replica failures (3 pure probe loops, 2 no-op edit loops, 1 post-wrong-edit probe loop; isbn the same attractor in text form). All 4 panels found it independently. It is the *only* finding that transfers across the fidelity gap: it depends only on the prompt + nudge semantics, both verified scorer-faithful in the replica — not on serving, scale, or hidden tests. K2's prompt has no anti-repetition rule. | Observed on 7-line synthetic tasks where the bug is visible at T1; unproven that the same attractor fires on real repos (the agent may never reach the probe stage there). Single run, temp 1.0 noise. | Anti-repetition prompt arm fixes ≥4/7 failures in the replica (§7), then scores >0 on a Kaggle daily | Anti-repetition arm fixes 0/7 in the replica, or the loop never appears in a scale-probe trace |
| **2** | **Real-repo scale / orientation** — agent burns its 5 min finding the bug among ~2,700 files through truncated reads/outputs | Every scorer cap that never binds synthetically binds only at real scale: `read_file` 150 lines / 10,000 chars, `run_command` 5,000-char truncation, workspace layout capped at 200 BFS entries / maxdepth 3, context compaction at 14,336 tokens, no graph tools declared by K2. Toeda mean 326 s > our 300 s cap. | Replica cat-3 count is 0 — every failure oriented correctly at T1. The loop finding cuts both ways: the agent fails even when orientation is trivial, so scale is not *necessary* to produce a zero. Hidden repo sizes/bug types unverified (2,684–2,798-file figure from wave-10 notes, not verified against the hidden split). | Scale probe (§7): same bugs re-hosted in a large repo collapse to ~0 with §4 categories 3/7/8 dominating | Same probe resolves at a rate compatible with synthetic (≥~4/8, same failure distribution), or a Kaggle daily scores at Toeda-like rates with serving unchanged |
| **3** | **vLLM-specific serving/format failure** — tool-call format, thinking-channel handling, or sampling under vLLM's `gemma4` parsers fails where transformers/llama.cpp succeed | Toeda's parse failure is proven stack-dependent and 12B-specific on llama.cpp (25/104 tasks killed pre-fix). Replica cannot test vLLM at all (D1/D2/D3/D9). 13/13 zeros across all prompt arms is consistent with a serving-level floor. | Replica loop perfectly healthy under transformers (333/334 standard tier). E4B arms also scored 0.00 despite no 12B-style parse failure on llama.cpp — weakens *12B-specific* serving as a sole cause (not a vLLM-wide failure). No direct vLLM failure ever observed. | A vLLM replica with the scorer's exact flags reproduces category 1/2 dominance; or Day S > 0 *kills* it as sole cause (same vLLM serving delivered a patch) | vLLM replica shows a healthy loop and nonzero resolves; or **any** Kaggle daily scores > 0 with serving unchanged |
| **4** | **Task-distribution shift** — hidden bugs are multi-file, semantic, cross-cutting real PRs, not single-token mutations with the answer embedded in the test | Synthetic operators are 8 single-token mutations; hidden tasks are real SWE-bench-style PRs; replica "resolved" is an upper bound (D7). The guard-semantics finding (§1) hints the model mishandles anything beyond single-token value fixes. | Toeda scored 25.2% on *real* public tasks with the same minimal prompt — real-repo bugs are fixable by this loop under llama.cpp. Distribution alone (without serving/scale) is insufficient. | Real-repo-shaped tasks using *real* bug types collapse the rate while re-hosted synthetic operators do not | Re-hosted synthetic operators at real scale already collapse the rate (then it is scale, not bug type) |
| **5** | **Budget exhaustion as standalone cause (5 vs 6 min)** | Toeda mean 326 s > 300 s cap; 43.8% time-capped at 6 min | Toeda resolved 25.2% *despite* 57% cap-hit; +60 s cannot explain a ≥5× suppression (arithmetic bound below); replica resolves in ~50 s median; failures burn 300 s *looping*, not reasoning — more time buys more loops. **K3 is unanimously rejected by all four panels.** | K3 6-min daily scores > 0 while K2 at 5 min scored 0 | K3 scores 0 (or Day S at 5 min scores > 0) |
| 6 | **Gate false-negatives + revert** | — | **Exonerated as sole cause.** Cat 5 count 0 (structurally impossible for K2); K2 has no gate and still scored 0.00 on Kaggle; the 3 verification-failed tasks left patches in tree via the §8.1 fallback. Live only as a contributor for the earlier gated arms. | — | Already killed as sole cause |
| 7 | **Loop completion / never-submit** | Toeda: 66.3% never submit | Fallback (§8.1) scores the working-tree diff with or without `submit_patch`; K2 has no brake and still scored 0. The replica failure is never-*edit*, not merely never-submit — distinct from Toeda's pattern. | — | Already killed as sole cause |
| 8 | **Grader divergence / hidden-test strictness** | Phase-2 requires exit 0 + all required nodes passed unskipped | Wave-9 C5: 0 disagreements on 116 synthetic tasks. Cannot produce literal zero across 13 arms. | A Kaggle daily with a patch known to pass visible tests still scores 0 | Any Kaggle nonzero |

**Arithmetic bound (Researchers):** if the true Kaggle rate were Toeda's 25.2%,
P(0/58) ≈ 10⁻⁷; P(0/58) ≈ 5% at a true rate of 5%. The scorer-side interaction
crushes the rate from ~25% to <~5% — a ≥5× suppression. Any surviving hypothesis must
explain a large effect, which rules out small effects (prompt wording, 1-min budget,
single-run noise) as sole causes.

**Toeda comparison (Researchers):** Toeda 12B-min 25.2% (131/520) proves the 12B model +
minimal prompt + real harness loop can score on real-repo tasks under llama.cpp. The gap
to our 0.00 is attributable to (serving: llama.cpp → vLLM) and/or (tasks: public
real-repo → hidden real-repo) and/or (our 5-min cap) — it does not discriminate serving
from tasks, because both changed at once. Our replica vs the scorer differs on the same
two axes (transformers vs vLLM; synthetic vs real-repo), with different values on the
serving axis.

## 4. What is exonerated, locally (Diagnosis + Engineers)

- **Serving/format as sole cause, replica-locally:** 333/334 standard parse, 0 tasks
  in §4 categories 1, 2, 3, 4, 5, 9, 10. No `edit_file` ever returned `FileEditError`
  except the two budget-exhausted terminal calls. **Caveat (all panels):** replica ≠
  vLLM serving (D1/D2) — the scorer-side variant is untested, not disproved.
- **Orientation at synthetic scale:** cat 3 count 0; all failures read the implicated
  file at T1 and never touched a wrong file. Real-repo scale remains entirely untested.
- **Gate/revert:** cat 5 count 0; K2's Kaggle 0.00 cannot be gate/revert (K2 has no
  gate).
- **Latency as sole cause:** already dead (wave-9 C1); replica median resolved 45.8 s.
- **The fallback works:** the 3 verification-failed tasks have `patch_submitted=False`
  but `final_patch_files=1` — a non-empty diff was extracted by the §8.1 fallback and
  failed verification (wrong patch). The 4 budget-exhaustion tasks have 0 patch files:
  nothing was ever left to extract. The zeros are not "agent never submitted, diff
  lost" (Judges F5).

## 5. The replica's own defects (Engineers — all verified from code + JSON)

1. **`model.py:583` — no intra-turn deadline in `generate()`.** Demonstrated by isbn
   turn 11 (2,594.84 s for one 16,384-token generation). The existing wall-clock tests
   use a slow *fake* backend whose turns return quickly, so they exercise only the
   turn-boundary check and cannot catch this defect. DESIGN D5's bound ("a capped task
   can overshoot by at most one slow turn, ~145 s") is falsified by 16× (Diagnosis:
   17.9×). Run total 5,947.5 s, mean 185.9 s/task; capping isbn at 300 s gives
   ~3,553 s, mean ~111.0 s/task — **the reported mean is inflated 67%**, and any
   projection built on it is inflated the same way.
2. **`loop.py:277,325` — `would_have_timed_out` computed only at turn start.** All
   334 turns record `False`, including isbn turn 11. The flag is false-negative for
   exactly the case it exists to catch.
3. **`loop.py:316-325` — `TurnRecord` drops `GenerateResult.kv_fallback` and
   `thought_budget_forced`** (both exist in `model.py:88-89,645-646`, neither printed
   anywhere). D9/D10 cannot be audited from traces; whether the KV cache worked during
   the 6.3 tok/s runs is unknowable. (`self.kv_fallback` also latches cumulatively,
   never reset per turn.)
4. **`tools.py` `_edit_file_impl` — a no-op edit (old == new) returns `status: ok`
   with empty diff.** Scorer-faithful (ADK would do the same), so this is a *missing
   observability* defect, not a wrong return: the loop never flags repeated/no-op
   edits (13× in `swapped_operands-6`). Fix by reporting, not by changing the return.
5. **D3/D9 unexercised:** every tool turn's thought channel is empty
   (`<|channel>thought\n<channel|>`), so the thinking path was never observably
   engaged; `loop.py:383` appends model turns with `"turn_closed": False`, so no
   `<turn|>` separates a model turn from its inline `<|tool_response>` blocks
   (`model.py:168`) — plausibly correct, unverifiable without `gemma4_tool.jinja`.
   D2 is closed for this run (0 fallback calls in 334 turns; a `--strict-parse`
   re-run would be a no-op, verifiable offline by re-parsing stored `raw_text`).
   D8 threat is zero for this run (0 compaction events; max prompt 5,432 < 14,336)
   but unretired for real repos. D6 strengthened (safe direction): all 32 verifies
   ran, every non-empty patch applied on pass 1, zero R0 faults.

**Engineers' verdict on the headline, coordinator-endorsed:** "25/32 resolved, so the
loop works on synthetic tasks" is attackable from the published patches alone. The
defensible claim is narrower and still real: *the loop, as rendered here, fixes 17/32
simple synthetic bugs with gold-equivalent patches, 18/18 outside the loan domain, and
its failures on loan tasks are genuine agent loop-completion / float-exactness failures,
not serving or tooling failures.*

## 6. Lane / pick assessment (Judges — coordinator-endorsed)

Internal panel judgement, not a judge score. All unchanged by the replica result.

| Lane / pick | Score | Rationale |
|---|---|---|
| Main-track readiness | **2/10 — unchanged** | No Kaggle evidence moved. The one new datum (17/32 gold local) is from a package that already scored 0.00 publicly. Raising readiness on it would quote a replica rate as a Kaggle rate, barred by DESIGN §3 red line 4 and §9. |
| Small Model bonus (E4B) | **Dead as it stands — unchanged** | No E4B arm ran in the replica at all. Zero evidence for or against E4B v2. |
| Best New Resource ($10k) | **Only live lane — unchanged** | The replica adds a diagnostic appendix, not a new demonstrated asset. It neither strengthens nor weakens the resource case — but the honest 17/32 (not 25/32) must be what is disclosed. |
| Paper Track | **40/50 projected — unchanged, final** | Submitted; deadline passed 12 Nov. Replica is irrelevant. |
| Pick 1 — live scaffold 12B | **Unchanged** | No replica result alone changes picks (DESIGN §9/§10). Three independent reasons (Judges): (i) the ladder bar presupposes a scorer-faithful local measurement; the replica is explicitly disqualified as a pick-changing instrument; (ii) K2 is a *prompt* arm, governed by the public-score branch rule — and K2's public score is already known: 0.00; (iii) no head-to-head: Pick 1 was never run in the replica, so no ≥6 lead over Pick 1 is established. |
| Pick 2 — E4B v2 | **Unchanged, locked** | No E4B evidence in this run. |

**§6 continue bar:** satisfied on its face (25/32 ≥ 4/32, all 8 operators, 32/32
classified, zero R0 faults) — and still satisfied on the corrected 17/32. Judges'
caveat, endorsed: the bar's *intent* is to license a run that can discriminate the
next cause. The flaws (synthetic-only, transformers, isbn anomaly, D7 upper bound) do
not void the bar; they bound what run 2 may claim. **Run 2 is licensed, but only for
one question at a time** — see §7.

**Day S branches: unchanged by the replica** (Judges, unanimous). Day S > 0 → repeat
the identical Day S package (md5 `d3cfbe49`); a 1/58 score sits inside Toeda run-to-run
noise, so the repeat is a noise check, not a Pick 1 swap. Day S = 0 → hold dailies, no
K3. The replica *strengthens* the hold: K2 at 17/32 gold locally and 0.00 publicly is
direct evidence that a high replica count does not predict a Kaggle nonzero, so no
replica count can justify spending a daily slot.

**What a replica-directed daily must look like (Judges' test, coordinator-endorsed):**
(a) a defect reproduced in the replica traceable to a scorer-visible field
(prompt/config/tool declaration/budget — not serving stack or synthetic task shape);
(b) a single-variable fix to the submitted package; (c) a stated mechanism for why it
yields 0 on hidden tasks; (d) a replica re-run of the fixed package showing the
specific failure class fixed on the same 32. **K3 (6-min budget) fails this test on the
replica's own evidence:** resolved tasks take a median 46 s; failures burn 300 s
looping, not reasoning — more time buys more loops. The isbn task shows the cap isn't
even enforced within a turn.

## 7. Next steps — coordinator's sequencing (with costs)

**Step 0 — free, first, already scheduled: Day S score check, Thu 19 Nov ~09:59 GMT
(cron, 0 quota).** One binary bit with high leverage (Researchers): Day S > 0 kills
serving-as-sole-cause (same vLLM serving delivered a patch) and validates the skill
mechanism; Day S = 0 exonerates the skills defect and leaves the ranking in §3
standing. **If Day S > 0: branch (a) — repeat the identical Day S package; replica
work pauses.** Everything below applies only if Day S = 0.

**Step 1 — CPU-only, 0 quota, this week (do first; all panels' findings support
them):**
- (a) **Gold-equivalence re-grading** (Engineers #2): construct the reference fix per
  kill-32 task (mechanical single-operator undo) + a small held-out probe-test set per
  task (guard semantics), verify each gold fix passes and the empty patch fails, and
  re-grade attempt-7 patches as `gold_equivalent` vs `probe_weak`. Acceptance: 17 gold
  / 8 probe-weak with each classification citing the failing probe input. This turns
  the D7 judgement from operator-inference into measurement, and it is what a
  reviewer would demand first.
- (b) **Intra-generation deadline** (Engineers #1): deadline check inside
  `generate()`'s token loop (e.g. every 64 tokens), aborting with
  `finish_reason="length"` + `generation_aborted_by_deadline`. ~15 lines + 1
  regression test with a slow fake backend that stalls *inside* one `generate()`
  call. Acceptance: a re-run of the isbn task aborts within ~5 s of the 300 s
  deadline; existing 35 tests still pass. This is the only *verified defect* in the
  replica, and it protects the scarcest resource (one turn cost 43 min of a 30 h/week
  quota; a repeat on run 2 could eat ~15% of remaining quota).
- (c) **Repetition-streak reporting, detection only** (Engineers #3): flag
  `repeated_call_streak` / `repeated_noop_edit` in traces and attribution flags at
  ≥3 identical calls or an empty-diff edit. Acceptance: re-running attribution over
  attempt-7 JSONs flags all 3 loan runaways and the 13 no-op edits with zero false
  positives on resolved tasks. Do **not** add a loop-breaking intervention to the
  replica — the scorer has none, so intervening would make the replica *less*
  faithful (Engineers; Diagnosis's intervention proposal is recorded as a labelled
  ablation only — see dissent).
- (d) **Trace observability** (Engineers #4, trivial): record `kv_fallback`,
  `thought_budget_forced`, per-turn tokens/s in `TurnRecord`. Plus offline
  `--strict-parse` re-parse of stored `raw_text` (Engineers #5, CPU-only) to close
  D2 formally.

**Step 2 — the one recommended GPU step, only if Day S = 0: replica test of an
anti-repetition + edit-first prompt arm.** (Diagnosis #1; coordinator's pick over the
scale probe — see reasoning below.) Add to the K2 prompt (currently 22 words, none of
this): (i) never repeat an identical `run_command` — vary inputs or move on; (ii)
after a failing `pytest`, the next action must be `edit_file` (or `submit_patch` if
green); (iii) reuse existing intermediates (`factor`) — never rewrite an expression
into an algebraically-equivalent form; (iv) state the fix, then call `edit_file`
immediately — do not re-derive. **Pre-registered acceptance (before running):** max
run-length of identical calls ≤3 in every task; ≥4/7 failures resolve or reach a
*different* edit; 0 tasks end with 0 edits after a failing pytest. **Cost:** 7
failures only ≈ 7 × 300 s worst ≈ 35 min wall ≈ ~1.2 h quota; full frozen 32 ≈
3.3 h quota (V7 actuals: 5,947.5 s wall). Run the 7 failures first; extend to the
full 32 only if the 7 move. **If it passes, it earns the next Kaggle daily** — it is
the §9b exception to the stop rule (specific, fixable, scorer-faithful defect with a
predicted mechanism: on real repos the agent that re-probes instead of editing leaves
an empty tree → fallback finds nothing → 0). **If it fails, the loop is a symptom, not
a cause** — do not spend a daily on it, and move to step 3.
- *Why this before the scale probe (coordinator's reasoning):* the repetition
  attractor is the only finding that transfers across the fidelity gap (it depends
  only on prompt + nudge semantics, both scorer-faithful), and it is the only
  candidate that can *produce a daily*. The scale probe can only produce
  information. Temp 1.0 → 0.2 as a separate single-variable arm is also on the table
  (Diagnosis #3; the skill arm already uses 0.2) — do **not** stack it with the
  anti-repetition arm; test separately, and only after the anti-repetition result.
- *Cost basis:* kill32 ran 5,947.5 s wall (1.65 h) ≈ ~3.3 h quota at 2× burn on L4x4.
  Remaining quota ~22 h (estimate — live meter not re-verified; stale mid-run reading
  04:37/30 h). Read the live meter before any GPU run.

**Step 3 — Day S replica (Judges' conditional), only if Day S = 0.** Same frozen 32,
Day S fixed-skill package. **Single question:** does the model invoke `load_skill` /
`run_skill_script` in the replica, and does that change anything? Note the design
doc's "single-variable change" description is inaccurate (Judges): Day S confounds
five variables (skill mechanism + gate/revert + temp 0.2 + 9 tools + manual fallback)
— run 2 can answer the binary skill-invocation question; it cannot attribute any
count change to the skill alone. Cost ≈ ~4 h quota (V7 actuals + load/setup).
**If Day S > 0, do not run it** — the Kaggle repeat takes precedence.

**Step 4 — 8-task scale probe (Researchers' design), only if steps 2–3 fail to move
anything.** Re-host 8 frozen tasks (one per operator, including
`heldout-loan-off_by_one-0` as the known flip control) inside a real-repo-shaped
workspace (real public snapshot ~2,700 files, `task.py` deep in the package, identical
single-token bug), same K2 package and budgets. Pre-register: collapse to ≤1/8 with
§4 categories 3/7/8 dominating confirms scale/orientation; ≥4/8 with the synthetic
failure distribution kills it as a sole cause. Cost ≈ 1–2 h quota (8 tasks pro-rata
from kill32 wall time + load/setup). Stop on the first runaway turn. Full 32-task
scale probe (~3.3 h quota) only if the 8-task result is ambiguous (2–3/8). **Do not
do:** 116-task replica (~16–20 h quota, exceeds the remaining budget), K3 daily,
any Pick 1/Pick 2 change on replica evidence.

**Resource lane (parallel, 0 quota):** Judges' #1 — the only live lane. Disclose the
replica result in its honest form (17/32 gold-equivalent on synthetic vs 0.00 public)
as a diagnostic, never as a score claim. This runs in parallel with steps 1–4; it is
not either/or with the replica, since it costs no quota.

**Explicitly not recommended:** vLLM backend (infeasible in the notebook — TP=4
impossible, vLLM not installed); reducing `max_output_tokens` (breaks K2 config
fidelity); a replica-side repetition breaker as a *fix* (breaks scorer fidelity —
labelled ablation only); K3 6-min budget (saves 0/7, envelope risk 120 × 6 min =
exactly 12 h).

## 8. Preserved dissent

- **Judges vs the programme (steelman, substantially conceded):** 14 days remain; picks
  are settled insurance; the replica cannot replicate hidden tasks or the scorer. A
  full 116-task run would exceed the remaining budget. The coordinator concedes the
  score-prediction and full-run variants, as wave 11 did — but not the gated pilot:
  the repetition attractor is a mechanism finding the score alone could never have
  produced, and it is testable for ~1.2 h quota. If steps 2–3 both fail to move
  anything, the steelman wins outright and the programme stops.
- **Researchers vs Judges on the next GPU step:** Researchers rank the 8-task scale
  probe as the single recommended GPU experiment; Judges rank "do not do" the scale
  probe and put the Resource lane first with run 2 (Day S replica) as the only
  conditional replica step. Coordinator's ruling: anti-repetition test first (§7
  step 2), for the transferability reason given there; scale probe demoted to step 4.
  If the anti-repetition test passes and earns a daily, the scale probe's question
  becomes moot for the finals; if it fails, the scale probe is next.
- **Diagnosis vs Engineers on the repetition breaker:** Diagnosis proposes a
  replica-side forced message on ≥3 identical calls as a diagnostic probe of
  causality; Engineers oppose any loop-breaking intervention in the replica because
  the scorer has none — it would make the replica less faithful. Coordinator's
  ruling: detection/reporting only (step 1c); an intervention is admissible solely
  as a *labelled ablation* whose results are never quoted as a replica score, and
  only after the prompt-side test (step 2) — the prompt fix is the shippable
  version of the same idea and must be tested first.
- **Diagnosis vs Judges on run-2 priority:** Diagnosis's run 2 is the
  anti-repetition arm; Judges' run 2 is the Day S replica. Coordinator orders
  anti-repetition first (step 2) because it can produce a daily; Day S replica is
  step 3. Both are conditional on Day S = 0.
- **Engineers vs the headline:** the 25/32 figure was quoted by three panels without
  the D7 correction. Coordinator verified the correction from the patches and adopts
  17/32 gold-equivalent as the headline, with 25/32 as-graded retained alongside.
  The other three panels did not contest the correction (they did not check the
  patches); it is a coordinator verification, not a panel dispute.
- **All panels agree:** Day S branch rules unchanged; Picks unchanged; lanes
  unchanged; K3 rejected; no replica result alone changes Pick 1/Pick 2.

## 9. What Olly needs to hear (Judges §5, coordinator-endorsed, with the D7 correction)

*Your agent can fix these toy bugs — 17 out of 32 properly, 25 out of 32 if you count
patches that only pass because the visible tests never check the guard. The same
package scored zero on Kaggle. So the problem was never "the model can't fix
anything". It's that the real hidden tasks, or the real scorer, differ from this toy
setup in a way we still haven't measured — and this replica, by design, can't measure
it. The one new, actionable thing the replica did find: on 6 of the 7 tasks it failed,
the agent got stuck repeating the same probe or the same no-op edit until its time
ran out, instead of editing — and nothing in the harness stops that, on Kaggle
either. That's testable, and it's the next step if Day S scores zero tomorrow.*

**What it does NOT prove:** that the scorer is broken; that Pick 1 is wrong; that any
prompt tweak, budget change, or skill fix will score >0; that 17/32 or 25/32 means
anything like 0.5+ on the public split (1 task ≈ 0.017 on 58 tasks; a replica rate is
not a Kaggle rate); or that E4B would do better (untested).

## 10. Files and evidence index

- Report (this file): `kaggle/wave11-attribution-2026-11-18.md`
- Run data: `kaggle/shadow-scorer/runs/attempt7/k2-kill32-summary.json`
  (334 turns), `k2-probe4-summary.json` (46 turns), `notebook-v3.ipynb`
- Design/pre-registration: `kaggle/shadow-scorer-DESIGN.md` (§4 taxonomy, §6 bar,
  §9 branches, §10 steelman); `kaggle/HARNESS_README.md` (§5.3 nudges, §8.1
  fallback, §8.2 grading); `submission/DAILY_LADDER.md` (ladder bars, finals
  trajectory); `kaggle/toeda-replication-pack.md` (Toeda numbers)
- Package that ran: `submission-k2/` (22-word prompt, temp 1.0, 6 tools)
- Panels run: Diagnosis, Researchers, Principal Software Engineers, Judges — all
  four independent, all completed 18 Nov. UI/UX, user panel, videographers not
  applicable to a diagnostic replica run (no user-facing artifact); stated, not
  omitted silently.
- Nothing submitted, no final selected, no GPU spent by this council wave.
  Analysis only.
