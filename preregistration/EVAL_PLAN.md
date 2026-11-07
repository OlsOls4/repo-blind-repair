# Dev-set evaluation plan — Gemma 4 Repo-Blind Repair

## What this is for

`eval_dev.py` is the quality measurement the Paper Track needs. The
Paper draft currently marks these as **TO BE MEASURED** because no
full model, dev-set run, or leaderboard result exists yet:

- fix rates (overall, with sample size, and by bug class where known)
- synthetic-to-real gap (how far clean-room training transfers)
- gap vs a real-fix baseline (see below — read the baseline note)
- failure audit (why the unfixed tasks failed, per task)
- composite-mutation gap question, full-run sizing, all dev-set figures

A dev-set run of `eval_dev.py` on the retrained 20k adapter fills the
first, second, and fourth items directly from its JSON report. It does
not fill leaderboard outcomes — only a submission does that.

## Baseline note (honest scope)

The paper must not claim a comparison against a real-fix-trained
baseline that does not exist. Repo-Blind Repair's standing ban on
real-fix training data means no such baseline will be trained here.
The honest comparison available is the **inference-only / base-model
agent on the same dev tasks** (`harness/inference_agent.py`, no
adapter): adapter fix rate vs base-model fix rate, on identical
tasks, gates, and budgets. Report both numbers side by side and
label them as such; never present the base-model agent as a
"real-fix baseline".

## What a good score does NOT prove

Scorer vLLM serving is still **UNPROVEN**. Gate 2 finished
INCONCLUSIVE: Stage 1 PASS, Stage 2 PASS on reproduced discovery
only (real starter discovery not importable), Stage 3 UNTESTED
(vLLM is not installed in the Kaggle notebook environment). This
evaluation runs the adapter under transformers with manual-LoRA
hooks — it proves the adapter *can repair* under that path, not that
the scorer's vLLM path can load it. A good dev-set score must never
be quoted as proof of serving. Serving is proven only by a real
submission or the starter's own validation environment. The same
applies to the still-unverified `agent.yaml` schema and the unknown
12B `hf-overrides` value.

## Stop-sequence probe rule (Gate 2)

**Decision rule (pre-registered 5 Nov 2026):** adopt `"\nFixed code:"`
as a stop sequence for the **LoRA variant only**, and only if a probe
shows it does not truncate legitimate agent output. Never adopt it
for the inference-only variant: the looping it would cut is a
trained-adapter artefact (v18 never taught EOS after the fixed
file), not a property of the base model, and a stop sequence tuned
to our local format may truncate legitimate agent output. The
sampling configs in `submission/` and `submission-minimal/` record
the same deferral.

**Why it is only a safety net:** v19 training appends EOS, so a
correctly trained adapter should stop on its own. The stop sequence
exists for the failure mode where the model re-emits the
`"Fixed code:"` cue and loops to the token cap. If the probe fails,
do not adopt it — rely on EOS plus the cue-truncation safety net in
local eval instead.

**Probe procedure** (runs inside the Gate 2 vLLM probe session —
see `kaggle/FIRST_GPU_RUNBOOK.md`, Step 4; `eval_dev.py` has no
stop-sequence flag, so this is a vLLM-level test, not an
`eval_dev.py` test):

1. Serve the 12B base + Stage 1 adapter under the scorer's
   settings (see the runbook), on the same small task set twice:
   once with `stop_sequences: ["\nFixed code:"]` in the generation
   config, once without. Same tasks, same sampling otherwise.
   Use 2–3 frozen tasks, or the 4-task timing-probe set.
2. **Adopt only if ALL of these hold:**
   - No output that was complete and parsable without the stop
     becomes incomplete with it (no file cut mid-way, no missing
     trailing lines, no newly unparsable output).
   - No gate-accepted fix is lost between the two runs.
   - The stop does not fire on a `"\nFixed code:"` occurring
     *inside* a legitimate file (docstring, comment, string
     literal) before the true end of the output.
   - The agent's tool-call output is not cut short.
   - The looping outputs terminate at the re-emitted cue instead
     of running to the token cap (stop reason is the stop
     sequence, not length, in the generation log).
3. **Record both runs** (outputs, stop reasons, parsable rates) in
   the saved version output. If adopted, the stop goes in the LoRA
   package's `generate_content_config` (`stop_sequences`) and is
   recorded in `submission/DAILY_LADDER.md` as a serving-config
   detail of the LoRA variant — it ships with the LoRA variant's
   first daily, not as a separate stacked change.

## How to run it on Kaggle

1. Attach as notebook inputs: the model, `eval_dev.py`, the
   dev-tasks JSONL file (format documented in the script header —
   mismatches fail loudly, never guessed), and the **saved** adapter
   from the committed retrain version's output.
2. Install the loguru + compressed_tensors wheels offline first
   (loguru first, `--no-deps --quiet --no-index`), same as the pilot.
3. Set `MODEL_PATH` to the attached local model path and pass
   `--adapter-dir` pointing at the saved adapter dir.
4. Dry-run first: `--dry-run` validates arguments and the task file
   and prints the planned task count without loading a model.
5. Pilot slice first: `--max-tasks` on a small slice, then size the
   full 129-task run from measured per-task times (default cap is
   600 s/task; a full run at worst case exceeds one session).
6. Run with internet OFF. **Save the notebook version with output
   immediately afterwards.** The 31 Oct 20k adapter was LOST because
   it lived only in a non-persistent interactive working directory;
   a report or adapter that exists only in a working directory does
   not exist.

## Reading the report

- `fix_rate` = gate-accepted fixes / tasks evaluated, always with
  `n_tasks` and the Wilson 95% interval beside it.
- `by_bug_class` is only as fine as the task file's labels; dev-set
  bug classes are hidden real bugs, so `unknown` will dominate
  unless labels are supplied.
- `regression_rate` = candidates that fixed the failing test but
  broke a regression test, divided by candidates that fixed the
  failing test. Accepted patches have zero regressions *by
  construction* (the gate requires it), so 0% there would be vacuous.
- `status` values: `fixed`, `no_patch` (model output unparsable at
  every rung), `gate_rejected`, `error`, `timeout`,
  `no_regression_tests` (gate can never accept without regression
  tests — recorded, not silently skipped).
- Candidates are adapter model output ONLY. Deterministic heuristic
  flips are deliberately disabled, so the fix rate is the adapter's,
  not the harness's.

## What is not claimed

No dev-set score exists yet, so none is stated here. No quality,
accuracy, or leaderboard claim follows from training loss
(8.5253 → 0.0245 on the lost 20k run was training loss only) or from
pilot timing. This plan will be updated with real figures only after
a dev-set run completes and its report is saved.
