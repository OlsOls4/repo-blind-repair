# First GPU session runbook — after the weekly quota reset

For the reset day only. Quota is **0/30 h, no reset date shown** (5 Nov 2026);
nothing in this file can run until the reset lands. CPU-only until then.
Order: **baseline → Stage 1 retrain (epoch 1) → kill pilot → Gate 2 vLLM probe**.
Each step ends with **Save Version with output** — Kaggle notebook
persistence is "No persistence": stopping a session wipes
`/kaggle/working`, and the 31 Oct adapter was lost exactly that way.

## Rules that bind the whole session

- **One interactive GPU session at a time.** Quota burns on session
  lifetime at 2× on L4x4, not just cell time — keep sessions short
  and purposeful, and close the session when a step is done.
- **Internet OFF** in the competition notebook. Never use GCP services
  from a Kaggle notebook; never buy GPU.
- **Read the quota meter first** (Settings) and pass
  `--quota-used-seconds` from the meter before EVERY training chunk —
  the script cannot query the meter.
- **Never run epoch 2 in the same session as epoch 1.** Stage 2 is a
  separate resume, only if the kill pilot does not kill the run.
- **Do not renegotiate the kill thresholds after seeing results.**
  They are pre-registered in `kaggle/KILL_TEST_READINESS.md` and
  repeated below.

## Quota budget (measured figures where they exist)

| Step | Wall time (measured / estimated) | Quota cost at 2× burn |
|---|---|---|
| Baseline eval, n=32 (`--no-adapter`) | UNMEASURED — size from the first 4 tasks, then extrapolate ×8 | load + 32 × per-task time, ×2 |
| Stage 1 retrain, 5,000 optimiser steps | 3.78 h stepping + 60 s load (dry-run sizing at 0.339 s/micro-step) | ~7.6 h |
| Kill pilot, n=32 (with adapter) | UNMEASURED — same sizing rule as the baseline | load + 32 × per-task time, ×2 |
| Gate 2 vLLM probe | UNMEASURED — budget 1–2 h wall, install included | ~2–4 h |
| Full 10,000-step run (for reference only) | ~9 h wall at the measured pace | ~18 h |

**Stop condition before starting:** do NOT start Stage 1 unless the
meter shows enough quota for baseline + Stage 1 + kill pilot + 1 h
margin. If quota < that, run the baseline only (it is the cheapest
step and its numbers stand alone), then stop. If quota runs out
mid-step, save the version with whatever output exists and stop —
do not start a second session to finish a chunk the meter cannot cover.

## Step 0 — session setup (CPU-speed, no GPU work yet)

1. Start ONE interactive GPU session. Confirm the accelerator and
   read the quota meter; record both in the notebook's first cell
   output.
2. Attach as notebook inputs: the 12B model, `full_train_manual_lora_v19prep.py`,
   `eval_dev.py`, `harness/`, `kaggle/dev_tasks_heldout.jsonl`,
   `kaggle/train_records_20k.jsonl` (verify sha256 `1786411c…`), and
   `kaggle/kill_test_task_ids.txt`.
3. Confirm input paths by listing `/kaggle/input/` — the private
   dataset slugs are not recorded locally, so do not guess them.
   The model path that works:
   `/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2`
4. Install the wheels offline first, loguru first
   (`--no-deps --quiet --no-index`), same as the pilot. Wheels live in
   `kaggle/pilot/wheels/` (loguru 0.7.3 + compressed_tensors 0.20.0).
5. Copy the scripts and task files into `/kaggle/working/` and run
   everything from there.

## Step 1 — baseline: base model on the frozen n=32

Why first: the kill pilot needs a base-model comparison on identical
tasks, gates, and budgets, and the baseline is the cheapest GPU step —
if quota is short, this is the step that still runs.

Frozen set: `kaggle/kill_test_task_ids.txt` — 32 stratified held-out
singles, 4 per operator, first 4 per operator in file order from
`kaggle/dev_tasks_heldout.jsonl` (see `kaggle/KILL_TEST_READINESS.md`).

```bash
export MODEL_PATH=/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2

# Dry run first (no model load): validates arguments + task file.
python3 eval_dev.py --dev-tasks dev_tasks_heldout.jsonl --no-adapter \
  --task-ids "$(grep -v '^#' kill_test_task_ids.txt | paste -sd,)" \
  --dry-run

# Timing probe: first 4 tasks only; record per-task seconds, then
# extrapolate ×8 for the full 32 before committing quota.
python3 eval_dev.py --dev-tasks dev_tasks_heldout.jsonl --no-adapter \
  --task-ids heldout-loan-off_by_one-0,heldout-loan-off_by_one-1,heldout-loan-off_by_one-2,heldout-circle-off_by_one-1 \
  --out baseline_probe4.json

# Full baseline on the frozen 32.
python3 eval_dev.py --dev-tasks dev_tasks_heldout.jsonl --no-adapter \
  --task-ids "$(grep -v '^#' kill_test_task_ids.txt | paste -sd,)" \
  --out baseline_n32.json
```

**Save Version with output immediately.** Record in the report:
fix rate, Wilson 95% interval, parsable-non-identical rate, and the
measured per-task seconds (these size the kill pilot).

## Step 2 — Stage 1 retrain: epoch 1 only, checkpoint-opt-5000

Exactly two changes vs v18: EOS appended in tokenisation, bug-class
label dropped. Held constant: r=8, alpha=16, v18 20,000 records.
Prompt-loss masking is the pre-registered SECOND experiment — do NOT
add it here.

```bash
python3 full_train_manual_lora_v19prep.py \
  --dataset train_records_20k.jsonl \
  --epochs 2 --max-chunk-optimizer-steps 5000 \
  --output-dir full_train_output \
  --weekly-quota-seconds <SECONDS_FROM_METER> \
  --quota-used-seconds <SECONDS_FROM_METER>
```

The script stops exactly at the epoch-1 boundary with
`full_train_output/run-<id>/checkpoint-opt-5000/` (adapter_config.json
+ adapter_model.safetensors, ~62.6 MiB). Verify the checkpoint files
exist and the reload check passed in the log.

**Save Version with output immediately, then STOP. Do not continue
to epoch 2 in the same session.**

## Step 3 — kill pilot: the v19 adapter on the frozen n=32

Same 32 task IDs, same gates, same budgets as the baseline. The
adapter directory is the Stage 1 checkpoint from Step 2.

```bash
python3 eval_dev.py --dev-tasks dev_tasks_heldout.jsonl \
  --adapter-dir full_train_output/run-<id>/checkpoint-opt-5000 \
  --task-ids "$(grep -v '^#' kill_test_task_ids.txt | paste -sd,)" \
  --out kill_pilot_n32.json
```

**Save Version with output immediately.** Then apply the
pre-registered rule — no renegotiation:

- **KILL the LoRA route** if fewer than **4/32** fixed, OR fixes fall
  in **≤1 operator**, OR parsable (syntactically valid Python)
  non-identical output is **<30%**. If killed: do NOT run epoch 2.
  Record the pilot report and stop the GPU programme; the main-track
  picks stay inference-only.
- **CONTINUE** at 4–9/32 with ≥3 operators.
- **STRONG** at ≥10/32 — proceed with confidence.
- **EOS falsifier:** if >50% of post-EOS generations still run to the
  token cap without EOS, the implementation failed — debug
  tokenisation, do not retrain again.
- **Diagnostic arbitration:** garbage output → training/format
  failure (kill if it persists post-EOS); parsable-but-wrong → data
  diversity is the next lever (v19 data run: composites 6,000→3,000,
  singles 14,000→17,000, ≥30% fresh template families — only if the
  pilot passes).

Between KILL and STRONG the decision is the operator's, on the pilot
evidence — but epoch 2 must still only ever continue from
checkpoint-opt-5000, never from scratch.

## Step 4 — Gate 2 vLLM probe (only if the pilot does not kill)

Proves the surviving adapter loads and generates under the scorer's
vLLM path — the one thing the local eval cannot prove. Stage 1 PASS
and Stage 2 PASS-on-reproduced-discovery are already banked; this
step is Stage 3.

1. Check whether `vllm` is importable in the session. If it is not,
   attempt the install once; if it cannot be installed in-session,
   STOP and record that — do not burn quota retrying.
2. Serve the 12B base with the Stage 1 adapter under the scorer's
   settings: tensor_parallel_size 4, gpu_memory_utilization 0.90,
   max_model_len 32768, dynamic LoRA, with hf-overrides
   `{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}`.
3. Generate on 2–3 frozen tasks and confirm the adapter attaches
   (328/328 target modules) and produces a candidate file.
4. Run the stop-sequence probe from `kaggle/EVAL_PLAN.md` in the
   same session (adopt `"\nFixed code:"` for the LoRA variant only
   if the probe shows no truncation of legitimate output).

**Save Version with output immediately.** Record pass/fail per stage.
A pass here plus a passing kill pilot is what makes a LoRA daily
legal; a fail here means the LoRA stays local-only no matter how
good the kill pilot looks.

## If the pilot does not kill — Stage 2 (separate session)

Resume ONLY from the saved checkpoint, with otherwise IDENTICAL
arguments (same dataset file, seed, LoRA, LR, accumulation):

```bash
python3 full_train_manual_lora_v19prep.py \
  --dataset train_records_20k.jsonl \
  --epochs 2 --resume-from full_train_output/run-<id>/checkpoint-opt-5000 \
  --output-dir full_train_output \
  --weekly-quota-seconds <SECONDS_FROM_METER> \
  --quota-used-seconds <SECONDS_FROM_METER>
```

The prompt_format guard aborts any resume onto a checkpoint this
script did not write (including all v18 checkpoints). Save Version
with output immediately afterwards.

## After the 12B sequence

- A surviving 12B LoRA enters the daily ladder as the next
  challenger daily (one variable: adapter added, prompt unchanged —
  see `submission/DAILY_LADDER.md`).
- The E4B LoRA sequence follows `kaggle/E4B_PLAN.md` (Gate 1 →
  epoch 1 → same kill pilot); it is a separate programme, not part
  of this first session.
- A LoRA daily is legal only with kill-pilot + Gate 2 evidence from
  versioned saved output; never ship a dummy adapter.
