"""Gemma 4 FULL training run, MANUAL-LoRA variant -- v19prep (post-quota-reset retrain).

STATUS: NOT EXECUTED. Prepared 1 Nov 2026 for the FIRST run after the
weekly Kaggle GPU quota reset. It is a copy of the PROVEN v18 script
(full_train_manual_lora.py -- the committed retrain that completed
10,000/10,000 optimiser steps, 2 epochs, on the v18 20,000-record
dataset, training loss 8.5253 -> 0.0191) with EXACTLY TWO behaviour
changes (below) plus a staging plan and a prompt-format resume guard.
The v18 script is UNTOUCHED: do not edit it to match this file, and
do not run this file against a v18 checkpoint (the guard aborts).

WHY v19 EXISTS (diagnosed 1 Nov 2026, held-out synthetic pilots 0/18):
  * v18 never taught the model an end-of-sequence after the fixed
    file, so at inference it runs to the 512-token cap and loops
    into repeated "Fixed code:" blocks of corrupted fragments.
  * v18 training prompts carried a bug-class label header that
    eval never supplies, a train/eval prompt mismatch.
Both are fixed here, and ONLY these two things change.

CHANGE 1 -- EOS appended (tokenisation only, never literal text):
  tokenize_training_example() takes the tokenizer's own ids for the
  training text (its automatic BOS handling is unchanged), asserts
  tokenizer.eos_token_id is not None, asserts the last id is NOT
  already EOS (a tokenizer configured to append EOS would otherwise
  teach a doubled EOS), appends [eos_id] as a TOKEN ID, and sets
  labels to the full id list. The manual shifted cross-entropy loss
  therefore trains the model to predict EOS immediately after the
  final fixed-code token. The EOS token string is NEVER concatenated
  onto the text. eos_token_id is logged once at startup, after the
  tokenizer loads. The token-length survey counts the appended EOS
  (+1 per record).

CHANGE 2 -- bug-class label dropped (prompt text only):
  to_text_example() now returns EXACTLY:
      Fix this Python bug:\n<buggy_code>\nFixed code:\n<fixed_code>
  which is the eval prompt (harness/patcher_stub.py
  build_trained_format_prompt) plus the fixed code. The v18 header
  "Fix this Python bug (bug class: <label>):" is gone: real tasks
  carry no label, and inventing one at inference would be a
  fabricated input. Records themselves are unchanged -- the v18
  20,000-record file is used exactly (same file, SHA-256-checked
  on resume); only the rendered prompt drops the label field.

EVERYTHING ELSE IS IDENTICAL to the proven v18 script (do not
"improve" these without a new pilot proving the change):
  * Declared base ONLY: google/gemma-4-12b-it-qat-w4a16-ct,
    dequantized load (run_compressed=False), MODEL_PATH env
    override, bf16, device_map="auto", gradient checkpointing
    OFF by default.
  * Manual LoRA in pure torch via register_forward_hook delta
    injection, r=8, alpha=16, scaling 2.0, AdamW lr=2e-4 over
    factors ONLY, all base parameters frozen.
  * Manual-logits path ONLY (body -> hidden states -> F.linear
    with the frozen lm_head weight -> manual shifted CE loss).
    The model's own lm_head/loss path is NEVER called.
  * PEFT-compatible adapter save format, reload check after
    every checkpoint. NO PEFT import and NO torchao import.
  * Effective batch B_eff = 4 via accumulation (micro-batch 1 x 4),
    quota-chunked resumable checkpoints (every 100 steps, every
    15 minutes, every epoch end, chunk end / run end), loud-abort
    resume validation. Quota: a full run needs ~9 h at the
    measured pace, so run ONLY after the weekly reset, and update
    --quota-used-seconds from the Kaggle quota meter before
    EVERY chunk (the script cannot query the meter; internet off).

STAGING -- MANDATORY, epoch 1 first (the kill-pilot gate lives
OUTSIDE this script, in eval_dev.py on held-out synthetic tasks):
  Stage 1: run with --epochs 2 --max-chunk-optimizer-steps 5000
    (plus the dataset, output-dir, and quota arguments). This
    stops exactly at the epoch-1 boundary with checkpoint
    checkpoint-opt-5000 (~5,000 optimiser steps). Save Version
    on /kaggle/working immediately, then STOP -- do not continue
    to epoch 2 in the same session.
  Kill pilot (separate eval session, n=32 stratified held-out
    singles, one task per operator class where available):
    KILL the run if <4/32 fixed, or fixes fall in <=1 operator
    class, or parsable non-identical output is <30% of tasks.
    STRONG (proceed with confidence) if >=10/32 fixed.
    Between KILL and STRONG, the decision is the operator's, on
    the pilot evidence -- but epoch 2 must still only ever
    continue from checkpoint-opt-5000, never from scratch.
  Stage 2 (ONLY if the pilot does not kill the run): resume with
    --resume-from <run-dir>/checkpoint-opt-5000 --epochs 2 and
    otherwise IDENTICAL arguments (same dataset file, seed,
    LoRA, LR, accumulation). The prompt_format guard below
    aborts any resume onto a checkpoint this script did not
    write (including all v18 checkpoints).

CUE-TRUNCATION IS A SAFETY NET, NOT THE MECHANISM: eval truncates
generation at a bare "Fixed code:" cue line so a looping model
still yields a candidate file. v19 must LEARN EOS; if the v19
epoch-1 pilot still loops to the token cap, that is itself a
kill-pilot signal, not something to paper over with truncation.

DELIBERATELY NOT IN THIS VARIANT: prompt-loss masking (masking
prompt tokens from the loss so only fixed-code tokens train) is
a PRE-REGISTERED SECOND experiment, held back so the EOS-append
+ label-drop effect is measured alone. Do NOT add it here; if
v19's kill pilot passes, masking is the next experiment, not a
mid-run patch to this one.

WHAT IS UNTESTED until a GPU session: everything that needs the
model -- the EOS append against the real Gemma tokenizer (the
assert fires if the tokenizer already appends EOS or has no EOS
id), the epoch-1 checkpoint + kill pilot + epoch-2 resume
sequence, and all quality claims. --dry-run validates arguments,
dataset, sizing, chunk budget, and resume metadata only.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# --- Declared base and pilot-faithful defaults (do not change silently)
MODEL_HANDLE = "google/gemma-4-12b-it-qat-w4a16-ct"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
EXPECTED_TARGET_COUNT = 328  # Gate 1 measurement; warning only, not a gate.

# --- Full-run defaults (all overridable on the command line)
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_EPOCHS = 2
DEFAULT_ACCUMULATION_STEPS = 4  # micro-batch 1 x 4 = B_eff 4
DEFAULT_MICRO_BATCH = 1  # fixed: the proven pilot micro-batch; not a flag
DEFAULT_CHECKPOINT_EVERY_STEPS = 100
DEFAULT_CHECKPOINT_EVERY_SECONDS = 15 * 60  # 15 minutes
DEFAULT_WARMUP_OPTIMIZER_STEPS = 10
DEFAULT_MEASURED_SECONDS_PER_MICRO_STEP = 0.339  # pilot median, steps 4-20
DEFAULT_LOAD_OVERHEAD_SECONDS = 60.0  # pilot session allowance
DEFAULT_WEEKLY_QUOTA_SECONDS = 30 * 3600  # 30:00 hrs
DEFAULT_QUOTA_USED_SECONDS = 3 * 3600 + 15 * 60  # 03:15, 31 Oct figure
DEFAULT_CHUNK_FRACTION = 0.60
MAX_CHUNK_FRACTION = 0.70  # hard cap: a chunk must not exceed 70% remaining
FORMAT_VERSION = 1
# v19 prompt format marker, stored in training_state.json and
# checked on resume: a v19 run must never resume a v18
# checkpoint (different prompt text AND different tokenisation),
# and a v18 run must never resume a v19 checkpoint.
PROMPT_FORMAT_V19 = "v19-eos-appended-no-bug-class-label"


# ---------------------------------------------------------------- sizing

def optimiser_steps_for(num_records: int, epochs: int,
                        accumulation: int) -> int:
    """Optimiser steps for a run: epochs * ceil(N / accumulation).

    Grouping is per epoch (a partial group at an epoch end still
    steps the optimiser), so the formula is per-epoch, not a
    single ceil over the whole run.
    """
    if num_records <= 0 or epochs <= 0 or accumulation <= 0:
        raise ValueError("N, epochs, and accumulation must all be > 0")
    return epochs * math.ceil(num_records / accumulation)


def compute_sizing(num_records: int, epochs: int, accumulation: int,
                   seconds_per_micro_step: float,
                   load_overhead_seconds: float) -> dict:
    """Pure sizing arithmetic, reproducible from the stated inputs.

    stepping_time uses optimiser_steps * accumulation *
    seconds_per_micro_step, which is exact when N is divisible by
    accumulation and a slight over-estimate otherwise (safe).
    """
    opt_steps = optimiser_steps_for(num_records, epochs, accumulation)
    stepping = opt_steps * accumulation * seconds_per_micro_step
    return {
        "num_records": num_records,
        "epochs": epochs,
        "accumulation_steps": accumulation,
        "effective_batch": accumulation * DEFAULT_MICRO_BATCH,
        "optimizer_steps": opt_steps,
        "micro_steps_upper": opt_steps * accumulation,
        "stepping_seconds": stepping,
        "load_overhead_seconds": load_overhead_seconds,
        "total_seconds": stepping + load_overhead_seconds,
    }


def compute_chunk_budget(weekly_quota_seconds: float,
                         quota_used_seconds: float,
                         chunk_fraction: float,
                         chunk_budget_override: float | None) -> dict:
    """Chunk wall-clock budget from the quota. Aborts on bad inputs.

    A chunk budget above 70% of remaining quota is refused even
    when explicitly overridden: the 60-70% band is a guard, not
    a suggestion. The caller must supply a CURRENT quota-used
    figure; this function cannot query the Kaggle meter.
    """
    if weekly_quota_seconds <= 0:
        raise ValueError("--weekly-quota-seconds must be > 0")
    if quota_used_seconds < 0 or quota_used_seconds >= weekly_quota_seconds:
        raise ValueError(
            f"--quota-used-seconds ({quota_used_seconds}) must be >= 0 "
            f"and < --weekly-quota-seconds ({weekly_quota_seconds}); "
            f"if the quota is exhausted, do not start a chunk.")
    if not (0 < chunk_fraction <= MAX_CHUNK_FRACTION):
        raise ValueError(
            f"--chunk-fraction ({chunk_fraction}) must be in "
            f"(0, {MAX_CHUNK_FRACTION}]: a chunk must not exceed "
            f"{MAX_CHUNK_FRACTION:.0%} of remaining quota.")
    remaining = weekly_quota_seconds - quota_used_seconds
    default_budget = remaining * chunk_fraction
    budget = default_budget if chunk_budget_override is None \
        else chunk_budget_override
    if budget <= 0:
        raise ValueError("--chunk-budget-seconds must be > 0")
    if budget > remaining * MAX_CHUNK_FRACTION + 1e-9:
        raise ValueError(
            f"chunk budget {budget:.0f}s exceeds "
            f"{MAX_CHUNK_FRACTION:.0%} of remaining quota "
            f"({remaining * MAX_CHUNK_FRACTION:.0f}s); refusing to "
            f"let one chunk eat the quota.")
    return {
        "remaining_seconds": remaining,
        "chunk_fraction": chunk_fraction,
        "chunk_budget_seconds": budget,
        "cap_70pct_seconds": remaining * MAX_CHUNK_FRACTION,
    }


def chunk_optimizer_steps_from_budget(chunk_budget_seconds: float,
                                      load_overhead_seconds: float,
                                      accumulation: int,
                                      seconds_per_micro_step: float) -> int:
    """Max optimiser steps that fit in a chunk budget (floor).

    Steps are floored, never rounded up: rounding up could push
    a chunk past its budget. If the budget does not even cover
    load overhead plus one optimiser step, the result is 0 and
    the caller must abort (a chunk that cannot do one step is
    not a chunk).
    """
    usable = chunk_budget_seconds - load_overhead_seconds
    if usable <= 0:
        return 0
    per_opt_step = accumulation * seconds_per_micro_step
    return int(usable // per_opt_step)


def resolve_chunk_plan(sizing: dict, chunk: dict, args) -> dict:
    """Resolve the chunk's optimiser-step limit. Aborts if it cannot fit.

    Priority: explicit --max-chunk-optimizer-steps if given (but
    it must fit inside the budget-derived maximum, else abort
    loudly BEFORE any model is loaded -- failing fast burns no
    quota); otherwise the budget-derived maximum. The chunk
    limit is also capped at the run's remaining optimiser steps.
    """
    budget_steps = chunk_optimizer_steps_from_budget(
        chunk["chunk_budget_seconds"], args.load_overhead_seconds,
        args.accumulation_steps, args.measured_seconds_per_micro_step)
    if budget_steps <= 0:
        raise ValueError(
            f"chunk budget {chunk['chunk_budget_seconds']:.0f}s does "
            f"not cover load overhead "
            f"({args.load_overhead_seconds:.0f}s) plus one optimiser "
            f"step ({args.accumulation_steps * args.measured_seconds_per_micro_step:.3f}s); "
            f"increase the budget (within the 70% cap) or do not run.")
    if args.max_chunk_optimizer_steps is not None:
        if args.max_chunk_optimizer_steps <= 0:
            raise ValueError("--max-chunk-optimizer-steps must be > 0")
        if args.max_chunk_optimizer_steps > budget_steps:
            raise ValueError(
                f"--max-chunk-optimizer-steps "
                f"({args.max_chunk_optimizer_steps}) exceeds the "
                f"budget-derived maximum ({budget_steps}) for a "
                f"{chunk['chunk_budget_seconds']:.0f}s chunk budget at "
                f"{args.measured_seconds_per_micro_step}s/micro-step; "
                f"lower the requested chunk size, raise the budget "
                f"within the 70% cap, or re-measure step time.")
        chunk_steps = args.max_chunk_optimizer_steps
    else:
        chunk_steps = budget_steps
    return {
        "budget_derived_optimizer_steps": budget_steps,
        "chunk_optimizer_steps": chunk_steps,
    }


# ---------------------------------------------------------------- dataset

RECORD_KEYS = {
    "buggy_code", "fixed_code", "mutations", "failing_test", "bug_class",
}


def load_records(path: Path) -> list[dict]:
    """Load and validate synthetic-bug records (pilot format, exact keys).

    Loud rejection, as in the pilot: a record with missing/extra
    keys, identical buggy/fixed code, or a malformed mutation
    entry aborts the run instead of silently entering training.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if set(record) != RECORD_KEYS:
                raise ValueError(
                    f"line {lineno}: unexpected record keys "
                    f"{sorted(record)} (expected {sorted(RECORD_KEYS)})")
            if record["buggy_code"] == record["fixed_code"]:
                raise ValueError(f"line {lineno}: buggy_code == fixed_code")
            for mutation in record["mutations"]:
                for key in ("filename", "operator", "seed", "line",
                            "original_code", "mutated_code"):
                    if key not in mutation:
                        raise ValueError(
                            f"line {lineno}: mutation missing field {key!r}")
            records.append(record)
    if not records:
        raise ValueError(f"no records loaded from {path}")
    return records


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_text_example(rec: dict) -> str:
    """One training text, EXACTLY the eval prompt plus fixed code.

    v19 change (see header): the bug-class label is DROPPED. Eval
    (harness/patcher_stub.py build_trained_format_prompt) prompts
    with "Fix this Python bug:\n<buggy>\nFixed code:\n" and no
    label, so training now renders the identical prefix. The
    record's bug_class field is still validated on load; it is
    simply never rendered into the prompt.
    """
    return ("Fix this Python bug:\n" + rec["buggy_code"]
            + "\nFixed code:\n" + rec["fixed_code"])


def tokenize_training_example(tokenizer, text: str, device):
    """Tokenize one v19 training example, with EOS appended as an ID.

    v19 change (see header): the v18 script trained on
    tokenizer(text) alone, so the model never learned to emit EOS
    after the fixed file and loops to the generation cap at
    inference. This helper takes the tokenizer's own ids for the
    text (automatic BOS handling unchanged), asserts eos_token_id
    exists and is not already the final id, appends [eos_id] as a
    TOKEN ID (the EOS token string is NEVER concatenated onto the
    text), and returns (input_ids, attention_mask) tensors of
    shape [1, L] on `device`. The caller sets labels = input_ids,
    so the shifted loss trains EOS prediction right after the
    final fixed-code token. Any violation aborts loudly: training
    on a doubled EOS, a missing EOS, or an empty sequence would
    silently teach the wrong stopping behaviour.
    """
    import torch

    ids = tokenizer(text)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise SystemExit(
            "[full] ABORT: tokenizer.eos_token_id is None; cannot "
            "append EOS to training examples. Refusing to train a "
            "v19 run without a stopping token.")
    if not ids:
        raise SystemExit(
            "[full] ABORT: training text tokenized to 0 ids; "
            "refusing to train on an empty sequence.")
    if ids[-1] == eos_id:
        raise SystemExit(
            "[full] ABORT: tokenizer already appends EOS "
            f"(eos_token_id={eos_id} is the final id); appending "
            "again would teach a doubled EOS. Fix the tokenizer "
            "configuration instead of training on doubled EOS.")
    ids = list(ids) + [eos_id]
    attention_mask = [1] * len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.tensor([attention_mask], dtype=torch.long,
                        device=device)
    return input_ids, mask


def epoch_order(num_records: int, epoch: int, seed: int) -> list[int]:
    """Deterministic per-epoch shuffle: seed + epoch, never global RNG.

    Resume reconstructs exactly this order, so a resumed chunk
    continues the same sequence instead of re-shuffling.
    """
    import random

    order = list(range(num_records))
    random.Random(seed + epoch).shuffle(order)
    return order


def expected_optimizer_steps(epoch: int, index_in_epoch: int,
                             num_records: int, accumulation: int) -> int:
    """Optimiser steps completed at resume position (epoch, index).

    Position (epoch, index) means: epochs 1..epoch-1 fully done,
    and `index` micros of the current epoch done. Valid
    checkpoint positions are group boundaries (index % accum
    == 0), because checkpoints are only written after an
    optimiser step; the caller validates that separately.
    """
    per_epoch = math.ceil(num_records / accumulation)
    return (epoch - 1) * per_epoch + index_in_epoch // accumulation


# ---------------------------------------------------------------- arguments

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full manual-LoRA training run for Gemma 4 "
                    "Repo-Blind Repair (Kaggle only for real training; "
                    "--dry-run checks the plan without GPU/model).")
    parser.add_argument("--dataset", required=True,
                        help="Path to the JSONL dataset (pilot format). "
                             "Required, so a run never silently trains "
                             "on a stale default file.")
    parser.add_argument("--output-dir", default="full_train_output",
                        help="Base output directory. A fresh run writes "
                             "to <output-dir>/run-<run-id>/; on resume, "
                             "checkpoints continue in the resumed run's "
                             "own directory (see --resume-from). On "
                             "Kaggle, point this under /kaggle/working "
                             "and use Save Version afterwards.")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (default: UTC timestamp). "
                             "Fresh runs: new run directory name. "
                             "Resume: must match the resumed run, or "
                             "omit it (resume derives the run dir "
                             "from --resume-from).")
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint directory to resume from, e.g. "
                             "full_train_output/run-<id>/checkpoint-opt-100. "
                             "Must contain adapter_config.json, "
                             "adapter_model.safetensors, optimizer.pt, "
                             "scheduler.pt, training_state.json. Any "
                             "inconsistency aborts; there is no silent "
                             "restart from step 0.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Epochs (default {DEFAULT_EPOCHS}).")
    parser.add_argument("--accumulation-steps", type=int,
                        default=DEFAULT_ACCUMULATION_STEPS,
                        help="Micro-steps per optimiser step "
                             f"(default {DEFAULT_ACCUMULATION_STEPS}; "
                             "micro-batch is fixed at 1, so this is "
                             "also the effective batch size).")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-optimizer-steps", type=int,
                        default=DEFAULT_WARMUP_OPTIMIZER_STEPS,
                        help="Linear warmup in optimiser steps, then "
                             "cosine decay to 0 over total optimiser "
                             "steps (default "
                             f"{DEFAULT_WARMUP_OPTIMIZER_STEPS}).")
    parser.add_argument("--checkpoint-every-steps", type=int,
                        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
                        help="Full checkpoint every N optimiser steps "
                             f"(default {DEFAULT_CHECKPOINT_EVERY_STEPS}; "
                             "epoch-end, 15-minute, chunk-end, and "
                             "run-end checkpoints happen regardless).")
    parser.add_argument("--checkpoint-every-seconds", type=float,
                        default=DEFAULT_CHECKPOINT_EVERY_SECONDS,
                        help="Full checkpoint at least every N seconds "
                             "of training wall-clock, checked after "
                             "each optimiser step "
                             f"(default {DEFAULT_CHECKPOINT_EVERY_SECONDS:.0f} = 15 min).")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="Gradient clipping norm; 0 disables "
                             "clipping (pilot-faithful default -- the "
                             "pilot did not clip). Set > 0 only as an "
                             "explicit operator decision.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="ESCAPE HATCH ONLY: enable non-reentrant "
                             "gradient checkpointing. Default is OFF, "
                             "as proven by the pilot. Enabling it "
                             "reintroduces the v1 failure class; use "
                             "only if the first chunk OOMs on longer "
                             "sequences, and expect the step-1 smoke "
                             "check to abort if grad flow fails.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed (per-epoch order is "
                             "seed + epoch; default 42).")
    # Quota / chunking inputs. The script cannot query the Kaggle
    # quota meter (internet off, no API): the operator MUST pass
    # a current --quota-used-seconds before every chunk.
    parser.add_argument("--weekly-quota-seconds", type=float,
                        default=DEFAULT_WEEKLY_QUOTA_SECONDS,
                        help="Weekly Kaggle GPU quota in seconds "
                             f"(default {DEFAULT_WEEKLY_QUOTA_SECONDS:.0f} = 30 h).")
    parser.add_argument("--quota-used-seconds", type=float,
                        default=DEFAULT_QUOTA_USED_SECONDS,
                        help="Quota already used this week in seconds "
                             f"(default {DEFAULT_QUOTA_USED_SECONDS:.0f} = 03:15, "
                             "the 31 Oct figure -- UPDATE THIS from "
                             "the Kaggle quota meter before each chunk).")
    parser.add_argument("--chunk-fraction", type=float,
                        default=DEFAULT_CHUNK_FRACTION,
                        help="Chunk budget as a fraction of remaining "
                             f"quota (default {DEFAULT_CHUNK_FRACTION}; "
                             f"hard cap {MAX_CHUNK_FRACTION}).")
    parser.add_argument("--chunk-budget-seconds", type=float, default=None,
                        help="Explicit chunk wall-clock budget in "
                             "seconds (overrides the fraction-derived "
                             "budget, but still capped at 70% of "
                             "remaining quota).")
    parser.add_argument("--max-chunk-optimizer-steps", type=int, default=None,
                        help="Chunk size in optimiser steps (the chunk "
                             "size parameter). If omitted, it is "
                             "derived from the chunk budget and "
                             "--measured-seconds-per-micro-step. If "
                             "given, it must fit inside that budget, "
                             "else the run aborts before loading.")
    parser.add_argument("--measured-seconds-per-micro-step", type=float,
                        default=DEFAULT_MEASURED_SECONDS_PER_MICRO_STEP,
                        help="Seconds per micro-step used for sizing "
                             f"(default {DEFAULT_MEASURED_SECONDS_PER_MICRO_STEP}, "
                             "the pilot median for steps 4-20). After "
                             "the first full-run chunk, replace this "
                             "with the re-measured median.")
    parser.add_argument("--load-overhead-seconds", type=float,
                        default=DEFAULT_LOAD_OVERHEAD_SECONDS,
                        help="Load/attach time allowance in seconds "
                             f"(default {DEFAULT_LOAD_OVERHEAD_SECONDS:.0f}, "
                             "the pilot session allowance).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate arguments, dataset, sizing, "
                             "chunk budget, and (if given) resume "
                             "checkpoint metadata, then exit WITHOUT "
                             "loading a model, importing GPU "
                             "libraries, or training.")
    return parser


def validate_resume_metadata(resume_dir: Path, args, records: list[dict],
                             dataset_hash: str, sizing: dict) -> dict:
    """Validate a resume checkpoint WITHOUT a model (dry-run + pre-load).

    Checks: directory exists; all five files present; state JSON
    parses; format version, base, LoRA hyper-parameters, target
    modules, dataset record count + SHA-256, seed, epochs,
    accumulation, warmup, and total optimiser steps all match
    the current run; directory number matches state optimiser
    step; resume position is a group boundary; and the stored
    optimiser step equals the position-derived step count.
    ANY mismatch raises ValueError (loud abort, never fallback).
    The model-side checks (adapter keys/shapes vs the live
    enumeration, optimiser param count) run in main() after the
    model loads, because they need the model.
    """
    if not resume_dir.is_dir():
        raise ValueError(
            f"--resume-from {resume_dir} is not a directory; refusing "
            f"to resume. To start fresh, use a new --output-dir / "
            f"--run-id instead of pointing at a missing checkpoint.")
    required = ["adapter_config.json", "adapter_model.safetensors",
                "optimizer.pt", "scheduler.pt", "training_state.json"]
    missing = [f for f in required if not (resume_dir / f).is_file()]
    if missing:
        raise ValueError(
            f"resume checkpoint {resume_dir} is INCOMPLETE (missing "
            f"{missing}); refusing to resume from a partial "
            f"checkpoint. Resume from an earlier COMPLETE checkpoint "
            f"instead -- do not restart from step 0 silently.")
    state = json.loads((resume_dir / "training_state.json").read_text(
        encoding="utf-8"))
    try:
        dir_step = int(resume_dir.name.rsplit("-", 1)[1])
    except (ValueError, IndexError):
        raise ValueError(
            f"resume directory name {resume_dir.name} does not end in "
            f"a step number (expected checkpoint-opt-<step>); "
            f"refusing to guess the resume point.")
    checks: list[tuple[str, object, object]] = [
        ("format_version", state.get("format_version"), FORMAT_VERSION),
        ("prompt_format", state.get("prompt_format"), PROMPT_FORMAT_V19),
        ("base_model", state.get("base_model"), MODEL_HANDLE),
        ("lora_r", state.get("lora_r"), args.lora_r),
        ("lora_alpha", state.get("lora_alpha"), args.lora_alpha),
        ("learning_rate", state.get("learning_rate"), args.learning_rate),
        ("target_modules", state.get("target_modules"), TARGET_MODULES),
        ("dataset_records", state.get("dataset_records"), len(records)),
        ("dataset_sha256", state.get("dataset_sha256"), dataset_hash),
        ("seed", state.get("seed"), args.seed),
        ("epochs", state.get("epochs"), args.epochs),
        ("accumulation_steps", state.get("accumulation_steps"),
         args.accumulation_steps),
        ("warmup_optimizer_steps", state.get("warmup_optimizer_steps"),
         args.warmup_optimizer_steps),
        ("total_optimizer_steps", state.get("total_optimizer_steps"),
         sizing["optimizer_steps"]),
    ]
    for name, got, want in checks:
        if got != want:
            raise ValueError(
                f"resume checkpoint INCONSISTENT: {name} is {got!r} "
                f"in the checkpoint but {want!r} for this run; "
                f"refusing to resume onto mismatched configuration "
                f"or data. Start a fresh run (new run-id) or resume "
                f"with matching arguments and the same dataset file.")
    if state.get("optimizer_step") != dir_step:
        raise ValueError(
            f"resume checkpoint INCONSISTENT: directory step "
            f"{dir_step} != state optimizer_step "
            f"{state.get('optimizer_step')}; refusing to guess which "
            f"is the true resume point.")
    epoch = state.get("next_epoch")
    index = state.get("next_index_in_epoch")
    if not isinstance(epoch, int) or not isinstance(index, int):
        raise ValueError("resume state lacks a valid next position "
                         "(next_epoch / next_index_in_epoch).")
    if not (1 <= epoch <= args.epochs) or not (0 <= index < len(records)):
        # The one legal edge: a fully complete run has no next
        # position; that is caught by optimizer_step == total below.
        if not (state.get("optimizer_step") == sizing["optimizer_steps"]):
            raise ValueError(
                f"resume position out of range: epoch {epoch}, index "
                f"{index} for N={len(records)}, epochs={args.epochs}.")
    if index % args.accumulation_steps != 0:
        raise ValueError(
            f"resume checkpoint INCONSISTENT: next index {index} is "
            f"not an accumulation-group boundary "
            f"(accumulation={args.accumulation_steps}); checkpoints "
            f"are only written at optimiser-step boundaries, so a "
            f"mid-group resume point means a corrupt or foreign "
            f"checkpoint. Refusing to resume.")
    derived = expected_optimizer_steps(epoch, index, len(records),
                                       args.accumulation_steps)
    if derived != state.get("optimizer_step"):
        raise ValueError(
            f"resume checkpoint INCONSISTENT: position (epoch "
            f"{epoch}, index {index}) implies {derived} optimiser "
            f"steps, but state says {state.get('optimizer_step')}; "
            f"refusing to resume from an inconsistent point.")
    if state.get("optimizer_step") >= sizing["optimizer_steps"]:
        raise ValueError(
            f"resume checkpoint is already COMPLETE "
            f"({state.get('optimizer_step')} / "
            f"{sizing['optimizer_steps']} optimiser steps); there "
            f"is nothing to resume. Do not re-run a finished run.")
    return state


def run_dry_run(args) -> int:
    """Dry-run: plan + validate, no GPU libraries, no model, no training."""
    dataset_path = Path(args.dataset)
    records = load_records(dataset_path)
    dataset_hash = dataset_sha256(dataset_path)
    sizing = compute_sizing(len(records), args.epochs,
                            args.accumulation_steps,
                            args.measured_seconds_per_micro_step,
                            args.load_overhead_seconds)
    chunk = compute_chunk_budget(args.weekly_quota_seconds,
                                 args.quota_used_seconds,
                                 args.chunk_fraction,
                                 args.chunk_budget_seconds)
    plan = resolve_chunk_plan(sizing, chunk, args)
    resume_state = None
    if args.resume_from:
        resume_state = validate_resume_metadata(
            Path(args.resume_from), args, records, dataset_hash, sizing)
    print("=== FULL-RUN DRY-RUN PLAN (no training, no model load) ===")
    print(f"dataset: {dataset_path} (N={len(records)}, "
          f"sha256={dataset_hash[:16]}...)")
    print(f"base: {MODEL_HANDLE} (unchanged); LoRA r={args.lora_r} "
          f"alpha={args.lora_alpha} lr={args.learning_rate}")
    print(f"epochs={args.epochs}, micro-batch=1, accumulation="
          f"{args.accumulation_steps}, B_eff={sizing['effective_batch']}")
    print(f"total optimiser steps={sizing['optimizer_steps']}, "
          f"micro-steps (upper)={sizing['micro_steps_upper']}")
    print(f"sizing at {args.measured_seconds_per_micro_step}s/micro-step "
          f"+ {args.load_overhead_seconds:.0f}s load: stepping="
          f"{sizing['stepping_seconds']:.1f}s, total="
          f"{sizing['total_seconds']:.1f}s "
          f"({sizing['total_seconds'] / 60:.1f} min, "
          f"{sizing['total_seconds'] / 3600:.2f} h)")
    print(f"quota: remaining={chunk['remaining_seconds']:.0f}s "
          f"({chunk['remaining_seconds'] / 3600:.2f} h); chunk budget="
          f"{chunk['chunk_budget_seconds']:.0f}s "
          f"({chunk['chunk_budget_seconds'] / 3600:.2f} h, "
          f"{chunk['chunk_fraction']:.0%} of remaining; 70% cap="
          f"{chunk['cap_70pct_seconds']:.0f}s)")
    print(f"chunk: budget-derived max={plan['budget_derived_optimizer_steps']} "
          f"optimiser steps; this chunk limit="
          f"{plan['chunk_optimizer_steps']} optimiser steps")
    if resume_state is not None:
        print(f"resume: VALIDATED metadata for {args.resume_from}: "
              f"optimizer_step={resume_state['optimizer_step']} / "
              f"{sizing['optimizer_steps']}, next position epoch="
              f"{resume_state['next_epoch']} index="
              f"{resume_state['next_index_in_epoch']}. NOTE: dry-run "
              f"cannot verify adapter keys/shapes against the live "
              f"model or optimiser param counts -- main() re-checks "
              f"those after loading, before any training step.")
    else:
        print("resume: fresh run (no --resume-from). If the output "
              "run directory already exists and is non-empty, main() "
              "aborts rather than overwriting it.")
    print("DRY-RUN PASSED: argument parsing, dataset validation, "
          "sizing, chunk budget, and resume metadata (if any) all "
          "check out. This proves NOTHING about GPU step time, "
          "memory, resume in a fresh Kaggle session, or serving.")
    return 0


# ---------------------------------------------------------------- adapter

def adapter_config_dict(args) -> dict:
    """PEFT-compatible adapter_config.json contents (pilot format)."""
    return {
        "base_model_name_or_path": MODEL_HANDLE,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": list(TARGET_MODULES),
        "inference_mode": False,
    }


def peft_key(module_path: str, factor: str) -> str:
    """PEFT state-dict key for one factor (pilot convention, unchanged)."""
    assert factor in ("lora_A", "lora_B")
    return f"base_model.model.{module_path}.{factor}.default.weight"


def attach_manual_lora(model, args):
    """Attach manual LoRA via register_forward_hook (pilot v2/v3 method).

    Returns (factors, hook_calls). Factors are plain nn.Parameters
    held in a list, NOT registered on modules, so model.parameters()
    stays base-only. All base parameters are frozen first,
    including any weight_scale / weight_shape the dequantized
    load still carries. B starts at zero, so the adapter is an
    exact no-op at step 0.
    """
    import math
    import torch

    for param in model.parameters():
        param.requires_grad_(False)
    scaling = args.lora_alpha / args.lora_r
    factors: list[dict] = []
    hook_calls = [0]
    seen_ids: set[int] = set()
    for path, module in model.named_modules():
        if path.rsplit(".", 1)[-1] not in TARGET_MODULES:
            continue
        if id(module) in seen_ids:
            raise SystemExit(
                f"[full] ABORT: target module object shared between "
                f"paths (seen again at {path}); manual LoRA cannot "
                f"give shared modules independent factors.")
        seen_ids.add(id(module))
        if not hasattr(module, "weight"):
            raise SystemExit(
                f"[full] ABORT: target module {path} has no .weight "
                f"(packed load?). Run the dequantized load, not this "
                f"script, on a packed checkpoint.")
        weight = module.weight
        if weight.device.type == "meta":
            raise SystemExit(
                f"[full] ABORT: target module {path} is on the meta "
                f"device (offloaded, no storage); cannot place LoRA "
                f"factors for it. Re-load without meta offloading.")
        in_features = getattr(module, "in_features", weight.shape[1])
        out_features = getattr(module, "out_features", weight.shape[0])
        device = weight.device
        dtype = weight.dtype
        a_param = torch.nn.Parameter(
            torch.empty((args.lora_r, in_features), device=device,
                        dtype=dtype))
        torch.nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))
        b_param = torch.nn.Parameter(
            torch.zeros((out_features, args.lora_r), device=device,
                        dtype=dtype))

        def make_hook(a_mat, b_mat, scale, counter):
            def hook(_mod, inputs, output):
                counter[0] += 1
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "[full] manual-LoRA hook expected a Tensor "
                        "output from a target linear module; got "
                        f"{type(output).__name__}. Refusing to guess "
                        "how to inject the delta.")
                if not inputs:
                    raise TypeError(
                        "[full] manual-LoRA hook called without an "
                        "input tensor")
                x = inputs[0]
                delta = torch.matmul(torch.matmul(x, a_mat.t()), b_mat.t())
                delta = delta * scale
                return output + delta.to(dtype=output.dtype)
            return hook

        handle = module.register_forward_hook(
            make_hook(a_param, b_param, scaling, hook_calls))
        factors.append({
            "path": path,
            "module": module,
            "A": a_param,
            "B": b_param,
            "scaling": scaling,
            "in_features": in_features,
            "out_features": out_features,
            "hook_handle": handle,
        })
    if not factors:
        raise SystemExit("[full] ABORT: no target modules found; "
                         "check TARGET_MODULES against the model.")
    if any(p.requires_grad for p in model.parameters()):
        raise SystemExit("[full] ABORT: a base parameter is still "
                         "trainable after freezing; refusing to train.")
    if not all(f["A"].requires_grad and f["B"].requires_grad
               for f in factors):
        raise SystemExit("[full] ABORT: a LoRA factor is not "
                         "trainable; refusing to train.")
    print(f"[full] manual LoRA attached: {len(factors)} target modules, "
          f"{2 * len(factors)} trainable factors, r={args.lora_r}, "
          f"alpha={args.lora_alpha}, scaling={scaling}, via "
          f"register_forward_hook", flush=True)
    return factors, hook_calls


def save_adapter(checkpoint_dir: Path, factors: list[dict], args) -> None:
    """Save adapter_config.json + adapter_model.safetensors (pilot format)."""
    from safetensors.torch import save_file

    tensors = {}
    for factor in factors:
        tensors[peft_key(factor["path"], "lora_A")] = \
            factor["A"].detach().to("cpu").contiguous()
        tensors[peft_key(factor["path"], "lora_B")] = \
            factor["B"].detach().to("cpu").contiguous()
    (checkpoint_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config_dict(args), indent=2), encoding="utf-8")
    save_file(tensors, str(checkpoint_dir / "adapter_model.safetensors"))


def reload_check(checkpoint_dir: Path, factors: list[dict]) -> None:
    """Reload the saved safetensors; verify keys, shapes, values exactly."""
    import torch
    from safetensors.torch import load_file

    saved = load_file(str(checkpoint_dir / "adapter_model.safetensors"))
    expected: dict[str, object] = {}
    for factor in factors:
        expected[peft_key(factor["path"], "lora_A")] = factor["A"]
        expected[peft_key(factor["path"], "lora_B")] = factor["B"]
    if set(saved) != set(expected):
        missing = sorted(set(expected) - set(saved))
        extra = sorted(set(saved) - set(expected))
        raise SystemExit(
            f"[full] RELOAD CHECK FAILED: key mismatch "
            f"(missing {len(missing)}, extra {len(extra)}); "
            f"missing[:5]={missing[:5]}, extra[:5]={extra[:5]}")
    for key, live in expected.items():
        disk = saved[key]
        if tuple(disk.shape) != tuple(live.shape):
            raise SystemExit(
                f"[full] RELOAD CHECK FAILED: shape mismatch for "
                f"{key}: disk {tuple(disk.shape)} vs live "
                f"{tuple(live.shape)}")
        if not torch.equal(disk, live.detach().to("cpu")):
            raise SystemExit(
                f"[full] RELOAD CHECK FAILED: value mismatch for {key}")
    print(f"[full] reload check PASSED: {len(saved)} tensors, keys + "
          f"shapes + values match the live factors", flush=True)


def load_adapter_into_factors(checkpoint_dir: Path, factors: list[dict]) -> None:
    """Load a saved adapter into live factors, verifying keys/shapes first.

    Aborts loudly on any key/shape mismatch BEFORE overwriting any
    factor, so a bad checkpoint cannot half-load and then train.
    """
    import torch
    from safetensors.torch import load_file

    saved = load_file(str(checkpoint_dir / "adapter_model.safetensors"))
    expected_keys = set()
    for factor in factors:
        expected_keys.add(peft_key(factor["path"], "lora_A"))
        expected_keys.add(peft_key(factor["path"], "lora_B"))
    if set(saved) != expected_keys:
        missing = sorted(expected_keys - set(saved))
        extra = sorted(set(saved) - expected_keys)
        raise SystemExit(
            f"[full] RESUME ABORT: adapter key mismatch in "
            f"{checkpoint_dir} (missing {len(missing)}, extra "
            f"{len(extra)}); missing[:5]={missing[:5]}, "
            f"extra[:5]={extra[:5]}. Refusing to resume.")
    for factor in factors:
        for key_suffix, param_key in (("lora_A", "A"), ("lora_B", "B")):
            disk = saved[peft_key(factor["path"], key_suffix)]
            live = factor[param_key]
            if tuple(disk.shape) != tuple(live.shape):
                raise SystemExit(
                    f"[full] RESUME ABORT: shape mismatch for "
                    f"{peft_key(factor['path'], key_suffix)}: disk "
                    f"{tuple(disk.shape)} vs live {tuple(live.shape)}.")
    # All checks passed; only now overwrite the factors.
    with torch.no_grad():
        for factor in factors:
            factor["A"].copy_(
                saved[peft_key(factor["path"], "lora_A")].to(
                    device=factor["A"].device, dtype=factor["A"].dtype))
            factor["B"].copy_(
                saved[peft_key(factor["path"], "lora_B")].to(
                    device=factor["B"].device, dtype=factor["B"].dtype))
    print(f"[full] adapter loaded from {checkpoint_dir}: "
          f"{len(saved)} tensors, keys + shapes verified", flush=True)


def resolve_body_and_lm_head(model):
    """Resolve body + lm_head for the manual-logits path (pilot v3, unchanged)."""
    if not hasattr(model, "model"):
        raise SystemExit(
            "[full] ABORT: loaded model has no .model attribute "
            f"(type {type(model).__name__}); the manual-logits path "
            "requires the Gemma4Unified body at model.model. "
            "Refusing to guess the body location.")
    body = model.model
    if not hasattr(model, "lm_head"):
        raise SystemExit(
            "[full] ABORT: loaded model has no .lm_head attribute "
            f"(type {type(model).__name__}); the manual-logits path "
            "requires the outer lm_head. Refusing to guess.")
    lm_head = model.lm_head
    if not hasattr(lm_head, "weight"):
        raise SystemExit(
            "[full] ABORT: lm_head has no .weight tensor "
            f"(type {type(lm_head).__name__}); cannot compute logits "
            "manually. Refusing to guess.")
    if lm_head.weight.requires_grad:
        raise SystemExit(
            "[full] ABORT: lm_head weight is still trainable after "
            "freezing (it is tied to embed_tokens); the manual-logits "
            "path requires it frozen. Refusing to train embeddings.")
    print(f"[full] manual-logits path: body type "
          f"{type(body).__name__}, lm_head type "
          f"{type(lm_head).__name__}, lm_head weight shape "
          f"{tuple(lm_head.weight.shape)}, bias="
          f"{'present' if getattr(lm_head, 'bias', None) is not None else 'none'}",
          flush=True)
    return body, lm_head


def manual_logits_and_loss(body, lm_head, input_ids, attention_mask, labels):
    """Manual logits + causal-LM loss, bypassing lm_head forward (pilot v3)."""
    import torch
    import torch.nn.functional as F

    if labels.shape[-1] < 2:
        raise SystemExit(
            "[full] ABORT: sequence length "
            f"{labels.shape[-1]} is too short for the causal-LM "
            "shift (needs >= 2 tokens).")
    body_outputs = body(input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
    if not hasattr(body_outputs, "last_hidden_state"):
        raise SystemExit(
            "[full] ABORT: body output has no last_hidden_state "
            f"(type {type(body_outputs).__name__}); cannot compute "
            "logits manually. Refusing to guess the hidden-state "
            "location.")
    hidden_states = body_outputs.last_hidden_state
    if hidden_states.device != lm_head.weight.device:
        print(f"[full] manual-logits: moving hidden states from "
              f"{hidden_states.device} to lm_head device "
              f"{lm_head.weight.device}", flush=True)
        hidden_states = hidden_states.to(lm_head.weight.device)
    logits = F.linear(hidden_states, lm_head.weight,
                      getattr(lm_head, "bias", None))
    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels.to(device=logits.device)[..., 1:]
    loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1), ignore_index=-100)
    return loss, logits, hidden_states


# ---------------------------------------------------------------- scheduler

def make_scheduler(optimizer, total_optimizer_steps: int,
                   warmup_optimizer_steps: int):
    """Linear warmup (from 1/warmup, so step 1 is not a zero-LR step),
    then cosine decay to 0 over the TOTAL run steps (not the chunk).

    The scheduler is built over total steps so that a resumed
    chunk continues the same schedule instead of restarting it.
    Warmup longer than total is clamped to total.
    """
    import math
    import torch

    total = max(1, total_optimizer_steps)
    warmup = min(max(0, warmup_optimizer_steps), total)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if total <= warmup:
            return 0.0
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------- checkpoints

def build_training_state(args, sizing: dict, dataset_hash: str,
                         optimizer_step: int, next_epoch: int,
                         next_index_in_epoch: int) -> dict:
    """Training state JSON: everything resume validation compares."""
    return {
        "format_version": FORMAT_VERSION,
        "prompt_format": PROMPT_FORMAT_V19,
        "base_model": MODEL_HANDLE,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "target_modules": list(TARGET_MODULES),
        "dataset_records": sizing["num_records"],
        "dataset_sha256": dataset_hash,
        "dataset_path_as_run": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "accumulation_steps": args.accumulation_steps,
        "warmup_optimizer_steps": args.warmup_optimizer_steps,
        "total_optimizer_steps": sizing["optimizer_steps"],
        "optimizer_step": optimizer_step,
        "next_epoch": next_epoch,
        "next_index_in_epoch": next_index_in_epoch,
    }


def save_full_checkpoint(run_dir: Path, optimizer_step: int,
                         factors: list[dict], optimizer, scheduler,
                         state: dict) -> Path:
    """Save a FULL checkpoint atomically-ish (tmp dir + rename).

    Writes adapter (pilot format) + optimizer.pt + scheduler.pt +
    training_state.json into a tmp directory, renames it to
    checkpoint-opt-<step>, then runs the adapter reload check
    against the live factors. On reload-check failure the
    directory is renamed to *.CORRUPT and the run aborts: a
    bad checkpoint must never be mistaken for a resume point.
    """
    import torch

    final_dir = run_dir / f"checkpoint-opt-{optimizer_step}"
    tmp_dir = run_dir / f".tmp-checkpoint-opt-{optimizer_step}"
    if final_dir.exists():
        raise SystemExit(
            f"[full] ABORT: checkpoint directory {final_dir} already "
            f"exists; refusing to overwrite a checkpoint (it may be "
            f"the resume point of another session).")
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        # Adapter first (pilot format), then optimiser/scheduler/state.
        from safetensors.torch import save_file
        tensors = {}
        for factor in factors:
            tensors[peft_key(factor["path"], "lora_A")] = \
                factor["A"].detach().to("cpu").contiguous()
            tensors[peft_key(factor["path"], "lora_B")] = \
                factor["B"].detach().to("cpu").contiguous()
        (tmp_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config_dict_from_state(state),
                       indent=2), encoding="utf-8")
        save_file(tensors, str(tmp_dir / "adapter_model.safetensors"))
        torch.save(optimizer.state_dict(), str(tmp_dir / "optimizer.pt"))
        torch.save(scheduler.state_dict(), str(tmp_dir / "scheduler.pt"))
        (tmp_dir / "training_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        os.rename(tmp_dir, final_dir)
    except Exception:
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    try:
        reload_check(final_dir, factors)
    except SystemExit:
        corrupt = Path(str(final_dir) + ".CORRUPT")
        os.rename(final_dir, corrupt)
        print(f"[full] corrupt checkpoint moved to {corrupt}", flush=True)
        raise
    print(f"[full] full checkpoint saved: {final_dir}", flush=True)
    return final_dir


def adapter_config_dict_from_state(state: dict) -> dict:
    """Adapter config from a training state (same fields as the pilot)."""
    return {
        "base_model_name_or_path": MODEL_HANDLE,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": state["lora_r"],
        "lora_alpha": state["lora_alpha"],
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": list(state["target_modules"]),
        "inference_mode": False,
    }


# ---------------------------------------------------------------- main

def main() -> None:
    # Heavy imports live here so the file syntax-checks without
    # GPU libraries, and so --dry-run never touches them.
    # NOTE: deliberately NO PEFT import and NO torchao import
    # anywhere in this file (see header).
    import torch

    args = build_parser().parse_args()
    if args.dry_run:
        # build_parser already parsed; dry-run is handled in __main__
        # dispatch before main() is reached, so this branch is
        # defensive only and must never be reached in practice.
        raise SystemExit("[full] internal error: dry-run reached main()")

    run_start = time.monotonic()
    dataset_path = Path(args.dataset)
    records = load_records(dataset_path)
    dataset_hash = dataset_sha256(dataset_path)
    sizing = compute_sizing(len(records), args.epochs,
                            args.accumulation_steps,
                            args.measured_seconds_per_micro_step,
                            args.load_overhead_seconds)
    chunk = compute_chunk_budget(args.weekly_quota_seconds,
                                 args.quota_used_seconds,
                                 args.chunk_fraction,
                                 args.chunk_budget_seconds)
    plan = resolve_chunk_plan(sizing, chunk, args)
    total_opt_steps = sizing["optimizer_steps"]
    num_records = len(records)

    # --- Resolve run directory and resume point (before model load).
    resume_state = None
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        resume_state = validate_resume_metadata(
            resume_dir, args, records, dataset_hash, sizing)
        run_dir = resume_dir.parent
        if args.run_id is not None and args.run_id != run_dir.name:
            raise SystemExit(
                f"[full] ABORT: --run-id {args.run_id!r} does not "
                f"match the resumed run directory {run_dir.name!r}; "
                f"refusing to split one run across two directories. "
                f"Omit --run-id when resuming.")
        start_opt_step = resume_state["optimizer_step"]
        start_epoch = resume_state["next_epoch"]
        start_index = resume_state["next_index_in_epoch"]
        print(f"[full] RESUMING: run dir {run_dir}, checkpoint "
              f"{resume_dir}, optimiser step {start_opt_step} / "
              f"{total_opt_steps}, next position epoch {start_epoch} "
              f"index {start_index} (metadata validated; adapter / "
              f"optimiser / scheduler contents are verified again "
              f"after the model loads).", flush=True)
    else:
        run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ",
                                              time.gmtime())
        run_dir = Path(args.output_dir) / f"run-{run_id}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(
                f"[full] ABORT: run directory {run_dir} already "
                f"exists and is non-empty; refusing to overwrite a "
                f"previous run. Pick a new --run-id, or resume with "
                f"--resume-from pointing at a checkpoint in it.")
        run_dir.mkdir(parents=True, exist_ok=True)
        start_opt_step = 0
        start_epoch = 1
        start_index = 0
    chunk_target_opt_step = min(
        total_opt_steps, start_opt_step + plan["chunk_optimizer_steps"])
    print(f"[full] plan: N={num_records}, epochs={args.epochs}, "
          f"B_eff={sizing['effective_batch']}, total optimiser steps="
          f"{total_opt_steps}, this chunk: optimiser steps "
          f"{start_opt_step} -> {chunk_target_opt_step} "
          f"(limit {plan['chunk_optimizer_steps']}, budget-derived "
          f"max {plan['budget_derived_optimizer_steps']}), chunk "
          f"budget {chunk['chunk_budget_seconds']:.0f}s "
          f"({chunk['chunk_fraction']:.0%} of "
          f"{chunk['remaining_seconds']:.0f}s remaining; 70% cap "
          f"{chunk['cap_70pct_seconds']:.0f}s)", flush=True)

    # --- Model load: IDENTICAL to the pilot (dequantized load).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ.get("MODEL_PATH") or MODEL_HANDLE
    print(f"[full] model path: {model_path} "
          f"(MODEL_PATH env override applied if set)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[full] v19 EOS: tokenizer.eos_token_id="
          f"{tokenizer.eos_token_id} (appended as a token id to "
          f"every training example by tokenize_training_example; "
          f"never as literal text)", flush=True)
    try:
        from transformers import CompressedTensorsConfig
        quant_config = CompressedTensorsConfig(run_compressed=False)
    except Exception:
        quant_config = {"run_compressed": False}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_config,
    )
    # HARD GATE (same as the pilot): every enumerated target must
    # carry .weight. Abort loudly on any packed remainder.
    _total = _with = 0
    for _name, _mod in model.named_modules():
        if _name.rsplit(".", 1)[-1] in TARGET_MODULES:
            _total += 1
            _with += 1 if hasattr(_mod, "weight") else 0
    print(f"[full] dequantization gate: targets with .weight = "
          f"{_with}/{_total}")
    if _total == 0 or _with != _total:
        raise SystemExit(
            f"[full] ABORT: dequantized load failed -- only "
            f"{_with}/{_total} targets have .weight. Do not train on "
            f"a packed load.")
    if _total != EXPECTED_TARGET_COUNT:
        print(f"[full] WARNING: expected {EXPECTED_TARGET_COUNT} "
              f"target modules (Gate 1), found {_total}; proceeding "
              f"only because every found target has .weight.",
              flush=True)
    # Gradient checkpointing OFF by default (pilot-proven). The
    # escape-hatch flag enables non-reentrant checkpointing plus
    # input grads; the step-1 smoke check below still gates it.
    if args.gradient_checkpointing:
        print("[full] WARNING: gradient checkpointing ENABLED via "
              "--gradient-checkpointing (escape hatch only; see "
              "header). If the step-1 smoke check fails, this chunk "
              "aborts -- do not bypass the smoke check.", flush=True)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        print("[full] gradient checkpointing OFF (pilot-proven "
              "default); see header for the memory trade-off.",
              flush=True)
    model.config.use_cache = False

    factors, hook_calls = attach_manual_lora(model, args)
    body, lm_head = resolve_body_and_lm_head(model)
    trainable = []
    for factor in factors:
        trainable.extend([factor["A"], factor["B"]])
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scheduler = make_scheduler(optimizer, total_opt_steps,
                               args.warmup_optimizer_steps)

    # --- Resume content load (after attach, before any training).
    # Adapter factors first (keys/shapes verified before any
    # overwrite), then optimiser, then scheduler. Param-count
    # checks abort loudly: a state file for a different factor
    # set must never load "successfully" onto this model.
    if resume_state is not None:
        resume_dir = Path(args.resume_from)
        load_adapter_into_factors(resume_dir, factors)
        opt_state = torch.load(str(resume_dir / "optimizer.pt"),
                               map_location="cpu", weights_only=False)
        # Optimiser state must cover exactly the live factors.
        # torch optimiser state_dicts key params by index in the
        # param group; count the param entries, not the format.
        try:
            loaded_param_count = len(opt_state["state"])
        except (KeyError, TypeError):
            loaded_param_count = -1
        if loaded_param_count != len(trainable):
            raise SystemExit(
                f"[full] RESUME ABORT: optimiser state covers "
                f"{loaded_param_count} params but this run has "
                f"{len(trainable)} trainable factors; refusing to "
                f"resume onto a mismatched optimiser state.")
        optimizer.load_state_dict(opt_state)
        sched_state = torch.load(str(resume_dir / "scheduler.pt"),
                                 map_location="cpu", weights_only=False)
        scheduler.load_state_dict(sched_state)
        print(f"[full] resume content loaded: adapter + optimiser "
              f"({loaded_param_count} params) + scheduler "
              f"(last_epoch={sched_state.get('last_epoch')}); "
              f"verifying resume point against the live model...",
              flush=True)
        # Final resume-point verification: the scheduler's epoch
        # must agree with the checkpoint's optimiser step, and
        # the adapter reload check must pass on the resumed factors.
        if sched_state.get("last_epoch", -1) not in (
                start_opt_step, start_opt_step - 1):
            # LambdaLR initialises last_epoch at -1 and increments
            # on each scheduler.step(); after N steps it is N-1
            # in some versions and N in others -- accept both,
            # abort on anything else (a schedule from a different
            # run would differ by far more than one).
            raise SystemExit(
                f"[full] RESUME ABORT: scheduler last_epoch "
                f"{sched_state.get('last_epoch')} does not agree "
                f"with resume optimiser step {start_opt_step}; "
                f"refusing to resume with a mismatched schedule.")
        reload_check(resume_dir, factors)
        print(f"[full] RESUME POINT VERIFIED: optimiser step "
              f"{start_opt_step}, epoch {start_epoch}, index "
              f"{start_index}, scheduler consistent, adapter reload "
              f"check passed on the resumed factors.", flush=True)

    # --- Token-length survey (diagnostic only, never a filter).
    # Full-run sequences may be longer than pilot sequences; this
    # survey makes that visible BEFORE stepping, so a longer-
    # sequence slowdown is expected, not a surprise.
    token_lengths = []
    for rec in records:
        text = to_text_example(rec)
        # v19: +1 per record for the appended EOS (the survey
        # measures trained sequence length, not bare text length).
        token_lengths.append(len(tokenizer(text)["input_ids"]) + 1)
    token_lengths.sort()
    median_len = token_lengths[len(token_lengths) // 2]
    max_len = token_lengths[-1]
    print(f"[full] token-length survey over {num_records} records: "
          f"median={median_len}, p90="
          f"{token_lengths[int(0.9 * (len(token_lengths) - 1))]}, "
          f"max={max_len}. Pilot step time (0.339 s) was measured "
          f"on SHORT synthetic sequences; if these lengths are "
          f"much larger, expect slower steps and re-measure (see "
          f"header).", flush=True)
    if max_len > 1024:
        print(f"[full] WARNING: max token length {max_len} > 1024 "
              f"(heuristic threshold, not a pilot measurement). "
              f"Per-micro-step time may exceed the 0.339 s sizing "
              f"assumption; the first chunk MUST re-measure before "
              f"later chunks commit to a step budget.", flush=True)

    quota_log_path = run_dir / "full_quota_log.jsonl"
    chunk_log_path = run_dir / "chunk_log.jsonl"
    opt_step = start_opt_step
    epoch = start_epoch
    index = start_index  # next micro index within `epoch` (0-based)
    last_saved_opt_step = start_opt_step if resume_state is not None else -1
    last_saved_position = (start_epoch, start_index)
    last_time_ckpt = time.monotonic()
    group_seconds_list: list[float] = []  # per optimiser step, group wall
    first_micro_done = False
    group_start_time = time.monotonic()
    chunk_reason = "completed"
    interrupted = False
    try:
        while opt_step < chunk_target_opt_step and epoch <= args.epochs:
            order = epoch_order(num_records, epoch, args.seed)
            while index < num_records and opt_step < chunk_target_opt_step:
                # Chunk wall-clock guard: checked BEFORE each micro.
                elapsed = time.monotonic() - run_start
                if elapsed > chunk["chunk_budget_seconds"]:
                    print(f"[full] CHUNK BUDGET: elapsed {elapsed:.0f}s "
                          f"> budget {chunk['chunk_budget_seconds']:.0f}s "
                          f"-- stopping chunk before micro "
                          f"(epoch {epoch}, index {index}). Completed "
                          f"optimiser steps are checkpointed; the "
                          f"partial group (if any) is discarded and "
                          f"redone by the next chunk.", flush=True)
                    chunk_reason = "chunk-budget"
                    break
                group_start = (index // args.accumulation_steps) \
                    * args.accumulation_steps
                group_end = min(group_start + args.accumulation_steps,
                                num_records) - 1
                group_size = group_end - group_start + 1
                rec = records[order[index]]
                text = to_text_example(rec)
                if index == group_start:
                    group_start_time = time.monotonic()
                step_start = time.monotonic()
                input_ids, attention_mask = tokenize_training_example(
                    tokenizer, text, model.device)
                labels = input_ids
                loss, logits, hidden_states = manual_logits_and_loss(
                    body, lm_head, labels, attention_mask, labels)
                if not first_micro_done:
                    # Step-1 smoke check (this session's first
                    # micro, fresh or resumed): hook count +
                    # hidden/logits/loss requires_grad + grad_fn.
                    # Identical semantics to the pilot's check,
                    # extended to the live factor count.
                    hooks_fired = hook_calls[0]
                    print(f"[full] first-micro smoke check: hooks "
                          f"fired {hooks_fired} (targets "
                          f"{len(factors)}); hidden_states "
                          f"requires_grad={hidden_states.requires_grad}; "
                          f"logits requires_grad={logits.requires_grad}; "
                          f"loss requires_grad={loss.requires_grad}, "
                          f"grad_fn="
                          f"{type(loss.grad_fn).__name__ if loss.grad_fn is not None else None}",
                          flush=True)
                    if hooks_fired < len(factors):
                        raise SystemExit(
                            f"[full] ABORT before backward: hooks "
                            f"fired {hooks_fired} times, fewer than "
                            f"{len(factors)} targets -- the LoRA "
                            f"deltas did not run on every target.")
                    if not loss.requires_grad or loss.grad_fn is None:
                        raise SystemExit(
                            "[full] ABORT before backward: first-micro "
                            "loss has no gradient path (hooks fired "
                            f"{hooks_fired} times). Do not continue; "
                            "fix grad flow before retrying.")
                    first_micro_done = True
                loss_value = float(loss)
                if not math.isfinite(loss_value):
                    raise SystemExit(
                        f"[full] ABORT: non-finite loss "
                        f"({loss_value}) at epoch {epoch}, index "
                        f"{index}; refusing to train on NaN/inf.")
                (loss / group_size).backward()
                index += 1
                if index - 1 == group_end:
                    # Group complete: optimiser step (the ONLY
                    # place optimiser.step() is called).
                    if opt_step == start_opt_step:
                        # First optimiser step OF THIS SESSION
                        # (fresh or resumed): grads must exist
                        # before stepping, else this session
                        # trains nothing. Checked before step,
                        # because step() + zero_grad() erases
                        # the evidence.
                        if not any(f["A"].grad is not None or
                                   f["B"].grad is not None
                                   for f in factors):
                            raise SystemExit(
                                "[full] ABORT before first optimiser "
                                "step of this session: no LoRA factor "
                                "received a gradient. Do not continue; "
                                "fix grad flow before retrying.")
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            trainable, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    group_seconds = time.monotonic() - group_start_time
                    group_seconds_list.append(group_seconds)
                    opt_step += 1
                    # Position AFTER this optimiser step.
                    if index >= num_records:
                        next_position = (epoch + 1, 0)
                    else:
                        next_position = (epoch, index)
                    last_saved_position_candidate = next_position
                    # Per-optimiser-step quota log (group wall,
                    # excluding checkpoint save, like the pilot's
                    # per-step timing which excluded its save).
                    entry = {
                        "optimizer_step": opt_step,
                        "epoch": epoch,
                        "group_micro_steps": group_size,
                        "group_seconds": round(group_seconds, 3),
                        "seconds_per_micro_derived": round(
                            group_seconds / group_size, 3),
                        "loss_raw_last_micro": round(loss_value, 4),
                        "cumulative_seconds": round(
                            time.monotonic() - run_start, 3),
                        "timestamp": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    with quota_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry) + "\n")
                    print(f"[full] optimiser step {opt_step} / "
                          f"{total_opt_steps} (epoch {epoch}): "
                          f"group {group_seconds:.2f}s "
                          f"({group_seconds / group_size:.3f}s/micro), "
                          f"last-micro loss={loss_value:.4f}, "
                          f"lr={scheduler.get_last_lr()[0]:.2e}",
                          flush=True)
                    # Checkpoint triggers: every N steps, every
                    # 15 minutes, epoch end. (Chunk-end / run-end
                    # saves happen after the loop; see below.)
                    epoch_end = index >= num_records
                    time_due = (time.monotonic() - last_time_ckpt
                                >= args.checkpoint_every_seconds)
                    step_due = (opt_step % args.checkpoint_every_steps == 0)
                    if step_due or time_due or epoch_end:
                        state = build_training_state(
                            args, sizing, dataset_hash, opt_step,
                            next_position[0], next_position[1])
                        save_full_checkpoint(run_dir, opt_step, factors,
                                             optimizer, scheduler, state)
                        last_saved_opt_step = opt_step
                        last_saved_position = next_position
                        last_time_ckpt = time.monotonic()
                    else:
                        # Remember the last completed position even
                        # when no checkpoint was written, so a
                        # chunk-end save (below) records it.
                        last_saved_position = next_position
            if chunk_reason == "chunk-budget":
                break
            if index >= num_records:
                epoch += 1
                index = 0
    except KeyboardInterrupt:
        print("[full] interrupted (KeyboardInterrupt): saving a "
              "chunk-end checkpoint at the last completed optimiser "
              "step before exiting.", flush=True)
        interrupted = True
        chunk_reason = "interrupted"

    # --- Chunk-end / run-end checkpoint: NEVER lose a completed
    # optimiser step. If the last completed step has no checkpoint
    # yet (e.g. chunk ended at step 37 with no trigger), save one
    # now at the last completed position. Partial-group micros are
    # NOT in any checkpoint by design (they never stepped).
    if opt_step > last_saved_opt_step and opt_step > start_opt_step:
        # last_saved_position tracks the position after the last
        # completed optimiser step (updated on every step above),
        # even when no periodic checkpoint was written for it.
        state = build_training_state(
            args, sizing, dataset_hash, opt_step,
            last_saved_position[0], last_saved_position[1])
        save_full_checkpoint(run_dir, opt_step, factors,
                             optimizer, scheduler, state)
        last_saved_opt_step = opt_step
    elif opt_step == start_opt_step and resume_state is None \
            and opt_step == 0:
        print("[full] no optimiser step completed in this session; "
              "no checkpoint to save (resume would restart from "
              "step 0, which is correct only for a 0-step session).",
              flush=True)

    # --- Session summary + re-measurement (the first chunk's job).
    if group_seconds_list:
        derived = sorted(
            g / args.accumulation_steps for g in group_seconds_list)
        # Derived per-micro times use the group size actually
        # stepped; for partial epoch-end groups this slightly
        # mis-attributes optimiser-step overhead, which is why
        # the median below is labelled derived, not measured.
        median_derived = derived[len(derived) // 2]
        mean_group = sum(group_seconds_list) / len(group_seconds_list)
    else:
        median_derived = None
        mean_group = None
    total_elapsed = time.monotonic() - run_start
    run_complete = opt_step >= total_opt_steps
    if run_complete:
        chunk_reason = "completed"
        (run_dir / "RUN_COMPLETE.json").write_text(json.dumps({
            "optimizer_steps": opt_step,
            "total_optimizer_steps": total_opt_steps,
            "final_checkpoint": f"checkpoint-opt-{opt_step}",
            "total_elapsed_seconds": round(total_elapsed, 1),
        }, indent=2), encoding="utf-8")
        print(f"[full] RUN COMPLETE: {opt_step} / {total_opt_steps} "
              f"optimiser steps. Final adapter: "
              f"{run_dir / f'checkpoint-opt-{opt_step}'} "
              f"(adapter_config.json + adapter_model.safetensors). "
              f"On Kaggle, Save Version on /kaggle/working to make "
              f"this a versioned dataset output.", flush=True)
    summary = {
        "run_dir": str(run_dir),
        "reason": chunk_reason,
        "interrupted": interrupted,
        "optimizer_steps_start": start_opt_step,
        "optimizer_steps_end": opt_step,
        "optimizer_steps_this_session": opt_step - start_opt_step,
        "total_optimizer_steps": total_opt_steps,
        "last_checkpoint_opt_step": last_saved_opt_step,
        "last_checkpoint_position": last_saved_position,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "chunk_budget_seconds": chunk["chunk_budget_seconds"],
        "median_seconds_per_micro_derived": median_derived,
        "mean_group_seconds": mean_group,
        "sizing_assumption_seconds_per_micro": (
            args.measured_seconds_per_micro_step),
        "remeasure_warning": bool(
            median_derived is not None and median_derived >
            1.2 * args.measured_seconds_per_micro_step),
        "quota_log": str(quota_log_path),
    }
    (run_dir / "chunk_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== FULL-RUN CHUNK SUMMARY ===")
    print(f"reason: {chunk_reason}; optimiser steps: "
          f"{start_opt_step} -> {opt_step} / {total_opt_steps}")
    print(f"last checkpoint: checkpoint-opt-{last_saved_opt_step} "
          f"at position {last_saved_position}")
    print(f"total elapsed (incl. load): {total_elapsed:.1f}s; "
          f"chunk budget was {chunk['chunk_budget_seconds']:.0f}s")
    if median_derived is not None:
        print(f"median seconds/micro (DERIVED from group times, "
              f"this session): {median_derived:.3f} vs sizing "
              f"assumption {args.measured_seconds_per_micro_step:.3f}")
        if summary["remeasure_warning"]:
            print("[full] RE-MEASURE WARNING: this session's derived "
                  "median exceeds the sizing assumption by >20%. "
                  "Recompute later chunk budgets from THIS median "
                  "before launching chunk 2 -- do NOT reuse 0.339 s.")
        remaining_steps = total_opt_steps - opt_step
        if remaining_steps > 0:
            proj = (remaining_steps * args.accumulation_steps
                    * median_derived)
            print(f"projection (derived median only, NOT a promise): "
                  f"remaining {remaining_steps} optimiser steps ~= "
                  f"{proj:.0f}s stepping + load overhead. Recompute "
                  f"from the re-measured median and the CURRENT "
                  f"quota meter before chunk 2.")
    print(f"chunk summary: {run_dir / 'chunk_summary.json'}")
    print("This summary claims NO quality result, accuracy, or "
          "leaderboard outcome. Unproven until Kaggle: resume in a "
          "fresh session (run this script with --resume-from on "
          "the checkpoint above), and vLLM serving of the adapter "
          "(Gate 2, separate stream).")


if __name__ == "__main__":
    # Dry-run must not import GPU libraries: dispatch here, before
    # main() (which imports torch/transformers).
    _args = build_parser().parse_args()
    if _args.dry_run:
        try:
            _code = run_dry_run(_args)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"[full] DRY-RUN FAILED: {exc}", file=sys.stderr)
            sys.exit(2)
        sys.exit(_code)
    main()
