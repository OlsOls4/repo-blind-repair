"""Dev-set evaluation for Gemma 4 Repo-Blind Repair — Kaggle-notebook-ready.

WHAT IT MEASURES (and what it does not)
----------------------------------------
This script measures the TRAINED manual-LoRA adapter's repair quality on
the public dev set (129 tasks; fastapi / rich / requests / httpx only).
It is the quality measurement the Paper Track needs: fix rate with
sample size stated, broken down by bug class where known, regression
rate, diff sizes, and per-task time. It makes NO quality claim itself:
every number it prints is computed from the run it just performed, and
a good score here does NOT prove scorer serving — the scorer serves
the adapter with vLLM, which remains UNPROVEN (Gate 2 Stage 3 was
UNTESTED: vLLM is not installed in the Kaggle notebook environment).
Serving can only be proven by a real submission or the starter's own
validation environment. Do not quote a dev-set score as a leaderboard
result.

HOW IT WORKS
------------
1. Loads the declared 12B QAT base dequantized, exactly as the pilot
   does: CompressedTensorsConfig(run_compressed=False) so the loader
   materialises plain bf16 nn.Linear modules. Hard gate: every
   enumerated target module must carry .weight (328/328 on Gate 1);
   aborts loudly otherwise. Never silently evaluates a packed load.
2. Attaches a trained manual-LoRA adapter from --adapter-dir
   (PEFT-format adapter_config.json + adapter_model.safetensors,
   loaded with safetensors directly and injected with the pilot's
   forward-hook delta). NO PEFT import, NO torchao import anywhere:
   PEFT is unusable in the Kaggle environment (torchao 0.10.0 vs
   PEFT's >0.16.0 requirement), and reintroducing it would break
   this script the way it broke Gate 2's first run.
3. For each dev task, runs the harness scope ladder (rung 1 trace
   files, rung 2 imports + callers, rung 3 wider search) with the
   minimal-diff regression gate (max 15 tests / 90 s, abort on the
   first failure, atomic exact-once patch). Generation uses the
   TRAINED format (see FORMAT ALIGNMENT below): one greedy
   generation PER implicated source file, producing a full fixed
   file that is diffed into a patch. A patch counts as a FIX only
   if the gate accepts it (failing test passes AND the capped
   regression suite stays green).

FORMAT ALIGNMENT — the fix that makes measurement possible (1 Nov 2026)
-----------------------------------------------------------------------
Training taught exactly one text shape (``full_train_manual_lora.py``
``to_text_example``): "Fix this Python bug (bug class: <label>):\\n
<buggy_code>\\nFixed code:\\n<fixed_code>". The 1 Nov held-out pilot
failed 0/10 with a DIFFERENT eval format: a packed multi-file
context with "--- path (rung N) ---" headers, no instruction, no
"Fixed code:" cue, demanding FILE:/OLD:/NEW: patch blocks the model
was never trained to emit. Rung-1 generation ran the full 512
tokens and produced nothing parsable — the rung-2 empty-context
crash (fixed with the empty-rung skip below) was only the proximate
error; the format mismatch is the deeper reason no patch would
ever have parsed. This script now prompts EXACTLY as trained:

    Fix this Python bug:
    <implicated buggy file>
    Fixed code:

with ONE deliberate, documented distribution shift: the bug class
is omitted, because real tasks carry no label and inventing one
would be a fabricated input. Multi-file contexts are handled by
generating per implicated source file, one file per generation,
because training is single-file: the model is never asked to
repair a packed multi-file context. Rung 1's generation target is
the trace-implicated SOURCE file, never the test file (test files
are excluded as targets); rungs 2/3 are attempted only if every
rung-1 candidate is gate-rejected (or yields no usable diff), and
rung 3 is capped at --max-candidates-per-rung source files (the
wider search can otherwise be the whole repo; rungs 1/2 try every
implicated source file, bounded by the per-task budget). A
generated file that is empty, identical, or unparsable yields no
candidate — it is recorded, not rescued by heuristics and not
force-applied. This alignment does NOT prove real-bug transfer;
it removes the format barrier so transfer can finally be MEASURED.
4. Writes a JSON report (--out) with the summary, the per-bug-class
   breakdown, diff/time statistics, and one record per task.

ADAPTER-ONLY MEASUREMENT (read this before quoting a fix rate)
--------------------------------------------------------------
The harness inference agent also tries deterministic heuristic
one-line flips when no model patch parses. This evaluation
DELIBERATELY does not: candidates come from the adapter's model
output only. A heuristic fix would inflate the adapter's fix rate
and mislead the Paper Track. If the model output is unparsable at
every rung, the task is recorded as "no_patch", not rescued.

GENERATION (why it is simple, and what that costs)
--------------------------------------------------
Text is generated greedily by running the model body to hidden
states and computing logits manually with F.linear on the frozen
lm_head weight — the same manual-logits path the v3 pilot proved,
because the dequantized load's own lm_head forward detaches the
graph (training-side finding; under no_grad inference the values
are still computed, but the manual path avoids the module entirely
and keeps generation identical to the path the adapter was trained
with). There is NO KV cache: each new token recomputes the full
prompt. That is simple and correct, but slow for long outputs, so
--max-new-tokens defaults to 512 and --per-task-secs defaults to
600 s; a task that exceeds its budget is recorded as "timeout" and
the run continues. A KV-cached generator is future work, not a
silent assumption.

DEV-TASK FILE FORMAT (strict — mismatches fail loudly, never guessed)
----------------------------------------------------------------------
--dev-tasks is a JSONL file, one task per line, each line a JSON
object with these REQUIRED fields:
  task_id: str, unique across the file.
  files: object mapping relative repo paths -> file contents (str).
      Keys must be relative (no leading "/", no ".." component).
      This is the repo snapshot the harness patches in a temp dir.
  failing_test_id: str, a unittest dotted id, e.g.
      "tests.test_x.TestX.test_fails".
  regression_test_ids: list of unittest dotted ids (may be empty;
      an empty list means the gate can never accept — the task is
      then recorded as fixed=false with reason "no_regression_tests",
      not silently skipped).
OPTIONAL fields:
  bug_class: str (e.g. "off_by_one"); absent -> "unknown" in the
      breakdown. The dev set's bug classes are NOT known in advance
      (hidden real bugs), so "unknown" is expected for most tasks;
      the breakdown is only as fine as the labels supplied.
  issue_text: str, free-text issue description for the trace
      extractor (default "").
  test_output: str, failing-test output for the trace extractor.
      If absent, the failing test is run once (30 s timeout) to
      capture its output, exactly as the inference agent does.
Any line that is not a JSON object, misses a required field, has a
wrong type, duplicates a task_id, or carries an unsafe file path
aborts the whole run with the line number — a half-validated task
file must never produce a half-credible report.

KAGGLE RUN (read before running)
---------------------------------
- Attach the model, this script, the dev-tasks file, and the SAVED
  adapter (from the committed retrain version's output) as notebook
  inputs. Install the loguru + compressed_tensors wheels offline
  first (same as the pilot: pip install <local wheel> --no-deps
  --quiet --no-index, loguru first).
- Set MODEL_PATH to the attached local model path
  (/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2
  in the 31 Oct runs) and --adapter-dir to the saved adapter dir.
- Run with internet OFF. Save the notebook VERSION with output
  immediately after the run: the 31 Oct 20k adapter was LOST because
  it lived only in a non-persistent interactive working directory.
  A report that exists only in a working directory does not exist.
- Quota: 129 tasks at up to 600 s each is far beyond one session if
  every task times out; use --max-tasks for a pilot slice first and
  size the full run from measured per-task times.

CONSTRAINTS (do not remove)
----------------------------
- No teacher models, no paid APIs, no network calls, no GCP services.
- No real-fix training data is used or needed here: evaluation runs
  the adapter on dev tasks; it does not train on them.
- No PEFT import, no torchao import, no version spoof, no blocker.
- Heavy imports (torch, transformers, harness) live inside functions
  so --dry-run and py_compile work anywhere without a GPU.

RUN ON KAGGLE ONLY for a real evaluation. --dry-run runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Declared base only. Do NOT substitute a different base model.
MODEL_HANDLE = "google/gemma-4-12b-it-qat-w4a16-ct"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
EXPECTED_GATE_COUNT = 328  # Gate 1 measured exactly 328 targets.


def _check_prompt_token_length(input_ids) -> None:
    """Refuse a 0-length token sequence before it reaches the body.

    A 0-length sequence reaches the body's attention as a tensor with
    a 0 sequence dimension, and the attention reshape (batch, seq,
    heads, head_dim with an inferred dimension) fails with the cryptic
    "cannot reshape tensor of 0 elements into shape [1, 0, -1, 256]".
    That error is a symptom of an empty prompt, not a model failure:
    the Gemma tokenizer returns 0 tokens for an empty prompt string,
    and the generation loop only ever appends tokens, so a 0-length
    sequence can only occur on the FIRST forward. Raising here names
    the real cause. Callers that can skip the rung (evaluate_task)
    catch this ValueError and record an empty_prompt skip; any other
    caller sees a loud failure instead of a reshape traceback.
    The check uses only .shape, so it also works with stub tensors
    in unit tests (no torch import needed here).
    """
    try:
        seq_len = input_ids.shape[1]
    except (AttributeError, IndexError, TypeError):
        return  # Shape unknowable: let the body try (old behaviour).
    if seq_len == 0:
        raise ValueError(
            "empty prompt: tokenizer produced 0 tokens; refusing to run "
            "the body on a 0-length sequence (inside attention this "
            "surfaces as a reshape error on a 0-element tensor)")


# ---------------------------------------------------------------------------
# Dev-task format: strict parsing, loud failures
# ---------------------------------------------------------------------------

@dataclass
class DevTask:
    task_id: str
    files: dict[str, str]
    failing_test_id: str
    regression_test_ids: list[str]
    bug_class: str = "unknown"
    issue_text: str = ""
    test_output: str = ""


def _is_safe_rel_path(path: str) -> bool:
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    parts = path.replace("\\", "/").split("/")
    return all(p not in ("", ".", "..") for p in parts)


def load_dev_tasks(path: Path) -> list[DevTask]:
    """Parse and strictly validate a dev-tasks JSONL file.

    Raises SystemExit with the offending line number on any mismatch.
    """
    tasks: list[DevTask] = []
    seen_ids: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"[eval] ABORT: dev-tasks line {lineno} is not valid JSON: {exc}")
        if not isinstance(obj, dict):
            raise SystemExit(
                f"[eval] ABORT: dev-tasks line {lineno} is not a JSON object")
        for req in ("task_id", "files", "failing_test_id", "regression_test_ids"):
            if req not in obj:
                raise SystemExit(
                    f"[eval] ABORT: dev-tasks line {lineno} misses required "
                    f"field {req!r}")
        task_id = obj["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise SystemExit(
                f"[eval] ABORT: dev-tasks line {lineno}: task_id must be a "
                f"non-empty string")
        if task_id in seen_ids:
            raise SystemExit(
                f"[eval] ABORT: dev-tasks line {lineno}: duplicate task_id "
                f"{task_id!r}")
        seen_ids.add(task_id)
        files = obj["files"]
        if not isinstance(files, dict) or not files:
            raise SystemExit(
                f"[eval] ABORT: task {task_id} (line {lineno}): files must be "
                f"a non-empty object mapping relative paths to contents")
        for fpath, content in files.items():
            if not isinstance(fpath, str) or not _is_safe_rel_path(fpath):
                raise SystemExit(
                    f"[eval] ABORT: task {task_id} (line {lineno}): unsafe "
                    f"file path {fpath!r} (must be relative, no '..')")
            if not isinstance(content, str):
                raise SystemExit(
                    f"[eval] ABORT: task {task_id} (line {lineno}): content "
                    f"of {fpath!r} must be a string")
        failing = obj["failing_test_id"]
        if not isinstance(failing, str) or not failing:
            raise SystemExit(
                f"[eval] ABORT: task {task_id} (line {lineno}): "
                f"failing_test_id must be a non-empty string")
        regression = obj["regression_test_ids"]
        if not isinstance(regression, list) or not all(
                isinstance(t, str) and t for t in regression):
            raise SystemExit(
                f"[eval] ABORT: task {task_id} (line {lineno}): "
                f"regression_test_ids must be a list of non-empty strings")
        bug_class = obj.get("bug_class", "unknown")
        if not isinstance(bug_class, str) or not bug_class:
            raise SystemExit(
                f"[eval] ABORT: task {task_id} (line {lineno}): bug_class "
                f"must be a non-empty string when present")
        issue_text = obj.get("issue_text", "")
        test_output = obj.get("test_output", "")
        if not isinstance(issue_text, str) or not isinstance(test_output, str):
            raise SystemExit(
                f"[eval] ABORT: task {task_id} (line {lineno}): issue_text "
                f"and test_output must be strings when present")
        tasks.append(DevTask(
            task_id=task_id, files=files, failing_test_id=failing,
            regression_test_ids=regression, bug_class=bug_class,
            issue_text=issue_text, test_output=test_output))
    if not tasks:
        raise SystemExit("[eval] ABORT: dev-tasks file contains no tasks")
    return tasks


def validate_adapter_dir(adapter_dir: Path) -> dict:
    """Check an adapter dir's files exist and its config parses.

    Does NOT load any tensor: dry-run must stay model-free.
    """
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise SystemExit(
            f"[eval] ABORT: adapter dir {adapter_dir} has no adapter_config.json")
    if not weights_path.is_file():
        raise SystemExit(
            f"[eval] ABORT: adapter dir {adapter_dir} has no "
            f"adapter_model.safetensors")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"[eval] ABORT: adapter config unparsable: {exc}")
    if config.get("peft_type") != "LORA":
        raise SystemExit(
            f"[eval] ABORT: adapter config peft_type is "
            f"{config.get('peft_type')!r}, expected 'LORA'")
    return config


# ---------------------------------------------------------------------------
# Model + adapter (full run only; heavy imports live here)
# ---------------------------------------------------------------------------

def load_model_and_adapter(model_path: str, adapter_dir: Path | None = None,
                           attach_adapter: bool = True):
    """Load the dequantized base, gate it, and attach the saved adapter.

    Returns (model, tokenizer, generate_fn) where generate_fn(prompt)
    -> str is greedy decoding via the body + manual lm_head logits.
    Raises SystemExit on any gate failure — never evaluate a packed
    load or a half-attached adapter.

    With attach_adapter=False (the --no-adapter baseline mode, added
    1 Nov 2026), no adapter is attached and no adapter dir is needed:
    the base model alone generates, so a base-model baseline can be
    measured on IDENTICAL tasks, prompts, gates, and report shape as
    an adapter run. Everything else (dequantization gate, body /
    lm_head resolution, greedy decode, EOS stop) is unchanged.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] loading base from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        from transformers import CompressedTensorsConfig
        quant_config = CompressedTensorsConfig(run_compressed=False)
    except Exception:
        quant_config = {"run_compressed": False}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_config,
    )
    model.eval()

    # Dequantization gate (same as the pilot): every target must
    # carry .weight. A packed load has weight_packed/scale/shape
    # instead and must never reach evaluation.
    total = with_weight = 0
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in TARGET_MODULES:
            total += 1
            with_weight += 1 if hasattr(module, "weight") else 0
    print(f"[eval] dequantization gate: targets with .weight = "
          f"{with_weight}/{total}", flush=True)
    if total == 0 or with_weight != total:
        raise SystemExit(
            f"[eval] ABORT: dequantized load failed — only "
            f"{with_weight}/{total} targets have .weight. Do not "
            f"evaluate on a packed load.")
    if total != EXPECTED_GATE_COUNT:
        print(f"[eval] WARNING: expected {EXPECTED_GATE_COUNT} target "
              f"modules (Gate 1), found {total}; proceeding only because "
              f"every found target has .weight.", flush=True)

    # Body + lm_head resolution (same structure the v3 pilot verified
    # against transformers 5.16.1): outer model has .model (body
    # returning last_hidden_state) and .lm_head (bias-free Linear
    # tied to embed_tokens). The lm_head module is never called:
    # logits come from F.linear on its frozen weight, so whatever
    # detaches inside its forward cannot reach generation.
    body = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if body is None or lm_head is None or not hasattr(lm_head, "weight"):
        raise SystemExit(
            "[eval] ABORT: loaded model lacks .model body or a weight-"
            "carrying .lm_head; refusing to guess the structure.")
    for param in model.parameters():
        param.requires_grad_(False)

    if attach_adapter:
        if adapter_dir is None:
            raise SystemExit(
                "[eval] ABORT: attach_adapter=True but no adapter dir "
                "given; pass --adapter-dir, or --no-adapter for the "
                "base-model baseline.")
        attached = _attach_saved_manual_adapter(model, adapter_dir)
        print(f"[eval] attached {attached} saved adapter modules as "
              f"forward hooks", flush=True)
    else:
        print("[eval] no-adapter baseline: base model only, no "
              "adapter hooks attached", flush=True)

    lm_device = lm_head.weight.device
    eos_id = tokenizer.eos_token_id

    def generate_fn(prompt: str, max_new_tokens: int) -> str:
        """Greedy decode, no KV cache (full recompute per token)."""
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        # Shard-aware: move inputs to the body's first device.
        try:
            first_device = next(body.parameters()).device
        except StopIteration:
            first_device = lm_device
        input_ids = input_ids.to(first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(first_device)
        # Guard: an empty prompt tokenizes to 0 tokens, which fails
        # inside the body with a cryptic reshape error (see
        # _check_prompt_token_length). Fail loudly here instead.
        _check_prompt_token_length(input_ids)
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = body(input_ids=input_ids,
                               attention_mask=attention_mask)
                hidden = outputs.last_hidden_state \
                    if hasattr(outputs, "last_hidden_state") else outputs[0]
                last = hidden[:, -1, :]
                if last.device != lm_device:
                    last = last.to(lm_device)
                logits = F.linear(last, lm_head.weight,
                                  getattr(lm_head, "bias", None))
                nxt = int(torch.argmax(logits, dim=-1).item())
                generated.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
                nxt_tensor = torch.tensor([[nxt]], device=first_device)
                input_ids = torch.cat([input_ids, nxt_tensor], dim=1)
                if attention_mask is not None:
                    ones = torch.ones((attention_mask.shape[0], 1),
                                      device=first_device,
                                      dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, ones], dim=1)
        return tokenizer.decode(generated, skip_special_tokens=True)

    return model, tokenizer, generate_fn


def _attach_saved_manual_adapter(model, adapter_dir: Path) -> int:
    """Attach a saved PEFT-format adapter via forward hooks.

    Same key convention as the pilot's save_adapter and Gate 2's
    fallback loader: keys are
    ``base_model.model.<module path>.lora_A/.lora_B.default.weight``.
    Delta = (x @ A.T @ B.T) * (alpha / r), added to the module
    output. No PEFT library is involved. Raises on any inconsistency
    (missing file, missing A or B factor, unknown target, or a
    target without .weight on the loaded model).
    """
    import torch
    from safetensors.torch import load_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text(
        encoding="utf-8"))
    scaling = float(config.get("lora_alpha", 16)) / float(config.get("r", 8))
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    prefix = "base_model.model."
    grouped: dict[str, dict[str, object]] = {}
    for key, tensor in tensors.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        if rest.endswith(".lora_A.default.weight"):
            path, factor = rest[:-len(".lora_A.default.weight")], "A"
        elif rest.endswith(".lora_B.default.weight"):
            path, factor = rest[:-len(".lora_B.default.weight")], "B"
        else:
            continue
        grouped.setdefault(path, {})[factor] = tensor
    if not grouped:
        raise SystemExit(
            f"[eval] ABORT: no adapter tensors found in {adapter_dir} "
            f"(expected PEFT-compatible lora_A/lora_B keys)")
    modules = dict(model.named_modules())
    for path, parts in sorted(grouped.items()):
        if "A" not in parts or "B" not in parts:
            raise SystemExit(
                f"[eval] ABORT: adapter target {path} is missing an A "
                f"or B factor")
        module = modules.get(path)
        if module is None:
            raise SystemExit(
                f"[eval] ABORT: adapter target {path} not found in the "
                f"loaded model")
        if not hasattr(module, "weight"):
            raise SystemExit(
                f"[eval] ABORT: adapter target {path} has no .weight "
                f"on the loaded model; cannot place factors")

        def make_hook(a, b, scale):
            def hook(_mod, inputs, output):
                if not isinstance(output, torch.Tensor) or not inputs:
                    raise TypeError(
                        "[eval] adapter hook expected a Tensor output "
                        "and a non-empty input")
                delta = torch.matmul(
                    torch.matmul(inputs[0], a.t()), b.t()) * scale
                return output + delta.to(dtype=output.dtype)
            return hook

        a_mat = parts["A"].to(device=module.weight.device,
                              dtype=module.weight.dtype)
        b_mat = parts["B"].to(device=module.weight.device,
                              dtype=module.weight.dtype)
        module.register_forward_hook(make_hook(a_mat, b_mat, scaling))
    return len(grouped)


# ---------------------------------------------------------------------------
# Per-task evaluation (harness ladder + gate, adapter-only candidates)
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    bug_class: str
    status: str  # fixed | no_patch | gate_rejected | error | timeout | no_regression_tests
    # Note: an empty rung is NOT a status: it is skipped and recorded
    # in reason as empty_context/empty_prompt, and later rungs still run.
    rung: int | None = None
    diff_lines: int = 0
    files_changed: int = 0
    time_secs: float = 0.0
    candidates_tried: int = 0
    reason: str = ""
    failing_passed_but_regressed: bool = False


def evaluate_task(task: DevTask, generate_fn, harness_dir: str,
                  max_candidates_per_rung: int, per_task_secs: float,
                  max_new_tokens: int, token_budget: int) -> TaskResult:
    """Evaluate one task. Never raises: failures become TaskResults.

    ``token_budget`` is accepted for CLI compatibility but no longer
    used: prompts are single source files (trained format), not
    packed multi-file contexts, so there is nothing to pack under a
    budget. An overlong single file is a known limit (generation
    truncation yields no candidate), not a packing decision.
    """
    import sys as _sys
    if harness_dir not in _sys.path:
        _sys.path.insert(0, harness_dir)
    from patcher_stub import (  # noqa: E402
        build_trained_format_prompt, fixed_file_to_edits, is_test_file)
    from regression_gate import evaluate_patch  # noqa: E402
    from scope_ladder import discover_rungs  # noqa: E402
    from test_runner import run_one_test  # noqa: E402
    from trace_extractor import extract_trace  # noqa: E402

    start = time.monotonic()
    if not task.regression_test_ids:
        return TaskResult(task_id=task.task_id, bug_class=task.bug_class,
                          status="no_regression_tests", time_secs=0.0,
                          reason="no regression tests supplied; gate can "
                                 "never accept")
    tmpdir = tempfile.mkdtemp(prefix="eval_dev_task_")
    try:
        for rel, content in task.files.items():
            full = Path(tmpdir) / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
        output = task.test_output
        if not output:
            output = run_one_test(task.failing_test_id, tmpdir).output
        trace = extract_trace(output, task.issue_text)
        rungs = discover_rungs(tmpdir, trace)
        all_files: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(tmpdir):
            dirnames[:] = sorted(d for d in dirnames
                                 if not d.startswith(".")
                                 and d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith(".py"):
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, tmpdir)
                    try:
                        all_files[rel] = Path(full).read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
        tried = 0
        regressed = False
        empty_rungs: list[int] = []
        empty_notes: list[str] = []
        reason_notes: list[str] = []
        last_reason = "no usable model diff at any rung"
        for rung in (1, 2, 3):
            if time.monotonic() - start > per_task_secs:
                return TaskResult(
                    task_id=task.task_id, bug_class=task.bug_class,
                    status="timeout", time_secs=time.monotonic() - start,
                    candidates_tried=tried, reason="per-task budget exceeded")
            rung_files = [p for p in rungs[rung] if p in all_files]
            # Empty-rung guard (diagnosed 1 Nov 2026): on small repos
            # a rung can legitimately contain no files — e.g. the
            # held-out tasks have only task.py + tests, and the real
            # traceback puts both into rung 1, leaving rung 2 empty.
            # That is an empty context, not a model failure, so skip
            # the rung and keep climbing instead of erroring the task
            # (later rungs must still be attempted). Recorded in the
            # reason as empty_context; it is not a separate status.
            # (The old failure mode — generating on an empty packed
            # context whose 0 tokens crashed attention with a reshape
            # error — cannot recur: generation targets are single
            # non-empty source files, and the prompt header alone
            # already tokenizes to non-zero length.)
            if not rung_files:
                empty_rungs.append(rung)
                last_reason = (f"rung {rung}: empty_context (no files in "
                               f"rung; skipped, not generated)")
                empty_notes.append(last_reason)
                reason_notes.append(last_reason)
                continue
            # Rung 1's target is the trace-implicated SOURCE file,
            # never the test file: a test exercises the bug, it is
            # not the bug. A rung with only test files is skipped
            # the same way as an empty rung.
            source_files = [p for p in rung_files
                            if not is_test_file(p)]
            if not source_files:
                empty_rungs.append(rung)
                last_reason = (f"rung {rung}: no_source_files (rung "
                               f"contains only test files; skipped, "
                               f"not generated)")
                empty_notes.append(last_reason)
                reason_notes.append(last_reason)
                continue
            # Rung 3 is the wider search (potentially the whole
            # repo), so it is capped at max_candidates_per_rung
            # source files; rungs 1/2 try every implicated source
            # file, bounded by the per-task budget. One generation
            # per file: training is single-file.
            files_to_try = source_files
            if rung == 3 and len(source_files) > max_candidates_per_rung:
                files_to_try = source_files[:max(1, max_candidates_per_rung)]
                empty_notes.append(
                    f"rung 3: capped to {len(files_to_try)} of "
                    f"{len(source_files)} source files")
            for path in files_to_try:
                if time.monotonic() - start > per_task_secs:
                    return TaskResult(
                        task_id=task.task_id, bug_class=task.bug_class,
                        status="timeout",
                        time_secs=time.monotonic() - start,
                        candidates_tried=tried,
                        reason="per-task budget exceeded")
                content = all_files[path]
                if not content.strip():
                    note = (f"rung {rung} {path}: empty_file (blank "
                            f"source; skipped, not generated)")
                    empty_notes.append(note)
                    reason_notes.append(note)
                    last_reason = note
                    continue
                prompt = build_trained_format_prompt(content)
                try:
                    raw = generate_fn(prompt, max_new_tokens)
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    # Defence in depth: a prompt that still
                    # tokenizes to 0 tokens raises the loud
                    # empty-prompt ValueError from
                    # _check_prompt_token_length. Treat that the
                    # same way as an empty file: skip this file.
                    if isinstance(exc, ValueError) and \
                            "empty prompt" in str(exc):
                        note = (f"rung {rung} {path}: empty_prompt "
                                f"(tokenizer produced 0 tokens; skipped)")
                        empty_notes.append(note)
                        reason_notes.append(note)
                        last_reason = note
                        continue
                    return TaskResult(
                        task_id=task.task_id, bug_class=task.bug_class,
                        status="error", time_secs=time.monotonic() - start,
                        candidates_tried=tried,
                        reason=f"generation failed at rung {rung} "
                               f"({path}): "
                               f"{type(exc).__name__}: {exc}")
                patch = fixed_file_to_edits(path, content, raw)
                if patch is None:
                    last_reason = (
                        f"rung {rung} {path}: no usable diff "
                        f"(generation empty, identical, or unparsable)")
                    reason_notes.append(last_reason)
                    continue
                tried += 1
                gate = evaluate_patch(tmpdir, patch, task.failing_test_id,
                                      task.regression_test_ids)
                if gate.failing_test_passed and not gate.accepted:
                    regressed = True
                if gate.accepted:
                    return TaskResult(
                        task_id=task.task_id, bug_class=task.bug_class,
                        status="fixed", rung=rung,
                        diff_lines=gate.diff_size,
                        files_changed=gate.files_changed,
                        time_secs=time.monotonic() - start,
                        candidates_tried=tried,
                        reason=f"accepted at rung {rung} ({path}): "
                               f"{gate.reason}")
                last_reason = (f"rung {rung} ({path}) rejected: "
                               f"{gate.reason}")
                reason_notes.append(last_reason)
        if reason_notes:
            # Join EVERY recorded outcome, in order, deduplicated.
            # Earlier versions kept only skip notes (or only the last
            # reason), which hid rung-1 gate rejections behind later
            # rung-2/3 skips in the pilot reports.
            last_reason = "; ".join(dict.fromkeys(reason_notes))
        return TaskResult(
            task_id=task.task_id, bug_class=task.bug_class,
            status="gate_rejected" if tried else "no_patch",
            time_secs=time.monotonic() - start, candidates_tried=tried,
            reason=last_reason, failing_passed_but_regressed=regressed)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return TaskResult(
            task_id=task.task_id, bug_class=task.bug_class,
            status="error", time_secs=time.monotonic() - start,
            reason=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _wilson_interval(fixed: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = fixed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def build_report(results: list[TaskResult], meta: dict) -> dict:
    n = len(results)
    fixed = [r for r in results if r.status == "fixed"]
    by_class: dict[str, dict] = {}
    for r in results:
        entry = by_class.setdefault(r.bug_class, {"n": 0, "fixed": 0})
        entry["n"] += 1
        if r.status == "fixed":
            entry["fixed"] += 1
    for entry in by_class.values():
        entry["fix_rate"] = entry["fixed"] / entry["n"] if entry["n"] else 0.0
    fixed_passing = sum(1 for r in results if r.failing_passed_but_regressed)
    denom = len(fixed) + fixed_passing
    diffs = sorted(r.diff_lines for r in fixed)
    times = sorted(r.time_secs for r in results)

    def median(values: list[float]) -> float:
        if not values:
            return 0.0
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    lo, hi = _wilson_interval(len(fixed), n)
    return {
        "meta": meta,
        "summary": {
            "n_tasks": n,
            "n_fixed": len(fixed),
            "fix_rate": len(fixed) / n if n else 0.0,
            "fix_rate_wilson_95": [lo, hi],
            "status_counts": {s: sum(1 for r in results if r.status == s)
                              for s in sorted({r.status for r in results})},
            # Regression rate: among candidates that fixed the failing
            # test, the share that broke >=1 regression test and were
            # therefore rejected by the gate. Accepted patches have
            # zero regressions BY CONSTRUCTION (the gate requires it),
            # so reporting 0% there would be vacuous; this is the
            # honest denominator.
            "regression_rate": fixed_passing / denom if denom else 0.0,
            "regression_breaks": fixed_passing,
            "regression_denominator": denom,
            "diff_lines_median_fixed": median([float(d) for d in diffs]),
            "diff_lines_mean_fixed": (sum(diffs) / len(diffs)) if diffs else 0.0,
            "time_secs_median": median(times),
            "time_secs_mean": (sum(times) / len(times)) if times else 0.0,
            "time_secs_max": max(times) if times else 0.0,
        },
        "by_bug_class": by_class,
        "tasks": [
            {
                "task_id": r.task_id, "bug_class": r.bug_class,
                "status": r.status, "rung": r.rung,
                "diff_lines": r.diff_lines, "files_changed": r.files_changed,
                "time_secs": round(r.time_secs, 2),
                "candidates_tried": r.candidates_tried,
                "reason": r.reason,
            }
            for r in results
        ],
        "caveats": [
            "Fix = regression-gate acceptance only (failing test passes "
            "AND capped regression suite green). No exact-patch matching.",
            "Candidates are adapter model output ONLY; deterministic "
            "heuristic flips are disabled, so this fix rate is the "
            "adapter's, not the harness's.",
            "A good score here does NOT prove scorer serving: the "
            "scorer serves the adapter with vLLM, which is UNPROVEN "
            "(Gate 2 Stage 3 UNTESTED). Serving is proven only by a "
            "real submission or the starter's validation environment.",
            "Dev-set bug classes are hidden real bugs; bug_class is "
            "'unknown' unless the task file labels it.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dev-set evaluation for the trained manual-LoRA adapter.")
    parser.add_argument("--dev-tasks", required=True,
                        help="JSONL dev-task file (see module docstring)")
    parser.add_argument("--adapter-dir", default="",
                        help="dir with adapter_config.json + "
                             "adapter_model.safetensors (required "
                             "unless --no-adapter)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Base-model baseline mode (added 1 Nov "
                             "2026): skip adapter validation and "
                             "attachment entirely and evaluate the "
                             "base model alone on identical tasks, "
                             "prompts, gates, and report shape. "
                             "Must NOT be combined with --adapter-dir.")
    parser.add_argument("--model-path", default="",
                        help="base model path; default MODEL_PATH env or "
                             "the declared handle")
    parser.add_argument("--harness-dir", default="",
                        help="harness/ dir; default: <repo>/harness relative "
                             "to this script's parent")
    parser.add_argument("--out", default="eval_dev_report.json",
                        help="JSON report output path")
    parser.add_argument("--task-ids", default="",
                        help="Comma-separated task ids to run (subset of "
                             "the tasks file, in file order). Added 1 Nov "
                             "2026 so pilots can be stratified by bug class "
                             "instead of taking the file's first N tasks, "
                             "which are compound-heavy by construction.")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="evaluate at most this many tasks (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--per-task-secs", type=float, default=600.0)
    parser.add_argument("--max-candidates-per-rung", type=int, default=1,
                        help="max source files tried at rung 3 (the wider "
                             "search); rungs 1/2 try every implicated "
                             "source file, bounded by --per-task-secs. "
                             "Heuristics are disabled in this evaluation")
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate arguments and task file, print the "
                             "planned task count, and exit WITHOUT loading "
                             "a model")
    args = parser.parse_args(argv)

    dev_path = Path(args.dev_tasks)
    if not dev_path.is_file():
        raise SystemExit(f"[eval] ABORT: dev-tasks file not found: {dev_path}")
    tasks = load_dev_tasks(dev_path)
    if args.task_ids:
        wanted = {t.strip() for t in args.task_ids.split(",") if t.strip()}
        missing = wanted - {t.task_id for t in tasks}
        if missing:
            raise SystemExit(f"[eval] ABORT: task ids not in file: "
                             f"{sorted(missing)}")
        tasks = [t for t in tasks if t.task_id in wanted]
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    if args.no_adapter and args.adapter_dir:
        raise SystemExit("[eval] ABORT: --no-adapter combined with "
                         "--adapter-dir; pass one or the other, not "
                         "both.")
    if args.no_adapter:
        adapter_dir = None
        adapter_config = {"adapter": "none",
                          "note": "base-model baseline: no adapter "
                                  "attached (--no-adapter)"}
    else:
        if not args.adapter_dir:
            raise SystemExit("[eval] ABORT: --adapter-dir is required "
                             "unless --no-adapter is given.")
        adapter_dir = Path(args.adapter_dir)
        adapter_config = validate_adapter_dir(adapter_dir)
    if args.dry_run:
        by_class: dict[str, int] = {}
        for task in tasks:
            by_class[task.bug_class] = by_class.get(task.bug_class, 0) + 1
        print(f"[eval] DRY RUN: {len(tasks)} task(s) planned "
              f"(from {dev_path})")
        print(f"[eval] DRY RUN: by bug_class: {by_class}")
        # Regression visibility (1 Nov 2026, regression-derivation
        # fix): a converted dev file used to carry empty regression
        # lists, making every task no_regression_tests and the fix
        # rate 0 by construction. The dry run now states the
        # regression coverage outright, so a bad conversion is
        # visible before any GPU is spent.
        n_fail = sum(1 for t in tasks if t.failing_test_id)
        n_reg = sum(1 for t in tasks if t.regression_test_ids)
        total_reg = sum(len(t.regression_test_ids) for t in tasks)
        print(f"[eval] DRY RUN: failing ids present for "
              f"{n_fail}/{len(tasks)} tasks; regression lists "
              f"non-empty for {n_reg}/{len(tasks)} tasks "
              f"(total {total_reg} regression ids)")
        for t in tasks:
            print(f"[eval] DRY RUN: {t.task_id}: failing="
                  f"{t.failing_test_id} regression="
                  f"{len(t.regression_test_ids)}")
        if args.no_adapter:
            print("[eval] DRY RUN: no-adapter baseline (base model "
                  "only, no adapter dir required)")
        else:
            print(f"[eval] DRY RUN: adapter dir OK: {adapter_dir} "
                  f"(r={adapter_config.get('r')}, "
                  f"alpha={adapter_config.get('lora_alpha')})")
        print("[eval] DRY RUN: no model loaded, no task evaluated")
        return 0

    model_path = args.model_path or os.environ.get("MODEL_PATH") or MODEL_HANDLE
    harness_dir = args.harness_dir or str(
        Path(__file__).resolve().parent.parent / "harness")
    if not Path(harness_dir).is_dir():
        raise SystemExit(f"[eval] ABORT: harness dir not found: {harness_dir}")
    _, _, generate_fn = load_model_and_adapter(
        model_path, adapter_dir, attach_adapter=not args.no_adapter)

    def model_generate(prompt: str, max_tokens: int) -> str:
        return generate_fn(prompt, max_tokens)

    results: list[TaskResult] = []
    run_start = time.monotonic()
    for i, task in enumerate(tasks, start=1):
        print(f"[eval] task {i}/{len(tasks)}: {task.task_id} "
              f"(bug_class={task.bug_class})", flush=True)
        result = evaluate_task(
            task, model_generate, harness_dir,
            args.max_candidates_per_rung, args.per_task_secs,
            args.max_new_tokens, args.token_budget)
        results.append(result)
        print(f"[eval]   -> {result.status} "
              f"(rung={result.rung}, {result.time_secs:.1f}s): "
              f"{result.reason}", flush=True)
    meta = {
        "adapter_dir": ("none (base-model baseline --no-adapter)"
                        if adapter_dir is None else str(adapter_dir)),
        "adapter_config": adapter_config,
        "model_path": model_path,
        "dev_tasks": str(dev_path),
        "n_tasks_planned": len(tasks),
        "total_secs": round(time.monotonic() - run_start, 1),
        "max_new_tokens": args.max_new_tokens,
        "per_task_secs": args.per_task_secs,
        "token_budget": args.token_budget,
        "note": "Dev-set quality measurement only. Does NOT prove vLLM "
                "scorer serving (Gate 2 Stage 3 UNTESTED).",
    }
    report = build_report(results, meta)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(f"[eval] DONE: fixed {summary['n_fixed']}/{summary['n_tasks']} "
          f"(fix_rate={summary['fix_rate']:.3f}, Wilson 95% "
          f"{summary['fix_rate_wilson_95'][0]:.3f}-"
          f"{summary['fix_rate_wilson_95'][1]:.3f}); "
          f"report written to {out_path}", flush=True)
    print("[eval] SAVE THE REPORT AND THE ADAPTER VERSION NOW: a report "
          "or adapter that exists only in a non-persistent working "
          "directory does not exist.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
