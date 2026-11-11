# Repo-Blind Repair — Harness

Offline, stdlib-only Python harness for the Gemma 4 Developer Agent
Competition entry "Repo-Blind Repair": a Gemma 4 LoRA trained only on
synthetic, deterministically injected bugs, repairing hidden Python bugs
inside a scope ladder with a minimal-diff regression gate.

**No Kaggle joining, no training, no paid anything happens here.**
No paid API or teacher model is ever called. `patcher_stub.generate_patch`
is the inference-only candidate generator (first candidate through the
regression gate; the full loop is `inference_agent.run_inference_only`).
It is not a placeholder, and it is not the trained LoRA: no trained
adapter exists (v1 0/18 held-out, v19 0/32 on the pre-registered kill
test, 7 Nov 2026).

## Layout

| File | What it does |
|---|---|
| `trace_extractor.py` | Parse failing test output + issue text → failing tests, traceback frames, implicated files/symbols. |
| `context_packer.py` | Pack rung-ordered file contents under a token budget (tokens ≈ chars/4, default cap 32,768), truncating/dropping with markers. |
| `test_runner.py` | Run unittest ids in subprocesses with hard timeouts; capped regression suite (default max 15 tests / 90 s, abort on first failure). |
| `scope_ladder.py` | Ladder controller: rung 1 = trace files, rung 2 = imports/callers (AST), rung 3 = wider repo search; each rung time-boxed (30/60/120 s defaults). |
| `regression_gate.py` | Accept a patch only if the failing test passes AND the capped regression suite stays green; otherwise roll back. Reports diff size (lines/files). |
| `patcher_stub.py` | `FileEdit` (exact old→new text), apply/rollback, and `generate_patch` — inference-only candidate generator (first candidate, or `None` when there is none; full loop in `inference_agent.py`). |
| `demo.py` | Runnable demo against a temp copy of `demo_repo/`. |
| `demo_repo/` | Toy buggy repo: `calculator.add` subtracts instead of adding; `test_add` fails, `test_multiply`/`test_subtract` pass. |
| `tests/test_harness.py` | Unit tests for all of the above (stdlib unittest). |

## Interface

```python
from trace_extractor import extract_trace
from scope_ladder import run_ladder

trace = extract_trace(failing_test_output, issue_text)
result = run_ladder(
    repo_root, trace,
    failing_test_id="tests.test_x.TestX.test_fails",
    regression_test_ids=["tests.test_x.TestX.test_ok"],
    patch_fn=my_patch_fn,   # (context_text, rung) -> list[FileEdit] | None
)
# result.accepted, result.rung, result.gate.diff_size,
# result.rung_files, result.packs, result.rung_secs
```

`patch_fn` signature: `(context_text: str, rung: int) -> list[FileEdit] | None`.
Return `None` for "no patch at this rung". Pass your own `patch_fn`
(as `demo.py` does), or use `patcher_stub.generate_patch` — it returns
`None` when it has no candidate, so a missing model is never silently
treated as "no patch needed". (History: `generate_patch` used to raise
`NotImplementedError` as a placeholder for the future trained LoRA
generator; it no longer does — see `patcher_stub.py`.)

Patch application rule: `old_text` must occur **exactly once** in the
file, otherwise the patch is rejected atomically (nothing half-applied).
This is why demo/test patches anchor on the `def add(...)` line rather
than the bare `return a - b` line, which also appears in `subtract`.

## Run the demo

From `harness/`:

```bash
python3 demo.py
```

Expected output: failing test fails before, the ladder accepts a
one-line patch at rung 1 (diff +1/−1, 1 file), and the failing test
passes after. The demo runs on a temp copy, so `demo_repo/` stays buggy.

## Run the tests

From `harness/`:

```bash
python3 -m unittest discover -s tests -v
```

## Inference-only agent (STREAM 4, no adapter required)

`patcher_stub.generate_patch` is no longer a placeholder: it returns
the first inference-only candidate (base-model patch if a model
callable is supplied, else the first deterministic one-line heuristic
patch), or `None` when there is no candidate. `inference_agent.py`
(`run_inference_only`) is the full loop: it tries every candidate
rung by rung through the regression gate, minimal-diff first.

Honest scope: the heuristic source only flips a single operator /
boolean in a `return` expression (`-` <-> `+`, `*` <-> `/`,
comparisons, `and`/`or`, `not`, `True`/`False`), anchored so the old
text occurs exactly once. It cannot fix multi-line, wrong-constant,
or new-file bugs. The gate — not the generator — decides acceptance.
No repair-quality rate is claimed; local tests cover only the toy
`demo_repo/`. See `submission/NOTES.md` for the full limits and the
submission packaging (`package.py --variant inference-only`).

```bash
python3 inference_agent.py <repo_root> \
    --failing-test-id tests.test_x.TestX.test_fails \
    --regression tests.test_x.TestX.test_ok
```

## Design notes / limits

- Token estimate is chars/4 everywhere, so caps are comparable across runs.
- The runner uses `python -m unittest` only — no pytest dependency.
  Test ids are unittest dotted ids.
- Regression suite: max 15 tests, 90 s total, abort on first failure.
  Per-test subprocess timeout defaults to 30 s.
- Rung time budgets (30/60/120 s) are enforced by the ladder: a patch
  that passes the gate but exceeds its rung budget is rolled back.
- The harness never touches the network, never calls a model, and
  never writes outside the repo it is given (except the demo's temp dir).
