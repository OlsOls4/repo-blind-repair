---
name: enumerate-inverses
description: Deterministically enumerate single-operator inverse edits (comparison flip, operand swap, boolean flip, integer change) plus reconstruction of destroyed returns/assignments/guards inside one implicated Python function, running the failing test for each candidate, and return the first gate-accepted fix as a minimal edit. Use when a failing test implicates a single function and the bug is a small local edit; it replaces ~27 model round-trips of manual enumeration with one script call.
---

# enumerate-inverses

Deterministic inverse-enumeration + reconstruction for single-function
Python bug repair. The agent orients (finds the implicated function,
reads the failing test), then makes ONE `run_skill_script` call; the
script does the enumeration and the test runs inside the sandbox and
returns the first gate-accepted fix as a minimal edit the agent
applies with `edit_file`. If the skill tools are unavailable, fall
back to the manual procedure in the system prompt — the skill
strictly dominates the manual procedure, it never replaces the
agent's own gate or submit decisions.

## When to use it

- A failing test implicates ONE function (the test calls it
  directly), and the bug is a small local edit inside that function.
- The failing test command and the regression command are known
  (discover the repo's runner first — pytest or unittest — do not
  assume one).
- Do NOT use it when the bug spans multiple functions, when no
  single function is implicated, or when the failing test cannot
  run at all (missing files, import errors) — fix the environment
  first, or stop.

## Invocation

1. `load_skill` with skill name `enumerate-inverses` (this file).
2. `run_skill_script` for skill `enumerate-inverses`, script
   `scripts/enumerate_inverses.py`, with these arguments (all
   paths repo-relative, commands as single strings):
   - `--file <buggy file>` — e.g. `task.py`
   - `--function <implicated function>` — e.g. `monthly_payment`.
     Omit only if the failing test unambiguously calls one function
     (the script derives it from `test_failing` then); pass it
     explicitly whenever the test calls more than one function.
   - `--test-file <failing test file>` — e.g. `tests/test_task.py`
   - `--failing-test "<command>"` — the command that runs ONLY the
     failing test, exit 0 = pass, e.g.
     `python3 -m unittest -v tests.test_task.TestTask.test_failing`
     or `python3 -m pytest -q tests/test_task.py::test_failing`
   - `--regressions "<command>"` — the command that runs the related
     existing tests (at most 15 tests, 90 seconds), exit 0 = pass.
     Omit only if the repo has no other tests for the module.
   - `--time-budget <seconds>` — optional; default 210. The script
     stops enumerating after this many seconds and reports what it
     tried. Keep it under the per-tool-call timeout (120 s in this
     submission's eval config) — see "Time budget" below.
3. Read the single JSON line the script prints on stdout.

## Interpreting the result

The script prints exactly one JSON object:

```json
{
  "status": "fixed" | "not_found" | "error",
  "candidate": {"phase": "P0-guess|P1-enum|R1-return-expr|R2-assign|R3-guard",
                "inverse": "...", "site": 0, "param": -1,
                "detail": "...", "rank": 9, "edit_method": "difflib|span"},
  "edit": {"old_text": "...", "new_text": "..."},
  "candidates_tried": 9, "candidates_total": 28,
  "elapsed_secs": 0.9, "timed_out": false, "error": null
}
```

- `"fixed"` — the script found a candidate that passes the failing
  test AND the regressions, built a minimal edit for it, and
  verified the APPLIED edit itself passes both. Apply
  `edit["old_text"]` -> `edit["new_text"]` with `edit_file`
  (old_text occurs exactly once in the original file), then run
  YOUR OWN gate (failing test + regressions on the applied edit)
  and `submit_patch`. Do not submit the script's word for it —
  the gate is yours.
- `"not_found"` — the script exhausted its candidate order (or its
  time budget) without a gate-accepted fix. Do NOT retry the same
  call. Fall back to the manual procedure (rungs 2–3, wider search)
  or stop.
- `"error"` — read `"error"`; it is an invocation problem (bad
  path, unparseable file, undeterminable function). Fix the
  invocation and retry once; if it errors again, fall back to the
  manual procedure.
- `candidates_tried` vs `candidates_total`: how far through the
  order the script got. `timed_out: true` means the time budget
  cut enumeration short — the fix may sit beyond the cutoff.

## What the script does (so you can trust it, and know its limits)

- Candidate order (fixed, deterministic — see
  `resources/candidate-order.md` for the full order):
  P0 symptom-first guess (failing test returns `None`), then
  enumeration in prompt order — comparison flip, operand swap,
  boolean flip, integer +1/−1, integer −2…−6, reconstruction guess
  — restricted to the implicated function, then the reconstruction
  phase: R1 recomputed return, R2 re-inserted assignment,
  R3 re-inserted guard. Exhaustive within that order: no 27-try
  cap (the cap was a model-call budget, not a candidate budget).
- Each candidate is tested with the fixed-harness semantics:
  all `__pycache__` directories under the repo root are deleted
  before every run AND the test runs with bytecode writing
  disabled, because CPython validates a `.pyc` by (mtime in whole
  seconds, size) and a same-size same-second rewrite would
  otherwise load stale bytecode.
- The script NEVER leaves a rewritten file behind: candidates are
  written to the target file only for the duration of one test
  run, and the original bytes are restored after every candidate
  and on exit (try/finally). It returns the fix as an edit spec,
  not as a modified file. Candidates are generated by AST
  transform + `ast.unparse`, which would reformat a real
  (non-normalized) file if left in place — hence the minimal edit
  spec (difflib, with an AST node-span splice fallback) and the
  end-to-end verification of the applied edit.
- A candidate whose edit cannot be built or verified is SKIPPED,
  not reported (counted as `unappliable`). A reported fix always
  carries a verified edit.
- Limits (honest): the script can only find fixes in its fixed
  candidate order — a bug needing a genuinely novel expression
  outside that order is not found. It tests one candidate per
  failing-test run, so its wall time grows with the candidate
  count (measured: median 0.5 s, max 11.3 s on the 32-task kill
  set; see the submission README for the full distribution). It
  does not read the repo beyond the target file and the test
  file, and it does not judge whether a passing candidate is a
  true repair or a probe-weakness pass — the agent's gate and
  judgement still apply.

## Time budget

A `run_skill_script` call is ONE tool call: it must complete
inside the per-tool-call timeout (120 s in this submission's
`eval_config.yaml`), and its runtime debits against the 5-minute
per-task budget. The script's default time budget is 210 s, which
is LONGER than the 120 s tool-call timeout — pass
`--time-budget 100` on every invocation so the script always
returns its JSON before the call timeout can kill it. (Measured
script wall time on the held-out set: median well under 1 s, max
under 60 s — a 100 s budget loses no measured fix; see the
submission README.) If a future eval config raises the tool-call
timeout, the budget may be raised to match it, minus headroom.

## Files

- `scripts/enumerate_inverses.py` — the executable backend
  (stdlib only, self-contained). `--help` documents every flag.
- `resources/candidate-order.md` — the full candidate order the
  script enumerates, with the rationale for each family.
