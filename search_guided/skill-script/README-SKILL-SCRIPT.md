# Skill-script arm — `repo-blind-repair-skill-script`

**Status: EARNS Day I (Sat 14 Nov).** Simulated 103/116 (kill 29/32,
singles 95/107) with verified minimal edits — beats v3 (95/116,
28/32, 87/107) on every comparable set, AND with strictly lower
cost (8 model round-trips/task vs ~27, latency-proof to 8 s/call).
Package: `submission-skill-script.zip`, md5
`691f9a94f564339c9b41ad81e805257b`, 7 files, 99,782 bytes unpacked,
12B, inference-only, budgets unchanged (120 s / 60 calls / 5 min /
60 turns). Validated with the extended `package.py` (exit 0).
Rebuilt 8 Nov (wave 8): D1–D4 fixes, re-measured — sim numbers
unchanged (103/116, kill 29/32, singles 95/107).

## Why this arm exists

The mock scorer (`kaggle/MOCK_SCORER.md`) found LATENCY is the
binding risk for the prompt arms: the v2 prompt spends ~27 model
round-trips per task, so at 5 s/call it drops to 68/107 and at
8 s/call to 57/107 (from 69/107 at 4 s). A skill script collapses
that: the agent orients (≤4 calls), makes ONE `run_skill_script`
call that runs the deterministic enumeration + test runs inside the
sandbox, then applies the returned edit, gates, and submits —
about 8 model round-trips per task total. The script's internal
runtime debits against the 5-minute per-task budget, not the call
budget.

## Procedure (what the agent does per task)

1. **Orient (≤4 calls).** `get_status`, read the failing test,
   find the single function the failing test calls (the implicated
   function), discover the repo's test command.
2. **`load_skill` `enumerate-inverses`**, then ONE
   `run_skill_script` call: skill `enumerate-inverses`, script
   `scripts/enumerate_inverses.py`, `--file <buggy file>
   --function <implicated function> --test-file <failing test
   file> --failing-test "<failing-test command>" --regressions
   "<related-tests command>" --time-budget 100`.
3. **Interpret the JSON** the script prints. `"fixed"` → apply
   `edit.old_text` → `edit.new_text` with `edit_file`, run the
   agent's OWN gate (failing test + ≤15 regressions / 90 s), then
   `submit_patch`. `"not_found"` / `"error"` → do NOT retry the
   same call; fall back to the manual v3 procedure (rungs 2–3) or
   stop.
4. **Fallback.** If the skill tools are unavailable at all, the
   system prompt contains the complete v3 manual procedure — the
   arm strictly dominates v3 in expectation: skill-first when the
   tools exist, v3-manual when they do not.

## What the script does

`skills/enumerate-inverses/scripts/enumerate_inverses.py`
(63,849 bytes, stdlib only, self-contained): given a file, a
function, and the test commands, it enumerates candidates in the
v3 combined order — P0 symptom-first guess, P1 enumeration in
prompt order (comparison flip, operand swap, boolean flip,
integer +1/−1, integer −2…−6, reconstruction guess) restricted to
the implicated function, then the reconstruction phase (R1
recomputed return, R2 re-inserted assignment, R3 re-inserted
guard) — running the failing test for each with the fixed-harness
semantics (all `__pycache__` deleted before every run AND
bytecode writing disabled). It is EXHAUSTIVE within that order:
no 27-try cap (the cap was a model-call budget, not a candidate
budget). It returns the first gate-accepted fix as a MINIMAL edit
spec `{old_text, new_text}`, verified end-to-end (the applied edit
itself passes the failing test and the regressions) before it is
reported. A candidate whose edit cannot be built or verified is
skipped, not reported. The script NEVER leaves a rewritten file
behind: candidates are written to the target file only for one
test run each, and the original bytes are restored after every
candidate and on exit (try/finally). Full order in
`skills/enumerate-inverses/resources/candidate-order.md`; the
invocation contract in `skills/enumerate-inverses/SKILL.md`.

## Measured numbers (simulator drives the REAL script)

`kaggle/simulate_skill_script.py` runs the actual script as a
subprocess on every held-out task (live `--time-budget 210`
default), then independently re-verifies each returned edit on a
FRESH copy of the buggy file. Results:
`kaggle/simulate_skill_script_results.json` (116 tasks, 266 s).

| set | skill-script arm | v3 prompt sim | v2 prompt (mock scorer) |
|---|---|---|---|
| kill (32) | **29/32** | 28/32 | — |
| all singles (107) | **95/107** | 87/107 | 69/107 @ 4 s/call; 68 @ 5 s; 57 @ 8 s |
| compounds (9) | 8/9 | 8/9 | — |
| all 116 | **103/116** | 95/116 | — |

- 0 errors, 0 timeouts, 0 unappliable (gate-accepted but no
  verifiable edit), 13 not_found — ALL 13 are destructive-class
  tasks (removed_return / removed_assignment) with no fixing
  candidate in the order. Every invertible class is 100% fixed:
  off_by_one 20/20, swapped_operands 20/20, wrong_constant 20/20,
  wrong_comparator 6/6, swapped_boolean 5/5, dropped_condition
  5/5. removed_return 15/20, removed_assignment 12/20.
- By phase (fixed): P1-enum 77, R2-assign 11, R1-return-expr 7,
  P0-guess 4, P0-guard-literal 4. R3 measured 0 wins (as in v3).
- Rank of fix: median 5, p90 25, max 101. **8 fixes sit at rank
  >27** — beyond the v3 prompt's 27-try cap; these are the 8
  extra fixes exhaustive enumeration buys (predicted analytically
  from the v3 recon results before the run: 29/32, 95/107,
  103/116 — the run reproduced the prediction exactly).
- Edit method: difflib 103/103 on this normalized set. The
  AST node-span splice fallback was verified separately on a
  non-normalized file (comments, blank lines): it fixed the task
  with a one-line edit and left the comments intact.
- Script wall time per task (the skill call's cost): min 0.4 s,
  median 0.9 s, p90 5.1 s, p95 8.5 s, max 18.7 s. Total
  candidates tried: 2,968 across 116 tasks (median 7/task,
  max 266). All well inside the 100 s `--time-budget` the
  system prompt instructs (and inside the 120 s per-tool-call
  timeout) — a 100 s budget loses no measured fix.

### Latency sensitivity (task completes within the 300 s budget)

Model calls per task: the simulator modeled 8 (orient 4 + skill
call 1 + gate 2 + submit 1). The true worst case is 10 — add
`load_skill` (1) and the `edit_file` application (1) — which at
8 s/call is 10×8 + 18.9 = 98.9 s, still inside the 300 s budget.
Elapsed = 8 × L + script time in the table below; the +2-call
correction changes no row.

| L (s/call) | skill arm fixed-and-in-budget (all 116) |
|---|---|
| 0 | 103/116 |
| 4 | 103/116 |
| 5 | 103/116 |
| 6 | 103/116 |
| 8 | 103/116 |

The prompt arm (v2 mock scorer, singles) falls 69 → 68 → 57
over the same range. At 8 s/call the skill arm keeps all 103
fixes (worst case 8×8 + 18.9 = 82.9 s < 300 s); the prompt arm
loses 12. This is the latency argument, measured, not asserted.

## Acceptance

The bar was: beat v3 where comparable (kill >28/32 or singles
>87/107 or 116 >95/116) OR match v3 with strictly lower
call/time cost. The arm clears BOTH prongs: it beats v3 on all
three sets (29 > 28, 95 > 87, 103 > 95) AND it costs 8 model
round-trips per task instead of ~27 and is flat across the
latency range. **It earns Day I (Sat 14 Nov).** The parent owns
Day I scheduling (cron + manifest/ladder entries) — not done
here.

## Design decisions (the ones that could have gone wrong)

- **Skill tools are NOT declared in `agent.yaml`'s `tools`
  list.** The local validator (and, by the schema ground truth,
  the real compiler) accepts only the nine built-in names (+
  `agent_tool` dicts); adding unlisted names risks a compile
  rejection of the whole submission. The SkillToolset tools are
  harness-attached when `skills/` is present — the only reading
  under which the Overview's `skills/` structure, the guide's
  "skills can run scripts through run_skill_script", and
  HARNESS_README's ToolErrorPlugin handling of SkillToolset tools
  are all coherent. The system prompt falls back to the full v3
  manual procedure if the skill tools are unavailable.
- **The script returns an edit spec, not a modified file.**
  Candidates are generated by AST transform + `ast.unparse`,
  which would reformat a real (non-normalized) file if left in
  place — hence the minimal edit (difflib primary, AST node-span
  splice fallback with a parse check) and the end-to-end
  verification of the applied edit before reporting.
- **`--time-budget 100` in the system prompt, not the script's
  210 s default.** A `run_skill_script` call is one tool call
  with a 120 s timeout in this submission's eval config; the
  script must return its JSON before that timeout can kill it.
  Measured max script time is 18.9 s, so 100 s is headroom.
- **Budgets inherited from v3 byte-identical** (120 s / 60 calls
  / 5 min / 60 turns). The task brief's "50 turns" is the stale
  v2 value; v3's eval budget is 60 turns.

## Legality evidence (skill scripts are legal — verified 8 Nov 2026)

From the official competition Overview's Submission Structure
(checked live 8 Nov 2026):

> `skills/<skill-name>/` with `SKILL.md` and `scripts/`
> ("Python or bash scripts executed in a sandbox")

> "Scripts executed via `run_skill_script` run securely inside
> the competition's persistent Docker container, sharing
> filesystem access with `run_command` and debiting execution
> time against your central budget. Agents can inspect domain
> knowledge files using `load_skill_resource`."

From HARNESS_README (allowed extensions include):

> ".py (ADK skill scripts)"

and its ToolErrorPlugin coerces malformed output on the
SkillToolset tools (`load_skill`, `load_skill_resource`,
`run_skill_script`), which are ADK SkillToolset tools — NOT one
of the nine predefined tools. Entry-point Python
(`agent.py`/`agent_fn`) is still banned; skill scripts are the
exception. The Google Developers Blog ADK skills guide (1 Apr
2026) confirms the shape: a file-based skill is a `SKILL.md`
with YAML frontmatter (name + description) plus optional
`references/` / `assets/` / `scripts/` subdirectories, and the
SkillToolset auto-generates `list_skills` / `load_skill` /
`load_skill_resource`.

## Package contents and validation

```
agent.yaml                                  name repo-blind-repair-skill-script, model gemma-4-12b-it-qat-w4a16-ct
configs/sampling.yaml                       byte-identical to v3
eval_config.yaml                            byte-identical to v3 (120/60/5/60)
prompts/system.md                           skill-first workflow + full v3 procedure as fallback (verbatim)
skills/enumerate-inverses/SKILL.md          frontmatter name + description; invocation contract
skills/enumerate-inverses/scripts/enumerate_inverses.py   the executable backend (63,849 bytes)
skills/enumerate-inverses/resources/candidate-order.md    the full candidate order
```

`package.py` (this directory's copy) was extended MINIMALLY for
skills and nothing else was weakened:

- `_is_zippable`: `skills/` joins the starter-shaped allowlist.
- `check_extensions`: `.py`/`.sh` ship ONLY under
  `skills/<name>/scripts/`; `.py` anywhere else still fails.
- New `check_skill_dirs`: every `skills/<name>/` holds a
  `SKILL.md` whose YAML frontmatter `name` matches the directory
  name and carries a non-empty `description`; skill names match
  `^[a-z][a-z0-9-]*$`; only `scripts/` and `resources/` subdirs.
- The final `.py` guard now exempts only
  `skills/<name>/scripts/*.py`. The nine-tools rule, the
  single-base rule, the include rules, and every other validation
  are unchanged.
- Negative-tested: `.py` under `prompts/` → FAIL; `.py` under
  `skills/<name>/resources/` → FAIL; `.py` directly under
  `skills/<name>/` → FAIL; SKILL.md frontmatter name mismatch →
  FAIL. The v3 package still validates with the UNMODIFIED v3
  `package.py` (exit 0).

## Limits (honest)

- The simulator measures the SCRIPT exhaustively with no model
  in the loop — the same idealization as the v3 simulator, so
  the comparison is apples-to-apples at simulator level. The
  real scorer adds model behaviour neither sim measures:
  orient errors (wrong implicated function), skill-call
  failures, edit-application errors, and the model ignoring the
  skill result. The Day I daily is the measurement.
- The script can only find fixes in its fixed candidate order.
  A bug needing a genuinely novel expression outside that order
  is not found (13/116 not_found, all destructive-class).
- The script does not judge whether a passing candidate is a
  true repair or a probe-weakness pass (e.g. a boolean flip that
  makes a guard never fire) — the agent's gate and judgement
  still apply. R3's 0 measured wins is the same caveat as v3.
- The SkillToolset attachment (tools NOT declared in
  `agent.yaml`) is the one unproven link: it is the only reading
  under which the official structure is coherent, but it has not
  been observed at the real scorer. If the tools are absent, the
  arm degrades to the v3 manual procedure — it cannot score
  WORSE than v3 in expectation, only equal. The Day I daily
  tests this.
- No GPU, no Kaggle, no submissions were involved in building or
  measuring this arm. Nothing here edits any existing submission
  package or the paper.
