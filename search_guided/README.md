# Search-guided arms — simulator results and the skill script

The paper's "What repairs instead" section is backed by the files in this
directory. All figures are **simulator upper bounds** (perfect enumeration,
no wasted model calls), not live scorer results.

| File | Backs |
|---|---|
| `search_guided_sim_results.json` | v1 prompt sim: 22/32 kill, 60/107 singles (efficient order 24/32, 69/107) |
| `search_guided_sim_v2_results.json` | v2 (revert-folding): 24/32 kill, 69/107 singles |
| `search_guided_sim_v3_ordered_results.json` | v3 prompt-order sim: 24/32 kill, 71/107 singles |
| `search_guided_sim_v3_recon_results.json` | v3 with reconstruction phase: 28/32 kill, 87/107 singles, 95/116 overall |
| `simulate_skill_script_results.json` | Skill-script arm (drives the real script): 29/32 kill, 95/107 singles, 103/116 overall, 0 errors, script median 0.8 s (0.9 s wall) |

`skill-script/` is the executable arm itself: `SKILL.md` (the skill
definition), `scripts/enumerate_inverses.py` (the enumeration script —
stdlib-only, 63,849 bytes; includes the D1 fix of 8 Nov 2026: the
comparator inverter now skips unswappable operators before counting),
and `README-SKILL-SCRIPT.md` (procedure, budgets, verification).
The live submission package is `submission-skill-script.zip` in the
competition workspace; the E4B variant is `submission-e4b-skill-script.zip`.

The simulator drivers (`kaggle/simulate_search_guided*.py`,
`kaggle/simulate_skill_script.py`) are local evaluation code and are not
in this repo; the JSONs above are their checked-in outputs. The search
baseline they build on is in `search_baseline/` (`python3
search_baseline/search_baseline.py --task-set all116` reruns it:
77/116 any-inverse, 71/116 matched).
