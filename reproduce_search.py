"""Re-run the 116-task inverse-operator search baseline (CPU-only).

Runs ``search_baseline/search_baseline.py --task-set all116`` from
the published tree (stdlib only, no network, no GPU, ~12 min CPU)
and checks the headline figures against the recorded 8 Nov 2026
run in ``search_baseline/search_baseline_116_results.json``.

Expected headline (also printed on success):

    77/116 any-inverse, 71/116 matched-inverse, 77 gate-accepted

Run from the repo root:

    python3 reproduce_search.py

It expects the held-out tasks in a sibling ``dataset/`` directory
(the companion Kaggle dataset placed alongside the repo; see
README.md). The fresh run writes to a temp file — the recorded
results JSON is never overwritten.

Exit code 0 only if the fresh run reproduces the recorded headline
figures exactly (any-inverse, matched-inverse, gate-accepted,
task count, candidate count).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "search_baseline" / "search_baseline.py"
RECORDED = HERE / "search_baseline" / "search_baseline_116_results.json"


def main() -> int:
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    t_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="reproduce_search_") as td:
        fresh_path = Path(td) / "fresh_116_results.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--task-set", "all116",
             "--out", str(fresh_path)],
            cwd=HERE, capture_output=True, text=True,
            env={**__import__("os").environ,
                 "PYTHONDONTWRITEBYTECODE": "1"})
        print(proc.stdout, end="")
        if proc.stderr.strip():
            print(proc.stderr, end="")
        if proc.returncode != 0:
            print("REPRODUCE_SEARCH FAILED: search script exited "
                  f"{proc.returncode}")
            return 1
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    elapsed = time.monotonic() - t_start
    keys = ("n_tasks", "total_fixed_any_inverse",
            "total_fixed_matched_inverse", "total_gate_accepted",
            "total_candidates_tried")
    mismatches = [k for k in keys if fresh[k] != recorded[k]]
    n = fresh["n_tasks"]
    print(f"SEARCH BASELINE REPRODUCED: "
          f"{fresh['total_fixed_any_inverse']}/{n} any-inverse, "
          f"{fresh['total_fixed_matched_inverse']}/{n} matched-inverse, "
          f"{fresh['total_gate_accepted']} gate-accepted "
          f"({elapsed:.0f}s; {fresh['total_candidates_tried']} candidates; "
          f"Wilson any {fresh['wilson_95_pooled_any_inverse']}, "
          f"matched {fresh['wilson_95_pooled_matched_inverse']})")
    if mismatches:
        print("REPRODUCE_SEARCH FAILED: fresh run differs from recorded "
              "results on: " + ", ".join(
                  f"{k} (fresh {fresh[k]} vs recorded {recorded[k]})"
                  for k in mismatches))
        return 1
    print("REPRODUCE_SEARCH PASSED: fresh run matches the recorded "
          "8 Nov 2026 results exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
