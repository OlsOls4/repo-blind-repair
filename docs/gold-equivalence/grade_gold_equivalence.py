#!/usr/bin/env python3
"""Gold-equivalence re-grading for the attempt-7 kill-32 run.

Formalises the wave-11 attribution's headline (25/32 as-graded ->
17/32 gold-equivalent) as a runnable script, so the figure is a
measurement, not an assertion.

Method
------
* The 8 guard tasks (4 dropped_condition + 4 swapped_boolean) are
  re-graded by PROBE, not by the visible 3-test suite: the agent's
  patch is applied (git apply) to a fresh copy of the task files and
  the patched function is called on discriminating inputs whose
  gold behaviour is known. A patch is gold-equivalent only if it
  matches gold on EVERY probe.
* Gold is reconstructed by sibling consensus (wave-12 Diagnosis):
  for each guarded function the DEFAULT comes from the
  dropped_condition sibling (its guard line is the mutated one) and
  the GUARD CONDITION comes from the swapped_boolean sibling (its
  default is the mutated one). The reconstruction is CHECKED, not
  assumed: the gold file must pass the task's own failing +
  regression tests, else the script aborts.
* The 17 non-guard resolves are taken as gold on the attempt-7
  verify result (resolved=True) + the attribution report's
  probe-executed check of all 17 (including the algebraically
  regrouped ones). That basis is stated, not re-proven here.
* The 7 failures have no resolving patch and are not gold-equivalent
  by definition.

Borderline: dilution-swapped_boolean-0 differs from gold on exactly
one probe — final_volume(10,100,0,safe=False), where gold RAISES
ZeroDivisionError and the patch returns 0. A hidden test catches
that only if it asserts the exception, so the task is reported
probe-weak with a borderline note (its verdict stands: patch != gold).

Usage: python3 grade_gold_equivalence.py
Exit 0 iff the tally reproduces 17/32 (17 non-guard + 0/8 guard).
"""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "k2-kill32-summary.json"
TASKS_FILE = HERE / "dev_tasks_heldout.jsonl"

# Gold functions, reconstructed by sibling consensus (see header).
# Each is the guarded function only; the file's other function is
# unmutated in every variant and is taken from the buggy file.
GOLD_FN = {
    "circle_area": (
        "def circle_area(radius, checked=True):\n"
        "    if checked and radius < 0:\n"
        "        return 0\n"
        "    return 3.14159 * radius * radius\n"),
    "kinetic_energy": (
        "def kinetic_energy(mass, velocity, checked=True):\n"
        "    if checked and mass < 0:\n"
        "        return 0\n"
        "    return mass * velocity * velocity / 2\n"),
    "final_volume": (
        "def final_volume(c1, v1, c2, safe=True):\n"
        "    if safe and c2 <= 0:\n"
        "        return 0\n"
        "    return c1 * v1 / c2\n"),
    "pressure": (
        "def pressure(force, area, safe=True):\n"
        "    if safe and area <= 0:\n"
        "        return 0\n"
        "    return force / area\n"),
}

# (function, args, kwargs, expected) — expected is a number, or
# ("raises", ExceptionClass). Floats compared with rel_tol 1e-9.
PROBES = {
    "heldout-circle-dropped_condition-1": [
        ("circle_area", (-1,), {}, 0),
        ("circle_area", (-1,), {"checked": False}, 3.14159),
        ("circle_area", (2,), {}, 12.56636),
        ("circle_area", (2,), {"checked": False}, 12.56636),
    ],
    "heldout-circle-swapped_boolean-0": [
        ("circle_area", (-1,), {}, 0),
        ("circle_area", (-1,), {"checked": False}, 3.14159),
        ("circle_area", (2,), {}, 12.56636),
        ("circle_area", (2,), {"checked": False}, 12.56636),
    ],
    "heldout-kinetic-dropped_condition-1": [
        ("kinetic_energy", (-5, 4), {}, 0),
        ("kinetic_energy", (-5, 4), {"checked": False}, -40.0),
        ("kinetic_energy", (5, 4), {}, 40.0),
        ("kinetic_energy", (5, 4), {"checked": False}, 40.0),
    ],
    "heldout-kinetic-swapped_boolean-0": [
        ("kinetic_energy", (-5, 4), {}, 0),
        ("kinetic_energy", (-5, 4), {"checked": False}, -40.0),
        ("kinetic_energy", (5, 4), {}, 40.0),
        ("kinetic_energy", (5, 4), {"checked": False}, 40.0),
    ],
    "heldout-dilution-dropped_condition-1": [
        ("final_volume", (10, 100, 0), {}, 0),
        ("final_volume", (10, 100, 0), {"safe": False},
         ("raises", ZeroDivisionError)),
        ("final_volume", (10, 100, 2), {}, 500.0),
        ("final_volume", (10, 100, -1), {}, 0),
        ("final_volume", (10, 100, -1), {"safe": False}, -1000.0),
    ],
    "heldout-dilution-swapped_boolean-0": [
        ("final_volume", (10, 100, 0), {}, 0),
        ("final_volume", (10, 100, 0), {"safe": False},
         ("raises", ZeroDivisionError)),
        ("final_volume", (10, 100, 2), {}, 500.0),
        ("final_volume", (10, 100, -1), {}, 0),
        ("final_volume", (10, 100, -1), {"safe": False}, -1000.0),
    ],
    "heldout-pressure-dropped_condition-1": [
        ("pressure", (100, 0), {}, 0),
        ("pressure", (100, 0), {"safe": False},
         ("raises", ZeroDivisionError)),
        ("pressure", (100, 5), {}, 20.0),
        ("pressure", (100, -5), {}, 0),
        ("pressure", (100, -5), {"safe": False}, -20.0),
    ],
    "heldout-pressure-swapped_boolean-0": [
        ("pressure", (100, 0), {}, 0),
        ("pressure", (100, 0), {"safe": False},
         ("raises", ZeroDivisionError)),
        ("pressure", (100, 5), {}, 20.0),
        ("pressure", (100, -5), {}, 0),
        ("pressure", (100, -5), {"safe": False}, -20.0),
    ],
}

# The one borderline probe (see header): gold raises, patch returns 0.
BORDERLINE = {
    "heldout-dilution-swapped_boolean-0":
        "final_volume(10,100,0,safe=False): gold raises "
        "ZeroDivisionError, patch returns 0 — hidden test catches it "
        "only if it asserts the exception",
}


def load_tasks():
    tasks = {}
    for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        tasks[rec["task_id"]] = rec
    return tasks


def replace_function(source: str, fn_name: str, new_fn: str) -> str:
    """Replace the top-level `def fn_name(...):` block with new_fn."""
    lines = source.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines)
                 if l.startswith(f"def {fn_name}("))
    end = start + 1
    while end < len(lines) and not lines[end].startswith("def "):
        end += 1
    return "".join(lines[:start]) + new_fn + "".join(lines[end:])


def run_tests(workdir: Path, test_ids) -> bool:
    """Run the task's own tests in a subprocess; True iff exit 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "-q"] + list(test_ids),
        cwd=workdir, capture_output=True, text=True, timeout=120)
    return proc.returncode == 0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_call(fn, args, kwargs, expected):
    """Return (match, actual_repr)."""
    if isinstance(expected, tuple) and expected[0] == "raises":
        try:
            got = fn(*args, **kwargs)
        except expected[1]:
            return True, f"raised {expected[1].__name__}"
        except Exception as exc:  # noqa: BLE001 - any other error differs
            return False, f"raised {type(exc).__name__}: {exc}"
        return False, repr(got)
    try:
        got = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"
    if isinstance(expected, float) or isinstance(got, float):
        ok = math.isclose(float(got), float(expected), rel_tol=1e-9)
    else:
        ok = got == expected
    return ok, repr(got)


def grade_guard_task(task_id, rec, patch):
    """Returns (verdict, failing_probes). Verdict: gold | probe-weak |
    apply_failed. Raises SystemExit on a bad gold reconstruction."""
    files = rec["files"]
    py_file = next(f for f in files
                   if f.endswith(".py") and "test" not in f)
    fn_name = next(k for k in GOLD_FN if k in files[py_file])
    test_ids = [rec["failing_test_id"]] + list(
        rec["regression_test_ids"])
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        # 1. Validate the gold reconstruction against the task's tests.
        gold_source = replace_function(files[py_file], fn_name,
                                       GOLD_FN[fn_name])
        for rel, content in files.items():
            dest = workdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(gold_source if rel == py_file else content,
                            encoding="utf-8")
        if not run_tests(workdir, test_ids):
            print(f"GOLD RECONSTRUCTION FAILED for {task_id}: the "
                  f"reconstructed gold does not pass the task's own "
                  f"tests. Fix GOLD_FN, do not trust any verdict.",
                  file=sys.stderr)
            raise SystemExit(2)
        # 2. Apply the agent's patch to a fresh copy of the buggy files.
        for rel, content in files.items():
            dest = workdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        (workdir / "agent.patch").write_text(patch, encoding="utf-8")
        proc = subprocess.run(["git", "apply", "agent.patch"],
                              cwd=workdir, capture_output=True,
                              text=True, timeout=60)
        if proc.returncode != 0:
            return "apply_failed", [f"git apply: {proc.stderr.strip()}"]
        # 3. Probe the patched function.
        mod = load_module(workdir / py_file, f"patched_{task_id}")
        failing = []
        for fname, args, kwargs, expected in PROBES[task_id]:
            fn = getattr(mod, fname)
            ok, actual = probe_call(fn, args, kwargs, expected)
            if not ok:
                call = (f"{fname}{args}" +
                        (f", {kwargs}" if kwargs else ""))
                failing.append(f"{call}: gold {expected!r}, "
                               f"patch {actual}")
        return ("gold" if not failing else "probe-weak"), failing


def main():
    tasks = load_tasks()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    guard_ids = sorted(PROBES)
    gold_count = 0
    rows = []
    for entry in summary["tasks"]:
        task_id = entry["task_id"]
        resolved = (entry.get("verify") or {}).get("resolved", False)
        if task_id in PROBES:
            verdict, failing = grade_guard_task(
                task_id, tasks[task_id], entry["final_patch"])
            rows.append((task_id, verdict, failing))
            if verdict == "gold":
                gold_count += 1
        elif resolved:
            # Non-guard resolve: gold per attempt-7 verify + the
            # attribution report's probe-executed check of all 17.
            rows.append((task_id, "gold (attempt-7 verify + "
                         "attribution probe check)", []))
            gold_count += 1
        else:
            rows.append((task_id, "not resolved", []))
    print(f"{'task':42s}  verdict")
    for task_id, verdict, failing in sorted(rows):
        print(f"{task_id:42s}  {verdict}")
        for f in failing:
            print(f"{'':42s}    FAIL {f}")
        if task_id in BORDERLINE and verdict == "probe-weak":
            print(f"{'':42s}    BORDERLINE: {BORDERLINE[task_id]}")
    print()
    print(f"gold-equivalent: {gold_count}/32 "
          f"(17 non-guard resolved + "
          f"{sum(1 for _, v, _ in rows if v == 'gold')}/8 guard)")
    print(f"as-graded (visible tests): "
          f"{sum(1 for e in summary['tasks'] if (e.get('verify') or {}).get('resolved'))}/32")
    ok = gold_count == 17
    print("TALLY REPRODUCES 17/32: " + ("YES" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
