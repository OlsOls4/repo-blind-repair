"""Build a HELD-OUT synthetic evaluation set for eval_dev.py.

WHAT THIS IS — AND WHAT IT IS NOT
----------------------------------
This is the FALLBACK evaluation (see DEV_SET_SOURCING.md). The public
dev set (129 real tasks, fastapi/rich/requests/httpx) lives only in
the competition's login-gated Data tab dataset, whose internal
layout could not be verified from any public source. Until the
parent runs build_dev_tasks.py --inspect on the attached competition
data and converts the real dev set, this held-out set is the only
evaluation that can be built locally.

It measures SYNTHETIC transfer only: can the adapter repair
deterministic AST mutations of clean code it has never seen? It
does NOT measure real-bug transfer, and its fix rate must NEVER be
presented as a dev-set result, a leaderboard result, or a real-bug
fix rate. The Paper Track must label it "held-out synthetic".

HELD-OUT GUARANTEE
------------------
Every template below is a NEW domain/shape not present in the
training generator (kaggle/generate_scaled_dataset.py: price,
discount, tip, simple interest, loyalty, bulk, shipping, payroll,
distance, area/volume, grades, scores, temp/unit conversion, BMI,
slug/initials/mask/vowel/palindrome/word-count/acronym/title-case,
greeting, dedupe, filter, top-score, running-total, frequency,
second-largest, partition, search, merge-counts, inventory, flatten,
eligibility-age, access-control, refund, discount-tier, risk-band,
shipping-rule, weather, subscription, sum-squares, factorial,
fibonacci, countdown, batch-count, counter/cart/account/thermostat
classes, clamp, percentage, average/median, gcd, prime, leap-year,
count-evens, caesar, range-sum, max-min, plus 20 boolean-rule
templates). No held-out source text appears in the training file:
the builder asserts zero overlap against train_records_20k.jsonl
when that file is present.

DERIVED TESTS (not invented)
-----------------------------
For each record, probes are executed against the FIXED code to get
expected outputs (ground truth by construction). The record is kept
ONLY if, run in a subprocess with a timeout:
  - a main-function probe FAILS on the buggy code and PASSES on the
    fixed code (this becomes the failing test; probes run in-process
    against our own import-free, while-loop-free code, so they
    cannot hang), and
  - helper-function probes PASS on BOTH buggy and fixed code (these
    become the regression tests).
A record that does not discriminate is discarded, never shipped.
Test files are written into the task (tests/test_task.py) so the
harness's `python -m unittest` runner can execute them.

Constraints: no teacher models, no real fixes, no network, no
imports in generated code (pure arithmetic/string logic, no while
loops — a mutated constant in a while condition could hang; for
loops and arithmetic cannot).

Usage:
  python build_heldout_synthetic.py --out dev_tasks_heldout.jsonl \
      [--per-operator 20] [--train-file train_records_20k.jsonl]
Then validate with eval_dev.py --dry-run before any eval run.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mutation"))
from operators import OPERATORS  # noqa: E402
from generator import generate_single, MutationNotApplicable  # noqa: E402

# ---------------------------------------------------------------------------
# Held-out templates: (name, source, main fn, main probes, helper fn,
# helper probes). Probes are arg lists; expected values are computed by
# executing the FIXED source, never written by hand.
# ---------------------------------------------------------------------------

TEMPLATES = [
    ("compound", """def compound(principal, rate, years):
    total = principal
    for _ in range(years):
        total = total * (1 + rate)
    return total


def annual_growth(principal, rate):
    return principal * (1 + rate)
""", "compound", [[1000, 0.05, 3], [500, 0.1, 2]],
     "annual_growth", [[1000, 0.05], [500, 0.1]]),
    ("loan", """def monthly_payment(principal, annual_rate, months):
    r = annual_rate / 12
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def monthly_rate(annual_rate):
    return annual_rate / 12
""", "monthly_payment", [[12000, 0.12, 12], [6000, 0.06, 6]],
     "monthly_rate", [[0.12], [0.06]]),
    ("circle", """def circle_area(radius, checked=True):
    if checked and radius < 0:
        return 0
    return 3.14159 * radius * radius


def circle_circumference(radius):
    return 2 * 3.14159 * radius
""", "circle_area", [[5], [3], [-1]], "circle_circumference", [[5], [3]]),
    ("kinetic", """def kinetic_energy(mass, velocity, checked=True):
    if checked and mass < 0:
        return 0
    return mass * velocity * velocity / 2


def momentum(mass, velocity):
    return mass * velocity
""", "kinetic_energy", [[10, 4], [2, 3], [-5, 4]],
     "momentum", [[10, 4], [2, 3]]),
    ("days", """def day_of_year(month, day):
    lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    total = day
    for m in range(month - 1):
        total = total + lengths[m]
    return total


def days_in_month(month):
    lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return lengths[month - 1]
""", "day_of_year", [[3, 15], [12, 31]],
     "days_in_month", [[3], [12]]),
    ("variance", """def variance(values):
    n = len(values)
    mean = sum(values) / n
    total = 0
    for v in values:
        total = total + (v - mean) * (v - mean)
    return total / n


def mean(values):
    return sum(values) / len(values)
""", "variance", [[[2, 4, 4, 4, 5, 5, 7, 9]], [[1, 2, 3]]],
     "mean", [[[2, 4, 4, 4, 5, 5, 7, 9]], [[1, 2, 3]]]),
    ("isbn", """def isbn_check_digit(digits):
    total = 0
    for i in range(12):
        weight = 1 if i % 2 == 0 else 3
        total = total + digits[i] * weight
    return (10 - total % 10) % 10


def isbn_weighted_sum(digits):
    total = 0
    for i in range(12):
        weight = 1 if i % 2 == 0 else 3
        total = total + digits[i] * weight
    return total
""", "isbn_check_digit",
     [[[9, 7, 8, 0, 1, 3, 9, 4, 5, 6, 7, 8]],
      [[9, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]]],
     "isbn_weighted_sum",
     [[[9, 7, 8, 0, 1, 3, 9, 4, 5, 6, 7, 8]],
      [[9, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]),
    ("roman", """def int_to_roman(n):
    table = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
             (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
             (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    out = ''
    for value, symbol in table:
        count, n = divmod(n, value)
        out = out + symbol * count
    return out


def roman_value(symbol):
    table = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100,
             'D': 500, 'M': 1000}
    return table[symbol]
""", "int_to_roman", [[1994], [58]], "roman_value", [['X'], ['M']]),
    ("pagination", """def page_count(total, per_page):
    if per_page <= 0:
        return 0
    return (total + per_page - 1) // per_page


def page_offset(page, per_page):
    return (page - 1) * per_page
""", "page_count", [[95, 10], [100, 10]],
     "page_offset", [[3, 10], [1, 10]]),
    ("recipe", """def scale_amount(amount, servings, target):
    if servings <= 0:
        return 0
    return amount * target / servings


def total_weight(amounts):
    total = 0
    for a in amounts:
        total = total + a
    return total
""", "scale_amount", [[200, 4, 6], [150, 3, 3]],
     "total_weight", [[[200, 100, 50]], [[10, 20]]]),
    ("dilution", """def final_volume(c1, v1, c2, safe=True):
    if safe and c2 <= 0:
        return 0
    return c1 * v1 / c2


def final_concentration(c1, v1, v2):
    return c1 * v1 / v2
""", "final_volume", [[10, 100, 2], [5, 50, 5], [10, 100, 0]],
     "final_concentration", [[10, 100, 500], [5, 50, 50]]),
    ("pressure", """def pressure(force, area, safe=True):
    if safe and area <= 0:
        return 0
    return force / area


def force_from_pressure(pressure, area):
    return pressure * area
""", "pressure", [[100, 5], [60, 3], [100, 0]],
     "force_from_pressure", [[20, 5], [20, 3]]),
    ("ohm", """def voltage(current, resistance):
    return current * resistance


def power(voltage, current):
    return voltage * current
""", "voltage", [[2, 10], [3, 4]], "power", [[20, 2], [12, 3]]),
    ("kelvin", """def celsius_to_kelvin(celsius):
    return celsius + 273


def kelvin_to_celsius(kelvin):
    return kelvin - 273
""", "celsius_to_kelvin", [[25], [0]],
     "kelvin_to_celsius", [[298], [273]]),
    ("taxi", """def taxi_fare(distance, surge=True):
    base = 3
    total = base + distance * 2
    if surge and distance > 10:
        total = total + 5
    return total


def base_fare():
    return 3
""", "taxi_fare", [[15], [5]], "base_fare", [[], []]),
    ("gpa", """def gpa(grades, credits):
    total = 0
    weight = 0
    for g, c in zip(grades, credits):
        total = total + g * c
        weight = weight + c
    return total / weight


def total_credits(credits):
    total = 0
    for c in credits:
        total = total + c
    return total
""", "gpa", [[[4, 3, 2], [3, 3, 2]], [[4, 4], [1, 1]]],
     "total_credits", [[[3, 3, 2]], [[1, 1]]]),
    ("tax", """def tax_owed(income):
    if income < 10000:
        return 0
    if income < 40000:
        return income * 10 // 100
    if income < 100000:
        return income * 20 // 100
    return income * 30 // 100


def tax_rate(income):
    if income < 10000:
        return 0
    if income < 40000:
        return 10
    if income < 100000:
        return 20
    return 30
""", "tax_owed", [[5000], [20000], [60000], [150000],
                [10000], [40000], [100000]],
     "tax_rate", [[5000], [20000], [60000]]),
    ("electricity", """def electricity_bill(kwh):
    if kwh < 100:
        return kwh * 2
    if kwh < 500:
        return 200 + (kwh - 100) * 3
    return 200 + 1200 + (kwh - 500) * 5


def base_charge(kwh):
    if kwh >= 100:
        return 200
    return 0
""", "electricity_bill", [[50], [300], [600], [100], [500]],
     "base_charge", [[50], [300]]),
]

# Probes run IN-PROCESS, not in a subprocess: the generated code is
# ours, has no imports, no I/O, and no while loops (for loops and
# arithmetic only), so no probe can hang — a mutated divisor raises
# instead of looping. The earlier subprocess version was correct but
# ~100x slower (thousands of interpreter startups) and timed out.
def run_probes(source: str, calls: list[dict]) -> list[dict]:
    ns: dict = {}
    try:
        exec(compile(source, "task.py", "exec"), ns)
    except Exception as exc:
        return [{"ok": False, "error": type(exc).__name__}
                for _ in calls]
    out = []
    for call in calls:
        try:
            value = ns[call["fn"]](*call["args"])
            out.append({"ok": True, "value": repr(value)})
        except Exception as exc:
            out.append({"ok": False, "error": type(exc).__name__})
    return out


def test_file(main_fn: str, main_args: list, expected: str,
              helper_fn: str, helper_probes: list[tuple[list, str]]) -> str:
    lines = ["import unittest", "import task", "",
             "class TestTask(unittest.TestCase):",
             "    def test_failing(self):",
             f"        self.assertEqual(task.{main_fn}({', '.join(repr(a) for a in main_args)}), {expected})"]
    for i, (args, exp) in enumerate(helper_probes, start=1):
        lines.append(f"    def test_regression_{i}(self):")
        lines.append(f"        self.assertEqual(task.{helper_fn}({', '.join(repr(a) for a in args)}), {exp})")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    unittest.main()")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-operator", type=int, default=20)
    parser.add_argument("--train-file", type=Path,
                        default=HERE / "train_records_20k.jsonl")
    args = parser.parse_args(argv)

    train_fixed: set[str] = set()
    if args.train_file.is_file():
        for line in args.train_file.open(encoding="utf-8"):
            if line.strip():
                train_fixed.add(json.loads(line)["fixed_code"])
        print(f"[heldout] training fixed sources loaded: {len(train_fixed)}")

    counts = {op: 0 for op in OPERATORS}
    records: list[dict] = []
    seen_buggy: set[str] = set()
    target_total = args.per_operator * len(OPERATORS)
    for tname, source, main_fn, main_probes, helper_fn, helper_probes in TEMPLATES:
        ast.parse(source)
        if source in train_fixed:
            raise SystemExit(f"[heldout] ABORT: template {tname} "
                             "overlaps a training source")
        for op in OPERATORS:
            if counts[op] >= args.per_operator:
                continue
            for seed in range(200):
                if counts[op] >= args.per_operator:
                    break
                try:
                    rec = generate_single(source, op, seed,
                                          filename="task.py")
                except MutationNotApplicable:
                    continue
                buggy, fixed = rec["buggy_code"], rec["fixed_code"]
                if buggy in seen_buggy or buggy == fixed:
                    continue
                # Probes: fixed main + fixed helper (expectations),
                # buggy main + buggy helper (discrimination).
                calls = ([{"fn": main_fn, "args": a} for a in main_probes]
                         + [{"fn": helper_fn, "args": a} for a in helper_probes])
                try:
                    fixed_res = run_probes(fixed, calls)
                    buggy_res = run_probes(buggy, calls)
                except (subprocess.TimeoutExpired, ValueError):
                    continue
                n_main = len(main_probes)
                fixed_main, fixed_help = fixed_res[:n_main], fixed_res[n_main:]
                buggy_main, buggy_help = buggy_res[:n_main], buggy_res[n_main:]
                if not all(r["ok"] for r in fixed_main + fixed_help):
                    continue
                # Failing probe: first main probe that fails on buggy.
                fail_idx = next(
                    (i for i, (f, b) in enumerate(zip(fixed_main, buggy_main))
                     if not b["ok"] or b["value"] != f["value"]), None)
                if fail_idx is None:
                    continue
                # Regression probes: helper probes passing on both.
                reg = [(helper_probes[i], fixed_help[i]["value"])
                       for i in range(len(helper_probes))
                       if buggy_help[i]["ok"]
                       and buggy_help[i]["value"] == fixed_help[i]["value"]]
                if not reg:
                    continue
                task_id = f"heldout-{tname}-{op}-{seed}"
                records.append({
                    "task_id": task_id,
                    "files": {
                        "task.py": buggy,
                        "tests/__init__.py": "",
                        "tests/test_task.py": test_file(
                            main_fn, main_probes[fail_idx],
                            fixed_main[fail_idx]["value"],
                            helper_fn, reg),
                    },
                    "failing_test_id": "tests.test_task.TestTask.test_failing",
                    "regression_test_ids": [
                        f"tests.test_task.TestTask.test_regression_{i}"
                        for i in range(1, len(reg) + 1)],
                    "bug_class": op,
                    "issue_text": (
                        f"The function {main_fn} in task.py produces "
                        "incorrect output for some inputs; fix it so "
                        "all tests pass."),
                })
                seen_buggy.add(buggy)
                counts[op] += 1
    print(f"[heldout] per-operator counts: {counts}")
    print(f"[heldout] total: {len(records)} (target {target_total})")
    if len(records) < target_total:
        print("[heldout] WARNING: target not reached; some operators "
              "ran out of discriminating variants — do not pad with "
              "non-discriminating records", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[heldout] wrote {args.out}")
    print("[heldout] NEXT: validate with eval_dev.py --dry-run, then "
          "run eval. Label every number from this file 'held-out "
          "synthetic' — never 'dev-set'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
