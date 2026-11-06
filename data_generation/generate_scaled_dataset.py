"""Generate the SCALED synthetic training dataset for Gemma Repo-Blind Repair.

Route: (a) ONLY — programmatically generated toy programs, written for
this task. No real human bug fixes, no teacher model output, no code
copied from any third-party source, so no licence verification is
needed and no licence claim is made about any external code. The
permissively-licensed route (b) was NOT used.

How it works
------------
- ~60 parameterised template functions below each produce a small,
  self-contained Python module (2-4 functions, arithmetic / string /
  list-dict / conditional / loop / small-class shapes). A deterministic
  RNG (seeded per template + index) varies identifiers, constants,
  thresholds and string literals, so the same shape yields many
  distinct sources.
- Each source is mutated through the UNCHANGED deterministic pipeline
  (mutation/generator.generate_record + mutation/operators.py, the
  same 8 operators as the pilot).
- Output: 20,000 records — 14,000 singles (exactly 1,750 per operator)
  + 6,000 composites (30%), deduplicated by buggy_code (also against
  the pilot file, so the scaled set does not repeat pilot records).
- A provenance sidecar (train_records_20k_provenance.jsonl) records
  (source_index, operators, base_seed) per record, so the validator
  can regenerate every record exactly and prove determinism.

Record schema (unchanged from the pilot):
  {buggy_code, fixed_code, mutations, failing_test, bug_class}
  failing_test is always "" — no test is ever invented.

Usage: python3 generate_scaled_dataset.py [--out DIR]
  Default output dir is this file's directory. Writes
  train_records_20k.jsonl, train_records_20k_provenance.jsonl and
  prints a summary. No network, no training, no paid anything.
  Runtime: a few minutes on CPU (stdlib only).
"""

from __future__ import annotations

import ast
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTATION_DIR = HERE.parent / "mutation"  # repo-relative (patched for publication; was a hardcoded local path)
sys.path.insert(0, str(MUTATION_DIR))

from generator import MutationNotApplicable, generate_record  # noqa: E402
from operators import OPERATORS  # noqa: E402

TARGET_TOTAL = 20_000
TARGET_SINGLE_PER_OP = 1_750  # 8 * 1750 = 14,000 singles
TARGET_COMPOSITE = 6_000
PER_TEMPLATE = 12  # distinct parameter draws per template shape
MASTER_SEED = 20261031

COMPOSITE_PAIRS = [
    ("off_by_one", "wrong_comparator"),
    ("swapped_operands", "wrong_constant"),
    ("dropped_condition", "off_by_one"),
    ("swapped_boolean", "wrong_comparator"),
    ("removed_return", "wrong_constant"),
    ("removed_assignment", "off_by_one"),
    ("wrong_comparator", "wrong_constant"),
    ("dropped_condition", "swapped_operands"),
    ("swapped_boolean", "off_by_one"),
    ("removed_return", "dropped_condition"),
    ("wrong_constant", "swapped_operands"),
    ("removed_assignment", "wrong_comparator"),
    ("off_by_one", "swapped_boolean"),
    ("dropped_condition", "wrong_constant"),
    ("swapped_operands", "removed_return"),
    ("wrong_comparator", "removed_assignment"),
]

# Identifier pools (generic words; no external code involved).
FN = ["calc", "compute", "total", "check", "find", "make", "get", "apply"]
A = ["amount", "total", "price", "cost", "value", "count", "score", "num"]
B = ["limit", "rate", "pct", "fee", "tax", "step", "base", "cap"]
C = ["items", "values", "names", "scores", "rates", "marks", "nums", "data"]


def _pick(rng, pool):
    return rng.choice(pool)


# ---------------------------------------------------------------- templates
# Each template takes an rng and returns one source module string.
# Templates deliberately mix int constants, BinOps, comparisons,
# BoolOp chains, boolean literals, returns and assignments so that
# all 8 operators can apply somewhere across the set. Generated code
# avoids dict/set literals and f-strings (brace collisions in the
# outer f-strings); dicts use dict(), strings use concatenation.

def t_price_total(rng):
    f, a, t, s = _pick(rng, ["order_total", "checkout_total", "final_price", "bill_total"]), _pick(rng, A), _pick(rng, B), rng.randint(3, 12)
    return f"""def {f}({a}, qty):
    subtotal = {a} * qty
    tax = subtotal * {s} // 100
    shipping = {rng.randint(2, 9)}
    if subtotal > {rng.randint(40, 120)}:
        shipping = 0
    return subtotal + tax + shipping


def {f}_per_item({a}, qty):
    if qty <= 0:
        return 0
    return {f}({a}, qty) // qty
"""


def t_discount_price(rng):
    f = _pick(rng, ["discounted", "sale_price", "after_discount", "reduced_price"])
    p, q = rng.randint(5, 40), rng.randint(50, 95)
    return f"""def {f}(price, pct):
    if pct > 0 and pct < 100:
        return price - price * pct // 100
    return price


def bulk_{f}(price, pct, n):
    one = {f}(price, pct)
    total = one * n
    if n >= {rng.randint(3, 10)}:
        total = total - {p}
    return total
"""


def t_tip_calc(rng):
    f = _pick(rng, ["tip_amount", "gratuity", "tip_total", "service_tip"])
    return f"""def {f}(bill, pct):
    if bill < 0 or pct < 0:
        return 0
    return bill * pct // 100


def split_{f}(bill, pct, people):
    tip = {f}(bill, pct)
    total = bill + tip
    if people <= 0:
        return total
    return total // people
"""


def t_interest_simple(rng):
    f = _pick(rng, ["simple_interest", "interest_earned", "loan_interest", "growth"])
    return f"""def {f}(principal, rate, years):
    if principal < 0 or years < 0:
        return 0
    return principal * rate * years // 100


def {f}_total(principal, rate, years):
    return principal + {f}(principal, rate, years)
"""


def t_loyalty_points(rng):
    f = _pick(rng, ["loyalty_points", "reward_points", "bonus_points", "member_points"])
    return f"""def {f}(spent):
    points = spent // {rng.randint(2, 10)}
    if spent >= {rng.randint(100, 500)}:
        points = points + {rng.randint(10, 50)}
    return points


def {f}_tier(points):
    if points >= {rng.randint(200, 1000)}:
        return "gold"
    if points >= {rng.randint(50, 200)}:
        return "silver"
    return "bronze"
"""


def t_bulk_price(rng):
    f = _pick(rng, ["bulk_price", "pack_price", "bundle_total", "case_price"])
    return f"""def {f}(unit, n):
    total = unit * n
    if n >= {rng.randint(5, 20)}:
        total = total * {rng.randint(80, 95)} // 100
    return total


def {f}_saved(unit, n):
    return unit * n - {f}(unit, n)
"""


def t_shipping_cost(rng):
    f = _pick(rng, ["shipping_cost", "delivery_fee", "postage", "freight_cost"])
    return f"""def {f}(weight, distance):
    cost = {rng.randint(2, 8)} + weight * {rng.randint(1, 4)}
    if distance > {rng.randint(50, 200)}:
        cost = cost + {rng.randint(5, 20)}
    if weight > {rng.randint(10, 30)}:
        cost = cost + {rng.randint(3, 12)}
    return cost
"""


def t_payroll_hours(rng):
    f = _pick(rng, ["weekly_pay", "wages", "pay_total", "overtime_pay"])
    return f"""def {f}(hours, rate):
    if hours <= {rng.randint(35, 45)}:
        return hours * rate
    extra = hours - 40
    return 40 * rate + extra * rate * 3 // 2


def {f}_net(hours, rate, tax_pct):
    gross = {f}(hours, rate)
    return gross - gross * tax_pct // 100
"""


def t_distance_time(rng):
    f = _pick(rng, ["travel_time", "trip_hours", "journey_time", "eta_hours"])
    return f"""def {f}(distance, speed):
    if speed <= 0:
        return -1
    return distance // speed


def {f}_with_breaks(distance, speed, stops):
    base = {f}(distance, speed)
    if base < 0:
        return -1
    return base + stops * {rng.randint(1, 3)} // 2
"""


def t_area_volume(rng):
    f = _pick(rng, ["rect_area", "box_volume", "floor_area", "room_area"])
    return f"""def {f}(width, height, depth):
    area = width * height
    volume = area * depth
    if area > {rng.randint(50, 200)}:
        return volume
    return area


def {f}_perimeter(width, height):
    return 2 * (width + height)
"""


def t_grade_letter(rng):
    f = _pick(rng, ["grade_letter", "letter_grade", "mark_grade", "score_grade"])
    a, b, c = rng.randint(85, 95), rng.randint(70, 84), rng.randint(50, 69)
    return f"""def {f}(mark):
    if mark >= {a}:
        return "A"
    if mark >= {b}:
        return "B"
    if mark >= {c}:
        return "C"
    return "F"


def {f}_passed(mark):
    return {f}(mark) != "F"
"""


def t_score_curve(rng):
    f = _pick(rng, ["curved_score", "adjusted_score", "boosted_mark", "scaled_score"])
    return f"""def {f}(raw, bonus):
    total = raw + bonus
    if total > 100:
        total = 100
    if total < 0:
        total = 0
    return total


def {f}_average(raws, bonus):
    total = 0
    for r in raws:
        total = total + {f}(r, bonus)
    if len(raws) == 0:
        return 0
    return total // len(raws)
"""


def t_temp_convert(rng):
    f = _pick(rng, ["c_to_f", "celsius_to_fahrenheit", "f_to_c", "fahrenheit_to_celsius"])
    if "c_to_f" in f or "celsius" in f:
        body = f"return value * 9 // 5 + 32"
    else:
        body = f"return (value - 32) * 5 // 9"
    return f"""def {f}(value):
    {body}


def {f}_rounded(value):
    result = {f}(value)
    if result >= 0:
        return result
    return result
"""


def t_unit_convert(rng):
    f = _pick(rng, ["kg_to_lb", "miles_to_km", "cm_to_inch", "litres_to_gallons"])
    num, den = rng.randint(2, 9), rng.randint(1, 5)
    return f"""def {f}(value):
    if value < 0:
        return 0
    return value * {num} // {den}


def {f}_back(value):
    return {f}(value) * {den} // {num}
"""


def t_bmi_band(rng):
    f = _pick(rng, ["weight_band", "size_band", "health_band", "bmi_class"])
    return f"""def {f}(weight, height_cm):
    score = weight * 10000 // (height_cm * height_cm)
    if score < 18:
        return "low"
    if score < 25:
        return "ok"
    if score < 30:
        return "high"
    return "very high"
"""


def t_slugify(rng):
    f = _pick(rng, ["slugify", "make_slug", "url_slug", "dash_text"])
    return f"""def {f}(text):
    out = ""
    for ch in text.lower():
        if ch == " ":
            out = out + "-"
        elif ch.isalnum():
            out = out + ch
    return out


def {f}_words(text):
    slug = {f}(text)
    count = 0
    for ch in slug:
        if ch == "-":
            count = count + 1
    return count + 1
"""


def t_initials(rng):
    f = _pick(rng, ["initials", "name_initials", "short_name", "monogram"])
    return f"""def {f}(first, last):
    if len(first) == 0 or len(last) == 0:
        return ""
    return first[0].upper() + last[0].upper()


def {f}_greeting(first, last):
    return "Hi " + {f}(first, last)
"""


def t_mask_string(rng):
    f = _pick(rng, ["mask_text", "hide_middle", "mask_value", "redact_text"])
    n = rng.randint(1, 4)
    return f"""def {f}(text):
    if len(text) <= {n * 2}:
        return "*" * len(text)
    return text[:{n}] + "*" * (len(text) - {n * 2}) + text[-{n}:]


def {f}_short(text):
    masked = {f}(text)
    return len(masked)
"""


def t_vowel_count(rng):
    f = _pick(rng, ["count_vowels", "vowel_total", "vowel_count", "num_vowels"])
    return f"""def {f}(text):
    count = 0
    for ch in text:
        if ch in "aeiouAEIOU":
            count = count + 1
    return count


def {f}_ratio(text):
    if len(text) == 0:
        return 0
    return {f}(text) * 100 // len(text)
"""


def t_palindrome(rng):
    f = _pick(rng, ["is_palindrome", "reads_same", "is_mirror", "pal_check"])
    return f"""def {f}(text):
    clean = ""
    for ch in text.lower():
        if ch.isalnum():
            clean = clean + ch
    left = 0
    right = len(clean) - 1
    while left < right:
        if clean[left] != clean[right]:
            return False
        left = left + 1
        right = right - 1
    return True
"""


def t_word_count(rng):
    f = _pick(rng, ["word_count", "count_words", "num_words", "words_total"])
    return f"""def {f}(text):
    words = text.split()
    return len(words)


def {f}_long_words(text, min_len):
    total = 0
    for w in text.split():
        if len(w) >= min_len:
            total = total + 1
    return total
"""


def t_acronym(rng):
    f = _pick(rng, ["acronym", "make_acronym", "short_form", "initial_letters"])
    return f"""def {f}(phrase):
    out = ""
    for word in phrase.split():
        if len(word) > 0:
            out = out + word[0].upper()
    return out
"""


def t_title_case(rng):
    f = _pick(rng, ["title_case", "capitalise_words", "proper_case", "title_text"])
    return f"""def {f}(text):
    out = ""
    for word in text.split():
        if len(word) > 0:
            out = out + word[0].upper() + word[1:] + " "
    return out.strip()


def {f}_count(text):
    return len({f}(text).split())
"""


def t_greeting(rng):
    f = _pick(rng, ["greet", "welcome_text", "hello_message", "greeting"])
    return f"""def {f}(name, hour):
    if hour < 12:
        part = "morning"
    elif hour < 18:
        part = "afternoon"
    else:
        part = "evening"
    return "Good " + part + ", " + name


def {f}_short(name, hour):
    return len({f}(name, hour))
"""


def t_dedupe(rng):
    f = _pick(rng, ["dedupe", "unique_items", "remove_dupes", "distinct"])
    return f"""def {f}(items):
    seen = []
    out = []
    for item in items:
        if item not in seen:
            seen.append(item)
            out.append(item)
    return out


def {f}_count(items):
    return len({f}(items))
"""


def t_filter_positive(rng):
    f = _pick(rng, ["positives", "filter_positive", "above_zero", "keep_positive"])
    return f"""def {f}(values):
    out = []
    for v in values:
        if v > 0:
            out.append(v)
    return out


def {f}_sum(values):
    total = 0
    for v in {f}(values):
        total = total + v
    return total
"""


def t_top_score(rng):
    f = _pick(rng, ["top_score", "best_score", "max_value", "highest"])
    return f"""def {f}(values):
    if len(values) == 0:
        return -1
    best = values[0]
    for v in values[1:]:
        if v > best:
            best = v
    return best


def {f}_is_top(values, x):
    return x == {f}(values)
"""


def t_running_total(rng):
    f = _pick(rng, ["running_total", "cumulative", "prefix_sums", "running_sum"])
    return f"""def {f}(values):
    out = []
    total = 0
    for v in values:
        total = total + v
        out.append(total)
    return out


def {f}_last(values):
    sums = {f}(values)
    if len(sums) == 0:
        return 0
    return sums[-1]
"""


def t_frequency(rng):
    f = _pick(rng, ["frequency", "count_each", "tally", "occurrences"])
    return f"""def {f}(items):
    counts = dict()
    for item in items:
        if item in counts:
            counts[item] = counts[item] + 1
        else:
            counts[item] = 1
    return counts


def {f}_most(items):
    counts = {f}(items)
    best = None
    best_n = -1
    for key in counts:
        if counts[key] > best_n:
            best_n = counts[key]
            best = key
    return best
"""


def t_second_largest(rng):
    f = _pick(rng, ["second_largest", "runner_up", "second_best", "second_max"])
    return f"""def {f}(values):
    first = -1
    second = -1
    for v in values:
        if v > first:
            second = first
            first = v
        elif v > second:
            second = v
    return second
"""


def t_partition(rng):
    f = _pick(rng, ["split_even_odd", "partition", "evens_odds", "split_parity"])
    return f"""def {f}(values):
    evens = []
    odds = []
    for v in values:
        if v % 2 == 0:
            evens.append(v)
        else:
            odds.append(v)
    return evens, odds


def {f}_counts(values):
    evens, odds = {f}(values)
    return len(evens), len(odds)
"""


def t_search_index(rng):
    f = _pick(rng, ["find_index", "search", "first_index", "locate"])
    return f"""def {f}(items, target):
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1


def {f}_exists(items, target):
    return {f}(items, target) != -1
"""


def t_merge_counts(rng):
    f = _pick(rng, ["merge_counts", "add_counts", "combine_tally", "sum_counts"])
    return f"""def {f}(a, b):
    out = dict()
    for key in a:
        out[key] = a[key]
    for key in b:
        if key in out:
            out[key] = out[key] + b[key]
        else:
            out[key] = b[key]
    return out
"""


def t_inventory_update(rng):
    f = _pick(rng, ["update_stock", "adjust_stock", "stock_after", "inventory_move"])
    return f"""def {f}(stock, sold, restocked):
    left = stock - sold + restocked
    if left < 0:
        left = 0
    return left


def {f}_low(stock, sold, restocked, threshold):
    return {f}(stock, sold, restocked) < threshold
"""


def t_flatten_pairs(rng):
    f = _pick(rng, ["flatten_pairs", "unpack_pairs", "flat_list", "join_pairs"])
    return f"""def {f}(pairs):
    out = []
    for a, b in pairs:
        out.append(a)
        out.append(b)
    return out


def {f}_length(pairs):
    return len({f}(pairs))
"""


def t_eligibility_age(rng):
    f = _pick(rng, ["is_eligible", "can_apply", "meets_age", "qualifies"])
    lo, hi = rng.randint(16, 21), rng.randint(60, 70)
    return f"""def {f}(age, resident, banned):
    return age >= {lo} and age <= {hi} and resident and not banned


def {f}_reason(age, resident, banned):
    if not {f}(age, resident, banned):
        return "no"
    return "yes"
"""


def t_access_control(rng):
    f = _pick(rng, ["can_access", "has_access", "allowed_in", "may_enter"])
    return f"""def {f}(role, active, level, needed):
    return active and (role == "admin" or level >= needed)


def {f}_denied(role, active, level, needed):
    return not {f}(role, active, level, needed)
"""


def t_refund_rule(rng):
    f = _pick(rng, ["refundable", "can_refund", "refund_ok", "return_allowed"])
    d = rng.randint(14, 60)
    return f"""def {f}(days, used, damaged, receipt):
    return days <= {d} and not used and not damaged and receipt


def {f}_window(days):
    return days <= {d}
"""


def t_discount_tier(rng):
    f = _pick(rng, ["discount_tier", "tier_pct", "member_discount", "level_discount"])
    return f"""def {f}(spent, member, years):
    if spent >= {rng.randint(500, 1000)} and member and years >= 2:
        return {rng.randint(15, 25)}
    if spent >= {rng.randint(100, 400)} and member:
        return {rng.randint(5, 14)}
    return 0
"""


def t_risk_band(rng):
    f = _pick(rng, ["risk_band", "risk_level", "score_band", "risk_class"])
    return f"""def {f}(score, late, defaults):
    if score < {rng.randint(300, 500)} or defaults > 0:
        return "high"
    if score < {rng.randint(600, 750)} or late > {rng.randint(1, 3)}:
        return "medium"
    return "low"
"""


def t_shipping_rule(rng):
    f = _pick(rng, ["free_shipping", "ships_free", "no_delivery_fee", "free_delivery"])
    return f"""def {f}(total, member, region_ok, weight):
    return total >= {rng.randint(25, 75)} and member and region_ok and weight <= {rng.randint(10, 30)}
"""


def t_weather_advice(rng):
    f = _pick(rng, ["weather_advice", "what_to_wear", "day_advice", "outdoor_ok"])
    return f"""def {f}(temp, rain, wind):
    if temp < 0 or wind > {rng.randint(40, 80)}:
        return "stay in"
    if rain and temp < 10:
        return "coat"
    if temp >= {rng.randint(20, 30)} and not rain:
        return "sun"
    return "mild"
"""


def t_subscription_tier(rng):
    f = _pick(rng, ["plan_name", "tier_name", "sub_tier", "package_name"])
    return f"""def {f}(users, storage, support):
    if users >= {rng.randint(50, 200)} and storage >= {rng.randint(500, 1000)}:
        return "enterprise"
    if users >= {rng.randint(5, 20)} or support:
        return "pro"
    return "basic"
"""


def t_sum_squares(rng):
    f = _pick(rng, ["sum_squares", "squares_total", "sum_of_squares", "sq_sum"])
    return f"""def {f}(n):
    total = 0
    k = 1
    while k <= n:
        total = total + k * k
        k = k + 1
    return total
"""


def t_factorial_loop(rng):
    f = _pick(rng, ["factorial", "fact", "product_upto", "factorial_loop"])
    return f"""def {f}(n):
    result = 1
    k = 2
    while k <= n:
        result = result * k
        k = k + 1
    return result
"""


def t_fibonacci_nth(rng):
    f = _pick(rng, ["fib_nth", "fibonacci", "fib_number", "fib_at"])
    return f"""def {f}(n):
    a = 0
    b = 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""


def t_countdown_sum(rng):
    f = _pick(rng, ["countdown_sum", "sum_downto", "descending_sum", "down_sum"])
    return f"""def {f}(n):
    total = 0
    while n > 0:
        total = total + n
        n = n - 1
    return total
"""


def t_batch_count(rng):
    f = _pick(rng, ["batch_count", "num_batches", "pages_needed", "chunk_count"])
    return f"""def {f}(total, size):
    if size <= 0:
        return 0
    full = total // size
    if total % size == 0:
        return full
    return full + 1
"""


def t_class_counter(rng):
    c = _pick(rng, ["Counter", "Tally", "Clicker", "Scoreboard"])
    return f"""class {c}:
    def __init__(self):
        self.count = 0
        self.active = True

    def add(self, n):
        if self.active:
            self.count = self.count + n
        return self.count

    def reset(self):
        self.count = 0
        return self.count

    def is_big(self):
        return self.count >= {rng.randint(10, 100)} and self.active
"""


def t_class_cart(rng):
    c = _pick(rng, ["Cart", "Basket", "Order", "Trolley"])
    return f"""class {c}:
    def __init__(self):
        self.items = []
        self.total = 0

    def add(self, price, qty):
        self.items.append(price)
        self.total = self.total + price * qty
        return self.total

    def clear(self):
        self.items = []
        self.total = 0
        return self.total

    def over(self, limit):
        return self.total > limit and len(self.items) > 0
"""


def t_class_account(rng):
    c = _pick(rng, ["Account", "Wallet", "Purse", "Balance"])
    lim = rng.randint(100, 1000)
    return f"""class {c}:
    def __init__(self, balance):
        self.balance = balance
        self.open = True

    def deposit(self, amount):
        if amount > 0 and self.open:
            self.balance = self.balance + amount
        return self.balance

    def withdraw(self, amount):
        if amount > 0 and amount <= self.balance and self.open:
            self.balance = self.balance - amount
        return self.balance

    def can_spend(self, amount):
        return self.open and amount <= self.balance and amount <= {lim}
"""


def t_class_thermostat(rng):
    c = _pick(rng, ["Thermostat", "Heater", "Climate", "TempControl"])
    return f"""class {c}:
    def __init__(self, target):
        self.target = target
        self.current = target
        self.on = False

    def tick(self, reading):
        self.current = reading
        if reading < self.target - 1:
            self.on = True
        elif reading > self.target + 1:
            self.on = False
        return self.on

    def needs_heat(self):
        return self.on and self.current < self.target
"""


def t_clamp_range(rng):
    f = _pick(rng, ["clamp", "clamp_value", "bound", "limit_to"])
    return f"""def {f}(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def {f}_pct(value, low, high):
    clamped = {f}(value, low, high)
    if high == low:
        return 0
    return (clamped - low) * 100 // (high - low)
"""


def t_percentage(rng):
    f = _pick(rng, ["percentage", "pct_of", "share_pct", "percent"])
    return f"""def {f}(part, whole):
    if whole == 0:
        return 0
    return part * 100 // whole


def {f}_left(part, whole):
    return 100 - {f}(part, whole)
"""


def t_average_median(rng):
    f = _pick(rng, ["average", "mean_value", "avg", "mean_of"])
    return f"""def {f}(values):
    if len(values) == 0:
        return 0
    return sum(values) // len(values)


def {f}_above(values, x):
    total = 0
    for v in values:
        if v > x:
            total = total + 1
    return total
"""


def t_gcd_loop(rng):
    f = _pick(rng, ["gcd", "greatest_common", "common_divisor", "gcd_of"])
    return f"""def {f}(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def {f}_coprime(a, b):
    return {f}(a, b) == 1
"""


def t_prime_check(rng):
    f = _pick(rng, ["is_prime", "prime_check", "is_p", "check_prime"])
    return f"""def {f}(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d = d + 1
    return True
"""


def t_leap_year(rng):
    f = _pick(rng, ["is_leap", "leap_year", "is_leap_year", "leap_check"])
    return f"""def {f}(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def {f}_days(year):
    if {f}(year):
        return 366
    return 365
"""


def t_count_evens(rng):
    f = _pick(rng, ["count_evens", "even_count", "num_evens", "evens_in"])
    return f"""def {f}(values):
    total = 0
    for v in values:
        if v % 2 == 0:
            total = total + 1
    return total


def {f}_ratio(values):
    if len(values) == 0:
        return 0
    return {f}(values) * 100 // len(values)
"""


def t_caesar_shift(rng):
    f = _pick(rng, ["shift_text", "caesar", "rotate_text", "shift_letters"])
    k = rng.randint(1, 25)
    return f"""def {f}(text):
    out = ""
    for ch in text:
        if "a" <= ch <= "z":
            out = out + chr((ord(ch) - 97 + {k}) % 26 + 97)
        else:
            out = out + ch
    return out
"""


def t_range_sum(rng):
    f = _pick(rng, ["range_sum", "sum_between", "sum_range", "total_range"])
    return f"""def {f}(lo, hi):
    total = 0
    for n in range(lo, hi + 1):
        total = total + n
    return total
"""


def t_max_min(rng):
    f = _pick(rng, ["min_max", "range_of", "spread", "low_high"])
    return f"""def {f}(values):
    if len(values) == 0:
        return 0, 0
    low = values[0]
    high = values[0]
    for v in values[1:]:
        if v < low:
            low = v
        if v > high:
            high = v
    return low, high
"""


TEMPLATES = [
    t_price_total, t_discount_price, t_tip_calc, t_interest_simple,
    t_loyalty_points, t_bulk_price, t_shipping_cost, t_payroll_hours,
    t_distance_time, t_area_volume, t_grade_letter, t_score_curve,
    t_temp_convert, t_unit_convert, t_bmi_band, t_slugify,
    t_initials, t_mask_string, t_vowel_count, t_palindrome,
    t_word_count, t_acronym, t_title_case, t_greeting,
    t_dedupe, t_filter_positive, t_top_score, t_running_total,
    t_frequency, t_second_largest, t_partition, t_search_index,
    t_merge_counts, t_inventory_update, t_flatten_pairs,
    t_eligibility_age, t_access_control, t_refund_rule,
    t_discount_tier, t_risk_band, t_shipping_rule, t_weather_advice,
    t_subscription_tier, t_sum_squares, t_factorial_loop,
    t_fibonacci_nth, t_countdown_sum, t_batch_count,
    t_class_counter, t_class_cart, t_class_account, t_class_thermostat,
    t_clamp_range, t_percentage, t_average_median, t_gcd_loop,
    t_prime_check, t_leap_year, t_count_evens, t_caesar_shift,
    t_range_sum, t_max_min,
]


# --- Boolean-rich templates (Gate 0 lesson, scaled) ---
# Single general templates hold very few BoolOp/boolean sites
# (swapped_boolean: ~56 variants across all 62 of them). These 20
# templates are deliberately dense in BoolOp chains (3-4 operands)
# and boolean literals, so dropped_condition and swapped_boolean
# reach the 1,750-per-operator target without padding repeats.

def t_bool_library(rng):
    f = _pick(rng, ["can_borrow", "may_borrow", "loan_ok", "borrow_allowed"])
    return f"""def {f}(member, overdue, fines, limit_ok):
    return member and not overdue and fines == 0 and limit_ok


def {f}_renew(member, overdue, reserved, days):
    return member and not overdue and not reserved and days <= {rng.randint(7, 30)}


def {f}_defaults():
    auto_renew = True
    notify = False
    return auto_renew or notify


def {f}_blocked(overdue, fines, banned):
    return overdue or fines > 0 or banned is True
"""


def t_bool_gym(rng):
    f = _pick(rng, ["gym_entry", "can_enter_gym", "gym_ok", "member_entry"])
    return f"""def {f}(member, paid, banned, hours_ok):
    return member and paid and not banned and hours_ok


def {f}_class_booked(member, paid, full, waitlist):
    return member and paid and (not full or waitlist)


def {f}_defaults():
    towel = True
    locker = False
    return towel and not locker


def {f}_peak(hour, member, guest):
    return hour >= 17 and hour <= 20 and (member or guest)
"""


def t_bool_event(rng):
    f = _pick(rng, ["event_entry", "can_attend", "ticket_ok", "entry_allowed"])
    return f"""def {f}(ticket, age_ok, banned, sold_out):
    return ticket and age_ok and not banned and not sold_out


def {f}_vip(ticket, vip, age_ok, banned):
    return ticket and vip and age_ok and not banned


def {f}_defaults():
    seated = True
    standing = False
    return seated or standing


def {f}_late(arrived, started, allowed_late):
    return arrived and started and allowed_late
"""


def t_bool_exam(rng):
    f = _pick(rng, ["exam_pass", "passed_exam", "exam_ok", "grade_pass"])
    return f"""def {f}(score, attended, cheated, retake):
    return score >= {rng.randint(40, 60)} and attended and not cheated and not retake


def {f}_distinction(score, attended, cheated):
    return score >= 85 and attended and not cheated


def {f}_defaults():
    graded = True
    moderated = False
    return graded and not moderated
"""


def t_bool_driving(rng):
    f = _pick(rng, ["licence_ok", "can_drive", "drive_allowed", "licence_valid"])
    return f"""def {f}(age, passed, suspended, insured):
    return age >= 17 and passed and not suspended and insured


def {f}_hire(age, licence, deposit, banned):
    return age >= {rng.randint(21, 25)} and licence and deposit and not banned


def {f}_defaults():
    automatic = True
    manual = False
    return automatic or manual
"""


def t_bool_hotel(rng):
    f = _pick(rng, ["booking_ok", "can_book", "room_ok", "stay_allowed"])
    return f"""def {f}(rooms_left, paid, banned, dates_ok):
    return rooms_left > 0 and paid and not banned and dates_ok


def {f}_cancel_free(days, paid, used, flexible):
    return days >= {rng.randint(2, 14)} and paid and not used and flexible


def {f}_defaults():
    breakfast = True
    parking = False
    return breakfast or parking
"""


def t_bool_course(rng):
    f = _pick(rng, ["can_enrol", "enrol_ok", "course_entry", "place_ok"])
    return f"""def {f}(prereq, places, paid, banned):
    return prereq and places > 0 and paid and not banned


def {f}_audit(prereq, places, member):
    return prereq and places > 0 and member


def {f}_defaults():
    online = True
    recorded = False
    return online and not recorded
"""


def t_bool_warranty(rng):
    f = _pick(rng, ["warranty_ok", "claim_ok", "covered", "warranty_valid"])
    return f"""def {f}(months, damaged, misuse, receipt):
    return months <= {rng.randint(12, 36)} and not damaged and not misuse and receipt


def {f}_extended(months, plan, receipt, misuse):
    return months <= 60 and plan and receipt and not misuse


def {f}_defaults():
    parts = True
    labour = False
    return parts or labour
"""


def t_bool_parking(rng):
    f = _pick(rng, ["parking_ok", "permit_ok", "can_park", "bay_allowed"])
    return f"""def {f}(permit, spaces, banned, paid):
    return permit and spaces > 0 and not banned and paid


def {f}_disabled(permit, badge, spaces):
    return permit and badge and spaces > 0


def {f}_defaults():
    overnight = True
    weekend = False
    return overnight or weekend
"""


def t_bool_volunteer(rng):
    f = _pick(rng, ["shift_ok", "can_volunteer", "volunteer_ok", "shift_allowed"])
    return f"""def {f}(trained, available, banned, age_ok):
    return trained and available and not banned and age_ok


def {f}_lead(trained, available, experience, banned):
    return trained and available and experience >= 2 and not banned


def {f}_defaults():
    weekday = True
    weekend = False
    return weekday or weekend
"""


def t_bool_newsletter(rng):
    f = _pick(rng, ["subscribed", "signup_ok", "mail_ok", "list_ok"])
    return f"""def {f}(email_ok, consent, bounced, unsubscribed):
    return email_ok and consent and not bounced and not unsubscribed


def {f}_digest(email_ok, consent, weekly, daily):
    return email_ok and consent and (weekly or daily)


def {f}_defaults():
    html = True
    plain = False
    return html or plain
"""


def t_bool_app_perm(rng):
    f = _pick(rng, ["perm_ok", "can_use_feature", "access_granted", "feature_ok"])
    return f"""def {f}(granted, os_ok, revoked, age_ok):
    return granted and os_ok and not revoked and age_ok


def {f}_camera(granted, os_ok, revoked):
    return granted and os_ok and not revoked


def {f}_defaults():
    location = True
    camera = False
    return location or camera
"""


def t_bool_device(rng):
    f = _pick(rng, ["device_ok", "settings_ok", "mode_ok", "device_ready"])
    return f"""def {f}(charged, updated, locked, airplane):
    return charged and updated and not locked and not airplane


def {f}_night(charged, quiet, locked, scheduled):
    return charged and quiet and not locked and scheduled


def {f}_defaults():
    wifi = True
    bluetooth = False
    return wifi or bluetooth
"""


def t_bool_pet(rng):
    f = _pick(rng, ["adopt_ok", "can_adopt", "pet_ok", "adoption_ok"])
    return f"""def {f}(age_ok, home_check, banned, fee_paid):
    return age_ok and home_check and not banned and fee_paid


def {f}_foster(age_ok, home_check, experience, banned):
    return age_ok and home_check and experience and not banned


def {f}_defaults():
    indoor = True
    garden = False
    return indoor or garden
"""


def t_bool_job(rng):
    f = _pick(rng, ["shortlist_ok", "can_hire", "hire_ok", "candidate_ok"])
    return f"""def {f}(qualified, experience, references, banned):
    return qualified and experience >= {rng.randint(1, 5)} and references and not banned


def {f}_interview(qualified, available, references):
    return qualified and available and references


def {f}_defaults():
    remote = True
    onsite = False
    return remote or onsite
"""


def t_bool_insurance(rng):
    f = _pick(rng, ["quote_ok", "insure_ok", "cover_ok", "policy_ok"])
    return f"""def {f}(age_ok, claims_free, licence, banned):
    return age_ok and claims_free and licence and not banned


def {f}_young_driver(age_ok, claims_free, blackbox, banned):
    return age_ok and claims_free and blackbox and not banned


def {f}_defaults():
    comprehensive = True
    third_party = False
    return comprehensive or third_party
"""


def t_bool_bank(rng):
    f = _pick(rng, ["transfer_ok", "can_transfer", "payment_ok", "send_ok"])
    return f"""def {f}(balance_ok, verified, frozen, limit_ok):
    return balance_ok and verified and not frozen and limit_ok


def {f}_instant(balance_ok, verified, frozen, fee_paid):
    return balance_ok and verified and not frozen and fee_paid


def {f}_defaults():
    instant = True
    scheduled = False
    return instant or scheduled
"""


def t_bool_school_trip(rng):
    f = _pick(rng, ["trip_ok", "can_attend_trip", "trip_allowed", "outing_ok"])
    return f"""def {f}(consent, paid, medical, banned):
    return consent and paid and medical and not banned


def {f}_swimming(consent, paid, swim_ok, banned):
    return consent and paid and swim_ok and not banned


def {f}_defaults():
    packed_lunch = True
    school_meal = False
    return packed_lunch or school_meal
"""


def t_bool_team(rng):
    f = _pick(rng, ["team_ok", "selected", "squad_ok", "pick_ok"])
    return f"""def {f}(fit, trained, suspended, age_ok):
    return fit and trained and not suspended and age_ok


def {f}_captain(fit, trained, experience, suspended):
    return fit and trained and experience >= {rng.randint(2, 5)} and not suspended


def {f}_defaults():
    home = True
    away = False
    return home or away
"""


def t_bool_recipe(rng):
    f = _pick(rng, ["recipe_ok", "meal_ok", "dish_ok", "recipe_fits"])
    return f"""def {f}(vegan, gluten_free, nut_free, dairy_free):
    return vegan or gluten_free or nut_free or dairy_free


def {f}_strict(vegan, gluten_free, nut_free, dairy_free):
    return vegan and gluten_free and nut_free and dairy_free


def {f}_defaults():
    spicy = True
    mild = False
    return spicy or mild
"""


BOOL_TEMPLATES = [
    t_bool_library, t_bool_gym, t_bool_event, t_bool_exam, t_bool_driving,
    t_bool_hotel, t_bool_course, t_bool_warranty, t_bool_parking,
    t_bool_volunteer, t_bool_newsletter, t_bool_app_perm, t_bool_device,
    t_bool_pet, t_bool_job, t_bool_insurance, t_bool_bank,
    t_bool_school_trip, t_bool_team, t_bool_recipe,
]


def _top_names(src: str) -> set[str]:
    return {n.name for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def build_sources() -> list[str]:
    """Deterministically build all template sources (deduped).

    Single-template sources plus COMPOSED sources (3 template
    functions concatenated into one module, with top-level name
    collisions skipped). Composition is what lifts the per-operator
    variant ceiling: a single template has few BoolOp/boolean sites,
    but a composed module sums the sites of its parts. Without it,
    swapped_boolean tops out near ~56 distinct variants across all
    single sources — far below the 1,750-per-operator target.
    """
    sources: list[str] = []
    seen: set[str] = set()

    def _add(src: str) -> bool:
        ast.parse(src)  # every source must itself be valid Python
        if src not in seen:
            seen.add(src)
            sources.append(src)
            return True
        return False

    for t_idx, template in enumerate(TEMPLATES + BOOL_TEMPLATES):
        for draw in range(PER_TEMPLATE):
            rng = random.Random(f"{MASTER_SEED}:{t_idx}:{draw}")
            _add(template(rng))

    def _composed(kind: str, count: int) -> None:
        made = 0
        attempt = 0
        while made < count and attempt < count * 20:
            rng = random.Random(f"{MASTER_SEED}:combo:{kind}:{attempt}")
            attempt += 1
            if kind == "bool":
                picks = rng.sample(BOOL_TEMPLATES, 2) + [rng.choice(TEMPLATES)]
            else:
                picks = rng.sample(TEMPLATES, 3)
            parts: list[str] = []
            names: set[str] = set()
            ok = True
            for j, template in enumerate(picks):
                sub = random.Random(f"{MASTER_SEED}:combo:{kind}:{attempt}:{j}")
                src = template(sub)
                tn = _top_names(src)
                if tn & names:
                    ok = False
                    break
                names |= tn
                parts.append(src)
            if ok and _add("\n\n".join(parts)):
                made += 1

    _composed("general", 800)
    _composed("bool", 800)
    return sources


def _pilot_buggy_codes() -> set[str]:
    pilot = HERE / "pilot" / "pilot_records.jsonl"
    codes: set[str] = set()
    if pilot.exists():
        for line in pilot.read_text(encoding="utf-8").splitlines():
            if line.strip():
                codes.add(json.loads(line)["buggy_code"])
    return codes


def to_text_example(rec: dict) -> str:
    """Same training-text format as pilot_manual_lora.py (replicated
    here so this script stays stdlib-only and import-safe)."""
    header = f"Fix this Python bug (bug class: {rec['bug_class']}):\n"
    return header + rec["buggy_code"] + "\nFixed code:\n" + rec["fixed_code"]


def _try_record(source, operators, seed, src_idx, seen_buggy):
    """Return (record, provenance) or None if unusable/duplicate."""
    try:
        rec = generate_record(
            source, list(operators), seed,
            filename=f"scaled_source_{src_idx}.py",
        )
    except (MutationNotApplicable, ValueError):
        return None
    if rec["buggy_code"] == rec["fixed_code"]:
        return None
    if rec["buggy_code"] in seen_buggy:
        return None
    if rec["failing_test"] != "":
        return None
    expected = operators[0] if len(operators) == 1 else f"composite:{'+'.join(operators)}"
    if rec["bug_class"] != expected:
        return None
    if [m["operator"] for m in rec["mutations"]] != list(operators):
        return None
    ast.parse(rec["buggy_code"])
    ast.parse(rec["fixed_code"])
    seen_buggy.add(rec["buggy_code"])
    prov = {"source_index": src_idx, "operators": list(operators), "seed": seed}
    return rec, prov


def collect(sources: list[str]) -> tuple[list[dict], list[dict]]:
    seen_buggy = _pilot_buggy_codes()
    records: list[dict] = []
    provs: list[dict] = []

    # --- Singles: exactly TARGET_SINGLE_PER_OP per operator. ---
    # Seed efficiency note: for a fixed (source, operator), the
    # (candidate, delta) pair sequence is periodic, so once seeds
    # start repeating already-seen buggy codes they will keep doing
    # so; stop a source after 5 consecutive rejections instead of
    # burning 60 blind seeds per source (the first run wasted most
    # of its time this way, and still starved on capped operators).
    for operator in OPERATORS:
        got = 0
        for src_idx, source in enumerate(sources):
            if got >= TARGET_SINGLE_PER_OP:
                break
            misses = 0
            for seed in range(400):
                if got >= TARGET_SINGLE_PER_OP:
                    break
                hit = _try_record(source, [operator], seed, src_idx, seen_buggy)
                if hit is not None:
                    records.append(hit[0])
                    provs.append(hit[1])
                    got += 1
                    misses = 0
                else:
                    misses += 1
                    if misses >= 5:
                        break
        if got < TARGET_SINGLE_PER_OP:
            raise RuntimeError(
                f"single operator {operator}: only {got}/{TARGET_SINGLE_PER_OP} "
                f"distinct variants across {len(sources)} sources")
        print(f"singles {operator}: {got}", flush=True)

    # --- Composites: TARGET_COMPOSITE, round-robin over sources x
    # pairs (one seed per pass) so pair counts stay roughly balanced
    # instead of the first pair filling the whole quota. ---
    n_single = len(records)
    for seed in range(5000, 5200):
        if len(records) >= TARGET_TOTAL:
            break
        progress = len(records)
        for src_idx, source in enumerate(sources):
            if len(records) >= TARGET_TOTAL:
                break
            for pair in COMPOSITE_PAIRS:
                if len(records) >= TARGET_TOTAL:
                    break
                hit = _try_record(source, list(pair), seed, src_idx, seen_buggy)
                if hit is not None:
                    records.append(hit[0])
                    provs.append(hit[1])
        if len(records) == progress and seed > 5000:
            raise RuntimeError(f"no progress for composites at seed {seed}")
    print(f"composites: {len(records) - n_single}", flush=True)

    assert len(records) == TARGET_TOTAL, len(records)
    assert len(provs) == TARGET_TOTAL
    return records, provs


def token_survey(records: list[dict]) -> dict:
    """Chars/4 token ESTIMATE on the training-text format (no local
    Gemma tokenizer exists in this VM). Calibrated below against the
    pilot file, whose real-tokenizer survey is known (median 243,
    p90 317, max 383). The Kaggle full-train run must re-survey with
    the real tokenizer; these numbers are an estimate, not a claim."""
    est = sorted((len(to_text_example(r)) + 3) // 4 for r in records)
    n = len(est)
    return {
        "method": "chars/4 estimate on to_text_example format (NOT a real tokenizer)",
        "n": n,
        "median": est[n // 2],
        "p90": est[int(0.9 * (n - 1))],
        "max": est[-1],
    }


def pilot_calibration() -> dict:
    """Same estimator on the pilot file, for comparison with its known
    real-tokenizer survey (median 243, p90 317, max 383)."""
    pilot = HERE / "pilot" / "pilot_records.jsonl"
    if not pilot.exists():
        return {}
    recs = [json.loads(line) for line in pilot.read_text(encoding="utf-8").splitlines() if line.strip()]
    est = sorted((len(to_text_example(r)) + 3) // 4 for r in recs)
    n = len(est)
    return {
        "n": n,
        "estimate_median": est[n // 2],
        "estimate_p90": est[int(0.9 * (n - 1))],
        "estimate_max": est[-1],
        "real_tokenizer_median": 243,
        "real_tokenizer_p90": 317,
        "real_tokenizer_max": 383,
    }


def main() -> None:
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = build_sources()
    print(f"templates: {len(TEMPLATES) + len(BOOL_TEMPLATES)}, distinct sources: {len(sources)}", flush=True)
    records, provs = collect(sources)
    data_path = out_dir / "train_records_20k.jsonl"
    prov_path = out_dir / "train_records_20k_provenance.jsonl"
    with data_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with prov_path.open("w", encoding="utf-8") as fh:
        for prov in provs:
            fh.write(json.dumps(prov) + "\n")
    counts = Counter(r["bug_class"] for r in records)
    singles = {op: counts[op] for op in OPERATORS}
    n_comp = sum(v for k, v in counts.items() if k.startswith("composite:"))
    print(f"records: {len(records)} ({sum(singles.values())} single, {n_comp} composite)")
    print(f"singles per operator: {json.dumps(singles, sort_keys=True)}")
    print(f"composite pairs: {json.dumps({k: v for k, v in sorted(counts.items()) if k.startswith('composite:')}, sort_keys=True)}")
    print(f"file: {data_path} ({data_path.stat().st_size} bytes)")
    print(f"provenance: {prov_path}")
    print(f"token survey (estimate): {json.dumps(token_survey(records))}")
    print(f"pilot calibration: {json.dumps(pilot_calibration())}")


if __name__ == "__main__":
    main()
