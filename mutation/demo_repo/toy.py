"""Toy module for the mutation demo. Clean, original code written
for this demo only — not taken from any real project or bug-fix."""


def sum_upto(n):
    total = 0
    for i in range(n):
        total = total + i
    return total


def is_even(n):
    return n % 2 == 0


def find_max(values):
    best = values[0]
    for v in values:
        if v > best and v != 0:
            best = v
    return best


def greet(flag):
    if flag:
        return "hello"
    return "bye"


def is_enabled():
    return True
