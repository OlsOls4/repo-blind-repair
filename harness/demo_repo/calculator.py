"""Toy buggy module for the harness demo.

Bug (deliberate): ``add`` subtracts instead of adding. Everything else
is correct, so a patch that fixes ``add`` without touching ``multiply``
or ``subtract`` passes the regression gate.
"""


def add(a, b):
    return a - b


def multiply(a, b):
    return a * b


def subtract(a, b):
    return a - b
