"""Synthetic-bug generator: clean code -> buggy code + training record.

No teacher models, no real human bug-fixes, no network access. The
only "bugs" produced are deterministic AST mutations of clean code
(see operators.py), so the fixed code is exactly the original code
and the ground truth is known by construction.

Training record schema:
    {
      "buggy_code": str,        # full mutated file
      "fixed_code": str,        # full original file
      "mutations": [ {...}, ],  # one record per applied operator
      "failing_test": str,      # "" unless a test is actually derived
      "bug_class": str,         # operator, or "composite:op1+op2"
    }

failing_test policy: do NOT invent a test that does not exist. This
generator never synthesises a test case; the field is left empty
("") unless a caller supplies a real, runnable test that was observed
to fail on the buggy code (see demo.py, which does not claim one).
"""

from __future__ import annotations

import ast

# Import works both as a package module and as a plain script directory.
try:
    from operators import MutationNotApplicable, apply_mutation  # noqa: F401
except ImportError:  # pragma: no cover
    from .operators import MutationNotApplicable, apply_mutation  # type: ignore  # noqa: F401


def _strip_mutated_source(record: dict) -> dict:
    """Return a mutation record without the full mutated source."""
    return {k: v for k, v in record.items() if k != "mutated_source"}


def generate_record(
    source: str,
    operators: list[str],
    seed: int,
    filename: str = "<string>",
    failing_test: str = "",
) -> dict:
    """Generate one training record.

    ``operators`` has 1 entry (single-operator) or 2 entries
    (composite). Each mutation is applied to the output of the
    previous one. For the second mutation we try seed offsets
    (seed, seed+1, ...) so a composite record is still produced when
    the first mutation removed the second operator's only candidate;
    the actually-used seeds are recorded in each mutation entry.
    """
    if not operators or len(operators) > 2:
        raise ValueError("operators must contain 1 or 2 entries")
    # The input must parse before we mutate it.
    ast.parse(source)
    current = source
    applied: list[dict] = []
    for step, operator in enumerate(operators):
        last_error: Exception | None = None
        for offset in range(10):
            try:
                record = apply_mutation(
                    current, operator, seed + offset, filename=filename
                )
                break
            except MutationNotApplicable as exc:
                last_error = exc
        else:
            raise MutationNotApplicable(
                f"operator {operator!r} could not be applied after "
                f"{step} prior mutation(s): {last_error}"
            )
        applied.append(_strip_mutated_source(record))
        current = record["mutated_source"]
    bug_class = (
        operators[0] if len(operators) == 1 else f"composite:{'+'.join(operators)}"
    )
    return {
        "buggy_code": current,
        "fixed_code": source,
        "mutations": applied,
        "failing_test": failing_test,
        "bug_class": bug_class,
    }


def generate_single(source: str, operator: str, seed: int, filename: str = "<string>") -> dict:
    """Convenience wrapper for a 1-mutation record."""
    return generate_record(source, [operator], seed, filename=filename)


def generate_composite(
    source: str, operator1: str, operator2: str, seed: int, filename: str = "<string>"
) -> dict:
    """Convenience wrapper for a 2-mutation record."""
    return generate_record(source, [operator1, operator2], seed, filename=filename)
