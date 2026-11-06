"""Deterministic AST-based mutation operators for Python source.

Design rules (Repo-Blind Repair):
- No teacher models, no real human bug-fixes, no network access.
- Every mutation is reproducible from (operator, seed): the same
  (source, operator, seed) always yields the same mutated source.
- Every mutation records exactly what changed: file, line, operator,
  and the original vs mutated code snippet (via ast.unparse).
- Every mutated file must still parse (ast.parse); operators that
  cannot apply cleanly raise MutationNotApplicable instead of
  producing broken code.

Candidate selection: for a given operator we walk the AST in
deterministic (ast.walk) order, collect candidate nodes, and pick
candidate ``seed % len(candidates)``. The transformer then mutates
only that one candidate (tracked by traversal order), leaving all
other identical nodes untouched.
"""

from __future__ import annotations

import ast
import copy


class MutationNotApplicable(Exception):
    """Raised when an operator has no applicable candidate in the source."""


# Operator names exposed by this module.
OPERATORS = (
    "off_by_one",
    "swapped_operands",
    "wrong_comparator",
    "dropped_condition",
    "swapped_boolean",
    "wrong_constant",
    "removed_return",
    "removed_assignment",
)


def _is_int_constant(node) -> bool:
    # bool is a subclass of int in Python; exclude it explicitly.
    return isinstance(node, ast.Constant) and type(node.value) is int


def _iter_dfs(node: ast.AST):
    """Yield nodes in depth-first pre-order, matching NodeTransformer."""
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _iter_dfs(child)


def _candidates(tree: ast.AST, operator: str) -> list[ast.AST]:
    """Collect candidate nodes for ``operator`` in deterministic order."""
    out: list[ast.AST] = []
    for node in _iter_dfs(tree):
        if operator == "off_by_one":
            if _is_int_constant(node):
                out.append(node)
        elif operator == "swapped_operands":
            if isinstance(node, ast.BinOp):
                out.append(node)
        elif operator == "wrong_comparator":
            if (
                isinstance(node, ast.Compare)
                and node.ops
                and type(node.ops[0]) in _CMP_SWAP
            ):
                out.append(node)
        elif operator == "dropped_condition":
            if isinstance(node, ast.BoolOp) and len(node.values) >= 2:
                out.append(node)
        elif operator == "swapped_boolean":
            if isinstance(node, ast.Constant) and type(node.value) is bool:
                out.append(node)
        elif operator == "wrong_constant":
            if _is_int_constant(node):
                out.append(node)
        elif operator == "removed_return":
            if isinstance(node, ast.Return) and node.value is not None:
                out.append(node)
        elif operator == "removed_assignment":
            if isinstance(node, ast.Assign):
                out.append(node)
    return out


_CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


class _Mutator(ast.NodeTransformer):
    """Mutates only the ``target_index``-th candidate for ``operator``."""

    def __init__(self, operator: str, seed: int, target_index: int):
        self.operator = operator
        self.seed = seed
        self.target_index = target_index
        self._seen = -1  # counts candidates of this operator, in walk order
        self.original_snippet: str | None = None
        self.mutated_snippet: str | None = None
        self.line: int | None = None

    def _is_target(self, node) -> bool:
        self._seen += 1
        return self._seen == self.target_index

    def _record(self, node, mutated_node):
        self.original_snippet = ast.unparse(node)
        self.mutated_snippet = ast.unparse(mutated_node)
        self.line = getattr(node, "lineno", None)

    def visit_Constant(self, node):
        if self.operator == "off_by_one" and _is_int_constant(node):
            if self._is_target(node):
                delta = 1 if self.seed % 2 == 0 else -1
                new = copy.copy(node)
                new.value = node.value + delta
                self._record(node, new)
                return new
            return node
        if self.operator == "swapped_boolean" and type(node.value) is bool:
            if self._is_target(node):
                new = copy.copy(node)
                new.value = not node.value
                self._record(node, new)
                return new
            return node
        if self.operator == "wrong_constant" and _is_int_constant(node):
            if self._is_target(node):
                # Delta >= 2, so this never coincides with off_by_one (+-1).
                delta = 2 + (self.seed % 5)
                new = copy.copy(node)
                new.value = node.value + delta
                self._record(node, new)
                return new
            return node
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.operator == "swapped_operands":
            if self._is_target(node):
                new = copy.copy(node)
                new.left, new.right = node.right, node.left
                self._record(node, new)
                return new
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if self.operator == "wrong_comparator":
            if self._is_target(node):
                op_type = type(node.ops[0])
                if op_type not in _CMP_SWAP:
                    return self.generic_visit(node)
                new = copy.copy(node)
                new.ops = [_CMP_SWAP[op_type]()] + list(node.ops[1:])
                self._record(node, new)
                return new
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.operator == "dropped_condition":
            if self._is_target(node):
                drop = self.seed % len(node.values)
                new = copy.copy(node)
                new.values = [v for i, v in enumerate(node.values) if i != drop]
                self._record(node, new)
                return new
        return self.generic_visit(node)

    def visit_Return(self, node):
        if self.operator == "removed_return" and node.value is not None:
            if self._is_target(node):
                new = copy.copy(node)
                new.value = None
                self._record(node, new)
                return new
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if self.operator == "removed_assignment":
            if self._is_target(node):
                # Replace the assignment with `pass`: the file still
                # parses and the assignment's effect is removed.
                new = ast.Pass()
                ast.copy_location(new, node)
                self._record(node, new)
                return new
        return self.generic_visit(node)


def apply_mutation(source: str, operator: str, seed: int, filename: str = "<string>") -> dict:
    """Apply one deterministic mutation. Returns a mutation record.

    Record keys: filename, operator, seed, line, original_code,
    mutated_code, mutated_source. ``mutated_source`` is the full
    mutated file; ``original_code``/``mutated_code`` are the snippets
    of the single changed node.
    """
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator: {operator!r} (known: {OPERATORS})")
    tree = ast.parse(source)
    candidates = _candidates(tree, operator)
    if not candidates:
        raise MutationNotApplicable(
            f"operator {operator!r} has no candidate in {filename}"
        )
    target_index = seed % len(candidates)
    mutator = _Mutator(operator, seed, target_index)
    new_tree = mutator.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(new_tree)
    mutated_source = ast.unparse(new_tree)
    # The mutated file must still parse; unparse output always should,
    # but verify rather than assume.
    ast.parse(mutated_source)
    if mutator.original_snippet is None:
        raise MutationNotApplicable(
            f"operator {operator!r} did not mutate anything in {filename}"
        )
    return {
        "filename": filename,
        "operator": operator,
        "seed": seed,
        "line": mutator.line,
        "original_code": mutator.original_snippet,
        "mutated_code": mutator.mutated_snippet,
        "mutated_source": mutated_source,
    }
