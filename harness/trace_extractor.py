"""Trace extractor: parse a failing test run + issue text into structured facts.

Stdlib only. No model calls, no network.

Input: the text output of a failing test run (``python -m unittest`` style,
which is what :mod:`test_runner` produces) plus the free-text issue
description. Output: :class:`TraceInfo` with the failing test, traceback
frames, implicated files (in first-seen order) and implicated symbols.

The parser is deliberately tolerant: it accepts pytest-style output too
(``FAILED path::test`` lines and ``File "...", line N, in func`` frames),
because the same extractor is used when a developer pastes either format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_FAILED_UNITTEST_RE = re.compile(r"FAIL:\s+(\S+)\s+\(([^)]+)\)")
_FAILED_PYTEST_RE = re.compile(r"FAILED\s+(\S+::\S+)")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

# Words that look like identifiers but carry no code meaning.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "when", "then",
    "should", "expected", "actual", "error", "failed", "failure", "test",
    "file", "line", "return", "returns", "value", "result", "function",
}


@dataclass
class Frame:
    file: str
    line: int
    function: str


@dataclass
class TraceInfo:
    failing_tests: list[str] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    implicated_files: list[str] = field(default_factory=list)
    implicated_symbols: list[str] = field(default_factory=list)
    error_message: str = ""

    @property
    def primary_file(self) -> str | None:
        return self.implicated_files[0] if self.implicated_files else None


def _add_ordered(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def extract_trace(test_output: str, issue_text: str = "") -> TraceInfo:
    """Parse test output + issue text into a :class:`TraceInfo`."""
    info = TraceInfo()
    lines = test_output.splitlines()

    for line in lines:
        m = _FAILED_UNITTEST_RE.search(line)
        if m:
            _add_ordered(info.failing_tests, f"{m.group(2)}.{m.group(1)}")
            continue
        m = _FAILED_PYTEST_RE.search(line)
        if m:
            _add_ordered(info.failing_tests, m.group(1))
            continue
        m = _FRAME_RE.search(line)
        if m:
            info.frames.append(
                Frame(file=m.group(1), line=int(m.group(2)), function=m.group(3))
            )
            _add_ordered(info.implicated_files, m.group(1))
            _add_ordered(info.implicated_symbols, m.group(3))

    # Error message: last non-empty line that looks like an assertion/error.
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and (
            "Error" in stripped or "AssertionError" in stripped or stripped.startswith("assert")
        ):
            info.error_message = stripped
            break
    else:
        for line in reversed(lines):
            if line.strip():
                info.error_message = line.strip()
                break

    # Symbols from the issue text: backticked identifiers first (highest
    # signal), then function-like names (identifier followed by "(").
    for token in _BACKTICK_RE.findall(issue_text):
        token = token.strip()
        if _IDENTIFIER_RE.fullmatch(token):
            _add_ordered(info.implicated_symbols, token)
    for token in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", issue_text):
        if token.lower() not in _STOPWORDS:
            _add_ordered(info.implicated_symbols, token)

    # Files mentioned in the issue text (path-like tokens ending in .py).
    for token in re.findall(r"[\w./-]+\.py", issue_text):
        _add_ordered(info.implicated_files, token)

    return info
