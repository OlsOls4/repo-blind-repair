"""32K context packer: assemble rung-ordered file contents under a token cap.

Token estimate is deliberately crude and conservative: ``ceil(chars / 4)``.
Real tokenisers vary, but a chars/4 estimate is the convention this project
uses everywhere, so the cap is comparable across runs. The packer never
exceeds ``token_budget``: a file that does not fit whole is truncated with
an explicit ``[... truncated ...]`` marker, and later files are dropped.

Rungs (see :mod:`scope_ladder`):
  1 = files named by the failing trace,
  2 = their imports and callers,
  3 = wider repo search.

Per-rung time budgets (seconds) are recorded on the pack so the ladder
controller can enforce them; the packer itself does not run anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 32768
DEFAULT_RUNG_TIME_BUDGETS = {1: 30, 2: 60, 3: 120}


def estimate_tokens(text: str) -> int:
    """Estimate tokens as ceil(len(text) / 4)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


@dataclass
class ContextPack:
    text: str
    tokens_used: int
    token_budget: int
    files_included: list[str] = field(default_factory=list)
    truncated_files: list[str] = field(default_factory=list)
    dropped_files: list[str] = field(default_factory=list)
    rung_time_budgets: dict[int, int] = field(default_factory=dict)

    @property
    def headroom(self) -> int:
        return self.token_budget - self.tokens_used


def pack_context(
    rung_files: dict[int, dict[str, str]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    rung_time_budgets: dict[int, int] | None = None,
) -> ContextPack:
    """Pack ``{rung: {path: content}}`` in rung order under ``token_budget``.

    Files already included under an earlier rung are not repeated.
    """
    budgets = dict(DEFAULT_RUNG_TIME_BUDGETS)
    if rung_time_budgets:
        budgets.update(rung_time_budgets)

    chunks: list[str] = []
    included: list[str] = []
    truncated: list[str] = []
    dropped: list[str] = []
    used = 0

    for rung in sorted(rung_files):
        for path, content in rung_files[rung].items():
            if path in included:
                continue
            header = f"--- {path} (rung {rung}) ---\n"
            body_tokens = estimate_tokens(content)
            header_tokens = estimate_tokens(header)
            if used + header_tokens + body_tokens <= token_budget:
                chunks.append(header + content)
                included.append(path)
                used += header_tokens + body_tokens
                continue
            # File does not fit whole: truncate it if there is room for a
            # header plus at least some body, otherwise drop it.
            remaining = token_budget - used - header_tokens
            if remaining > 0:
                keep_chars = remaining * CHARS_PER_TOKEN
                marker = "\n[... truncated to fit token budget ...]\n"
                chunks.append(header + content[:keep_chars] + marker)
                included.append(path)
                truncated.append(path)
                used = token_budget
            else:
                dropped.append(path)
            if used >= token_budget:
                break
        if used >= token_budget:
            break

    # Files never reached because the budget ran out count as dropped,
    # so callers can see exactly what the agent did not get to read.
    if used >= token_budget:
        seen = set(included) | set(dropped)
        for rung in sorted(rung_files):
            for path in rung_files[rung]:
                if path not in seen:
                    dropped.append(path)
                    seen.add(path)

    return ContextPack(
        text="\n".join(chunks),
        tokens_used=used,
        token_budget=token_budget,
        files_included=included,
        truncated_files=truncated,
        dropped_files=dropped,
        rung_time_budgets=budgets,
    )
