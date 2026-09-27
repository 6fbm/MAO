"""Small text helpers used across layers (truncation and formatting)."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the head and the tail of ``text``; the tail usually holds errors/results."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(text) - head - tail
    marker = f"\n… [{omitted} characters omitted] …\n"
    return text[:head] + marker + text[-tail:], True


def truncate_end(text: str, max_chars: int, suffix: str = "…") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(suffix))] + suffix


def one_line(text: str | None, max_len: int = 120) -> str:
    if not text:
        return ""
    return truncate_end(_WS_RE.sub(" ", text).strip(), max_len)


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_cost(usd: float | None) -> str:
    if usd is None:
        return "n/a"
    if usd < 0.01 and usd > 0:
        return f"${usd:.4f}"
    return f"${usd:,.2f}"


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())
