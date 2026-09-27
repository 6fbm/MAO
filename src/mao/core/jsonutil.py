"""Tolerant JSON extraction from model output.

Models wrap JSON in prose or code fences, add trailing commas or use smart
quotes. We try the cheap, safe repairs and otherwise return ``None`` so the
caller can ask the model to fix its output.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON|javascript|js)?[ \t]*\r?\n?(.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_MAX_SCAN_STARTS = 25


def _try_load(candidate: str) -> Any | None:
    candidate = candidate.strip()
    if not candidate or candidate[0] not in "{[":
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    repaired = (
        candidate.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    )
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _balanced_spans(text: str) -> list[str]:
    """Return substrings that start at '{' or '[' and end at the matching bracket."""
    spans: list[str] = []
    starts = [i for i, ch in enumerate(text) if ch in "{["][:_MAX_SCAN_STARTS]
    for start in starts:
        stack: list[str] = []
        in_string = False
        escaped = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or stack.pop() != ch:
                    break
                if not stack:
                    spans.append(text[start : pos + 1])
                    break
    # prefer the longest span: models usually put the real payload outermost
    spans.sort(key=len, reverse=True)
    return spans


def extract_json(text: str | None) -> Any | None:
    if not text:
        return None
    stripped = text.strip()
    value = _try_load(stripped)
    if value is not None:
        return value
    for match in _FENCE_RE.finditer(stripped):
        value = _try_load(match.group(1))
        if value is not None:
            return value
    for span in _balanced_spans(stripped):
        value = _try_load(span)
        if value is not None:
            return value
    return None


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    value = extract_json(text)
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return None


def dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def dumps_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
