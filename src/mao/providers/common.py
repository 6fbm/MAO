"""Helpers shared by provider implementations."""

from __future__ import annotations

import json
import re
from typing import Any

from mao.core.types import ChatMessage

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def deep_merge(target: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Tool arguments arrive as JSON strings or objects depending on the API."""
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"Arguments are not valid JSON: {exc.msg}"
        if isinstance(value, dict):
            return value, None
        return {}, "Arguments must be a JSON object"
    return {}, f"Unerwarteter Argumenttyp: {type(raw).__name__}"


def native_history(message: ChatMessage, *, wire_format: str, provider: str, model: str) -> dict[str, Any] | None:
    """Return the stored native payload if it may be replayed to this provider+model."""
    data = message.provider_data
    if not data:
        return None
    if data.get("format") != wire_format or data.get("provider") != provider or data.get("model") != model:
        return None
    return data


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")
