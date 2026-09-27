"""Removes secrets from any text before it is logged, stored or displayed.

Two layers: exact values of every key we loaded (registered at runtime) and
generic patterns for common key formats (in case a key shows up that we
never loaded, e.g. inside a file an agent read).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

REDACTED = "***REDACTED***"

_VALUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-(?:proj-|svcacct-|or-v1-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"xai-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
]

# keyword = value  (keeps the keyword, hides the value). The value must look
# like a credential (letters and digits, >= 12 chars) to avoid hiding normal
# text such as "max_tokens: 100000".
_KEYED_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|x-api-key|x-goog-api-key|authorization|client[_-]?secret|secret[_-]?key|"
    r"access[_-]?token|auth[_-]?token|password|passwd)\s*[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?)"
    r"((?=[^\s\"',;]*[A-Za-z])(?=[^\s\"',;]*\d)[^\s\"',;]{12,})"
)
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]{16,})")


class Redactor:
    def __init__(self) -> None:
        self._secrets: set[str] = set()
        self._lock = threading.Lock()

    def add_secret(self, value: str | None) -> None:
        if value and len(value.strip()) >= 8:
            with self._lock:
                self._secrets.add(value.strip())

    def redact(self, text: str | None) -> str:
        if not text:
            return text or ""
        with self._lock:
            secrets = sorted(self._secrets, key=len, reverse=True)
        for secret in secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
        for pattern in _VALUE_PATTERNS:
            text = pattern.sub(REDACTED, text)
        text = _KEYED_PATTERN.sub(lambda m: m.group(1) + REDACTED, text)
        text = _BEARER_PATTERN.sub(lambda m: m.group(1) + REDACTED, text)
        return text

    def redact_obj(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {k: self.redact_obj(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact_obj(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.redact_obj(v) for v in value)
        return value


# process-wide instance: every key loaded anywhere is registered here
REDACTOR = Redactor()


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor or REDACTOR

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            message = str(record.msg)
        record.msg = self._redactor.redact(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            formatter = logging.Formatter()
            record.exc_text = self._redactor.redact(formatter.formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self._redactor.redact(record.exc_text)
        return True
