"""HTTP helpers: error classification and Retry-After parsing."""

from __future__ import annotations

import email.utils
import re
import time

import httpx

from mao.core.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    TransientProviderError,
)
from mao.security.redaction import REDACTOR

_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "exceeds the maximum number of tokens",
    "reduce the length",
    "input token count",
    "request too large",
    "maximum prompt length",
)
_QUOTA_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota, please check your plan and billing",
    "credit balance is too low",
    "billing",
    "per day",
    "perday",
    "payment required",
)
_AUTH_MARKERS = ("api key not valid", "invalid api key", "incorrect api key", "invalid x-api-key", "unauthorized")
_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def _parse_duration(value: str) -> float | None:
    value = value.strip().lower()
    try:
        return float(value)
    except ValueError:
        pass
    match = _DURATION_RE.match(value)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_retry_after(headers: httpx.Headers) -> float | None:
    if "retry-after-ms" in headers:
        try:
            return max(0.0, float(headers["retry-after-ms"]) / 1000)
        except ValueError:
            pass
    if "retry-after" in headers:
        raw = headers["retry-after"]
        try:
            return max(0.0, float(raw))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(raw) if raw else None
            if parsed is not None:
                return max(0.0, parsed.timestamp() - time.time())
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        if name in headers:
            duration = _parse_duration(headers[name])
            if duration is not None:
                return duration
    return None


def extract_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:400]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            parts = [str(error.get(k)) for k in ("message", "type", "code", "status") if error.get(k)]
            return " | ".join(parts) or str(error)
        if isinstance(error, str):
            return error
        for key in ("message", "detail"):
            if data.get(key):
                return str(data[key])
    return str(data)[:400]


def map_http_error(provider: str, response: httpx.Response) -> ProviderError:
    status = response.status_code
    message = REDACTOR.redact(extract_error_message(response)) or f"HTTP {status}"
    lower = message.lower()
    kwargs = {
        "provider": provider,
        "status_code": status,
        "retry_after": parse_retry_after(response.headers),
        "body_excerpt": REDACTOR.redact(response.text[:500]),
    }
    if status in (401, 403):
        if any(m in lower for m in _QUOTA_MARKERS):
            return QuotaExceededError(message, **kwargs)
        return AuthenticationError(message, **kwargs)
    if status == 402:
        return QuotaExceededError(message, **kwargs)
    if status == 404:
        return ModelNotFoundError(message, **kwargs)
    if status == 413:
        return ContextLengthError(message, **kwargs)
    if status == 429:
        if any(m in lower for m in _QUOTA_MARKERS):
            return QuotaExceededError(message, **kwargs)
        return RateLimitError(message, **kwargs)
    if status in (400, 422):
        if any(m in lower for m in _CONTEXT_MARKERS):
            return ContextLengthError(message, **kwargs)
        if any(m in lower for m in _AUTH_MARKERS):
            return AuthenticationError(message, **kwargs)
        if "credit balance" in lower:
            return QuotaExceededError(message, **kwargs)
        return InvalidRequestError(message, **kwargs)
    if status == 408 or status >= 500:
        return TransientProviderError(message, **kwargs)
    return InvalidRequestError(message, **kwargs)


def map_transport_error(provider: str, exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return TransientProviderError(f"Timeout ({type(exc).__name__})", provider=provider)
    if isinstance(exc, httpx.ConnectError):
        return TransientProviderError(f"Connection failed: {REDACTOR.redact(str(exc))}", provider=provider)
    return TransientProviderError(f"Network error {type(exc).__name__}: {REDACTOR.redact(str(exc))}", provider=provider)
