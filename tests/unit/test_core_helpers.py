from __future__ import annotations

import asyncio

import httpx
import pytest

from mao.config.schema import PricingConfig, RateLimitConfig
from mao.core.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    ModelNotFoundError,
    NoAvailableKeyError,
    QuotaExceededError,
    RateLimitError,
    TransientProviderError,
)
from mao.core.jsonutil import extract_json, extract_json_object
from mao.core.text import fmt_duration, fmt_int, truncate_middle
from mao.core.types import Usage
from mao.providers.http import map_http_error, parse_retry_after
from mao.providers.keypool import KeyPool
from mao.providers.ratelimit import ProviderLimiter, SlidingWindow
from mao.tokens.tracker import UsageRecord, UsageTracker, compute_cost, estimate_cost


# ------------------------------------------------------------------ json


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('Here is the result:\n```json\n{"a": [1, 2,]}\n```\nDone.', {"a": [1, 2]}),
        ('Text {"msg": "brace } in string", "n": {"x": 2}} end', {"msg": "brace } in string", "n": {"x": 2}}),
        ("no json", None),
        ("[1, 2, 3]", [1, 2, 3]),
    ],
)
def test_extract_json(text: str, expected: object) -> None:
    assert extract_json(text) == expected


def test_extract_json_object_prefers_outer_object() -> None:
    text = 'Plan: {"steps": [{"id": "s1"}, {"id": "s2"}], "title": "x"}'
    assert extract_json_object(text) == {"steps": [{"id": "s1"}, {"id": "s2"}], "title": "x"}


def test_text_helpers() -> None:
    out, truncated = truncate_middle("a" * 100 + "b" * 100, 50)
    assert truncated and out.startswith("a") and out.endswith("b")
    assert fmt_int(128430) == "128,430"
    assert fmt_duration(271) == "04:31"


# ------------------------------------------------------------------ http errors


def _response(status: int, body: object, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers or {})


@pytest.mark.parametrize(
    ("status", "body", "error_type"),
    [
        (401, {"error": {"message": "Incorrect API key provided"}}, AuthenticationError),
        (400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}, AuthenticationError),
        (429, {"error": {"message": "Rate limit reached", "type": "requests"}}, RateLimitError),
        (429, {"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}}, QuotaExceededError),
        (400, {"error": {"message": "This model's maximum context length is 8192 tokens", "code": "context_length_exceeded"}}, ContextLengthError),
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 250000 tokens > 200000 maximum"}}, ContextLengthError),
        (400, {"error": {"message": "Unsupported parameter: 'temperature'"}}, InvalidRequestError),
        (404, {"error": "model 'x' not found"}, ModelNotFoundError),
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, TransientProviderError),
        (503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}, TransientProviderError),
    ],
)
def test_map_http_error(status: int, body: object, error_type: type) -> None:
    error = map_http_error("p", _response(status, body))
    assert type(error) is error_type
    assert error.status_code == status


def test_retry_after_parsing() -> None:
    assert parse_retry_after(httpx.Headers({"retry-after": "7"})) == 7
    assert parse_retry_after(httpx.Headers({"retry-after-ms": "1500"})) == 1.5
    assert parse_retry_after(httpx.Headers({"x-ratelimit-reset-requests": "1m30s"})) == 90
    assert parse_retry_after(httpx.Headers({})) is None


# ------------------------------------------------------------------ key pool


def test_keypool_rotation_and_cooldown() -> None:
    now = [1000.0]
    pool = KeyPool("openai", [("key-aaaaaaaa1", "env:A"), ("key-bbbbbbbb2", "store")], clock=lambda: now[0])
    first = pool.acquire()
    assert first is not None
    pool.release(first)
    pool.report_rate_limited(first, retry_after=30)
    assert pool.has_other_ready_key(first)
    second = pool.acquire()
    assert second is not None and second.key != first.key
    pool.release(second)
    pool.report_invalid(second, "401")
    with pytest.raises(NoAvailableKeyError) as info:
        pool.acquire()
    assert info.value.wait_seconds == pytest.approx(30)
    now[0] += 31
    again = pool.acquire()
    assert again is not None and again.key == first.key
    pool.release(again)
    pool.report_invalid(again, "quota")
    with pytest.raises(NoAvailableKeyError):
        pool.acquire()
    assert not pool.is_usable()


def test_keypool_without_keys() -> None:
    assert KeyPool("ollama", [], requires_key=False).acquire() is None
    with pytest.raises(NoAvailableKeyError):
        KeyPool("openai", [], requires_key=True).acquire()


# ------------------------------------------------------------------ rate limiting


async def test_sliding_window_blocks_until_window_passes() -> None:
    clock = [0.0]
    window = SlidingWindow(2, clock=lambda: clock[0], window_s=60)
    await window.acquire()
    await window.acquire()
    task = asyncio.create_task(window.acquire())
    await asyncio.sleep(0.1)
    assert not task.done()
    clock[0] = 61
    await asyncio.wait_for(task, 2)
    assert window.used() == 1


async def test_limiter_concurrency() -> None:
    limiter = ProviderLimiter(RateLimitConfig(max_concurrency=2))
    active = 0
    peak = 0

    async def work() -> None:
        nonlocal active, peak
        async with limiter.slot(10):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(work() for _ in range(6)))
    assert peak == 2


# ------------------------------------------------------------------ costs


def test_cost_calculation() -> None:
    pricing = PricingConfig(
        input_per_mtok=2.0,
        cached_input_per_mtok=0.2,
        output_per_mtok=12.0,
        long_context_threshold=272_000,
        long_input_per_mtok=4.0,
        long_output_per_mtok=24.0,
    )
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cached_input_tokens=500_000)
    # long context: 500k*4 + 500k*0.4 + 100k*24 per million
    assert compute_cost(usage, pricing) == pytest.approx(2.0 + 0.2 + 2.4)
    assert estimate_cost(10_000, 1_000, pricing) == pytest.approx(0.02 + 0.012)
    assert compute_cost(Usage(input_tokens=5, reported_cost_usd=0.5), pricing) == 0.5


def test_usage_tracker_groups() -> None:
    tracker = UsageTracker()
    tracker.record(UsageRecord(agent="a", provider="openai", model="m1", input_tokens=10, output_tokens=5, cost_usd=0.1))
    tracker.record(UsageRecord(agent="b", provider="gemini", model="m2", input_tokens=30, output_tokens=5, cost_usd=0.2))
    tracker.record(UsageRecord(agent="a", provider="openai", model="m1", input_tokens=1, output_tokens=1, cost_usd=0.0))
    assert tracker.totals().total_tokens == 52
    assert tracker.by_agent()["a"].calls == 2
    assert list(tracker.by_provider()) == ["gemini", "openai"]
    tracker.update_context("a", 750, 1000)
    assert tracker.max_context_ratio() == 0.75
    assert tracker.snapshot()["totals"]["cost_usd"] == pytest.approx(0.3)
