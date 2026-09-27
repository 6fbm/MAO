"""Gateway resilience: retries, key rotation, request adjustment, fallback, budgets."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from mao.config.schema import LimitsConfig, ModelConfig, PricingConfig, ProviderConfig, Settings
from mao.core.errors import (
    AuthenticationError,
    BudgetExceededError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
    TransientProviderError,
)
from mao.core.events import EventBus, ProviderIssueEvent
from mao.core.types import ChatMessage, CompletionRequest, CompletionResponse, FinishReason, ToolSpec
from mao.models.catalog import ModelCatalog
from mao.providers.base import create_provider
from mao.providers.gateway import LLMGateway
from mao.providers.keypool import KeyPool
from mao.providers.mock import MockProvider
from mao.providers.ratelimit import ProviderLimiter
from mao.tokens.budget import BudgetGuard, LimitDecision
from mao.tokens.tracker import UsageTracker


class Harness:
    def __init__(self, provider_specs: dict[str, dict], limits: LimitsConfig | None = None) -> None:
        self.bus = EventBus()
        self.events: list = []
        self.bus.subscribe(self.events.append)
        configs = {}
        for name, spec in provider_specs.items():
            configs[name] = ProviderConfig(
                type="mock",
                requires_api_key=bool(spec.get("keys")),
                max_retries=spec.get("max_retries", 2),
                models={"m": spec.get("model", ModelConfig(pricing=PricingConfig(input_per_mtok=1.0, output_per_mtok=2.0)))},
            )
        client = httpx.AsyncClient()
        self.providers = {name: create_provider(name, cfg, client) for name, cfg in configs.items()}
        self.keypools = {
            name: KeyPool(name, [(k, "test") for k in spec.get("keys", [])], requires_key=bool(spec.get("keys")))
            for name, spec in provider_specs.items()
        }
        self.catalog = ModelCatalog(configs, Settings())
        for name, pool in self.keypools.items():
            self.catalog.set_key_availability(name, pool.is_usable())
        self.tracker = UsageTracker()
        self.sleeps: list[float] = []
        budget = BudgetGuard(limits, self.tracker, self.bus) if limits else None

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.gateway = LLMGateway(
            providers=self.providers,
            catalog=self.catalog,
            keypools=self.keypools,
            limiters={n: ProviderLimiter(cfg.rate_limit) for n, cfg in configs.items()},
            tracker=self.tracker,
            bus=self.bus,
            budget=budget,
            sleep=fake_sleep,
        )

    def mock(self, name: str) -> MockProvider:
        provider = self.providers[name]
        assert isinstance(provider, MockProvider)
        return provider

    def script(self, name: str, steps: list[Callable[[CompletionRequest], object] | Exception | str]) -> list[str | None]:
        keys_used: list[str | None] = []
        provider = self.mock(name)
        original = provider.complete

        async def complete(request, *, api_key, model, timeout):  # type: ignore[no-untyped-def]
            keys_used.append(api_key)
            return await original(request, api_key=api_key, model=model, timeout=timeout)

        provider.complete = complete  # type: ignore[method-assign]
        queue = list(steps)

        def responder(request: CompletionRequest):  # type: ignore[no-untyped-def]
            step = queue.pop(0) if queue else "done"
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(request)
            return step

        provider.set_responder(responder)
        return keys_used

    def issues(self) -> list[ProviderIssueEvent]:
        return [e for e in self.events if isinstance(e, ProviderIssueEvent)]


def _request(**kwargs) -> CompletionRequest:  # type: ignore[no-untyped-def]
    return CompletionRequest(model="", messages=[ChatMessage.user("Hello")], **kwargs)


async def test_retry_on_transient_error_then_success() -> None:
    h = Harness({"a": {}})
    h.script("a", [TransientProviderError("503 overloaded", provider="a"), "ok"])
    result = await h.gateway.complete(_request(), model_ref="a/m", agent="coder")
    assert result.response.message.content == "ok"
    assert [i.action for i in h.issues()] == ["retry"]
    assert h.issues()[0].attempt == 2 and h.issues()[0].max_attempts == 3
    assert len(h.sleeps) == 1
    assert h.tracker.totals().calls == 1 and h.tracker.totals().cost_usd > 0


async def test_rate_limit_rotates_to_other_key_without_sleeping() -> None:
    h = Harness({"a": {"keys": ["key-one-11111", "key-two-22222"]}})
    used = h.script("a", [RateLimitError("slow down", provider="a", retry_after=30), "ok"])
    await h.gateway.complete(_request(), model_ref="a/m", agent="x")
    assert used[0] != used[1]
    assert [i.action for i in h.issues()] == ["rotate_key"]
    assert h.sleeps == []


async def test_auth_error_invalidates_key_and_falls_back_when_none_left() -> None:
    h = Harness({"a": {"keys": ["key-one-11111"]}, "b": {}})
    h.script("a", [AuthenticationError("bad key", provider="a")])
    h.script("b", ["from b"])
    result = await h.gateway.complete(_request(), model_ref="a/m", agent="x", fallbacks=["b/m"])
    assert result.entry.provider == "b" and result.response.message.content == "from b"
    assert not h.catalog.is_available(h.catalog.resolve("a/m"))
    assert h.issues()[-1].action == "fallback"


async def test_all_retries_exhausted_then_fallback() -> None:
    h = Harness({"a": {"max_retries": 2}, "b": {}})
    h.script("a", [TransientProviderError("down", provider="a")] * 3)
    h.script("b", ["ok"])
    result = await h.gateway.complete(_request(), model_ref="a/m", agent="x", fallbacks=["b/m"])
    assert result.entry.provider == "b"
    actions = [i.action for i in h.issues()]
    assert actions == ["retry", "retry", "fallback"]


async def test_provider_unavailable_when_chain_fails() -> None:
    h = Harness({"a": {"max_retries": 0}})
    h.script("a", [TransientProviderError("down", provider="a")])
    with pytest.raises(ProviderUnavailableError) as info:
        await h.gateway.complete(_request(), model_ref="a/m", agent="x", fallbacks=["unknown/zzz"])
    assert len(info.value.attempts) == 2


async def test_invalid_temperature_is_adjusted() -> None:
    h = Harness({"a": {}})

    def check(request: CompletionRequest) -> str:
        if request.metadata.get("omit_temperature"):
            return "without temperature"
        raise InvalidRequestError("Unsupported parameter: 'temperature' is not supported with this model.", provider="a")

    h.script("a", [check, check])
    result = await h.gateway.complete(_request(temperature=0.3), model_ref="a/m", agent="x")
    assert result.response.message.content == "without temperature"
    assert [i.action for i in h.issues()] == ["adjust_request"]


async def test_prompt_tool_calling_for_models_without_native_tools() -> None:
    h = Harness({"a": {"model": ModelConfig(tool_calling="prompt")}})
    seen: list[CompletionRequest] = []

    def respond(request: CompletionRequest) -> str:
        seen.append(request)
        return '```tool\n{"name": "list_directory", "arguments": {"path": "."}}\n```'

    h.script("a", [respond])
    tool = ToolSpec(name="list_directory", description="List", parameters={"type": "object", "properties": {"path": {"type": "string"}}})
    result = await h.gateway.complete(_request(tools=[tool]), model_ref="a/m", agent="x")
    assert seen[0].tools == [] and "## Tools" in (seen[0].system or "")
    assert result.response.finish_reason is FinishReason.TOOL_CALLS
    assert result.response.message.tool_calls[0].name == "list_directory"


async def test_budget_blocks_calls_over_limit() -> None:
    h = Harness({"a": {}}, limits=LimitsConfig(max_tokens=1_000, max_cost_usd=None, on_limit="stop"))
    h.script("a", ["x" * 5_000])
    await h.gateway.complete(_request(), model_ref="a/m", agent="x")
    with pytest.raises(BudgetExceededError):
        await h.gateway.complete(_request(), model_ref="a/m", agent="x")


async def test_budget_extension_by_handler() -> None:
    bus = EventBus()
    tracker = UsageTracker()
    decisions: list[str] = []

    async def handler(metric: str, used: float, limit: float) -> LimitDecision:
        decisions.append(metric)
        return LimitDecision.EXTEND

    guard = BudgetGuard(LimitsConfig(max_tokens=1_000, max_cost_usd=None), tracker, bus, handler)
    await guard.check(est_tokens=1_500)
    assert decisions == ["tokens"] and guard.limit("tokens") == pytest.approx(1_500)


async def test_usage_is_estimated_when_provider_reports_nothing() -> None:
    h = Harness({"a": {}})

    def respond(_request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(message=ChatMessage.assistant("ok"))

    h.script("a", [respond])
    result = await h.gateway.complete(_request(), model_ref="a/m", agent="x")
    assert result.response.usage.input_tokens > 0
