"""The single path to every model: budgets, keys, rate limits, retries, fallbacks, tracking."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mao.core.errors import (
    AuthenticationError,
    BudgetExceededError,
    ConfigError,
    ContextLengthError,
    InvalidRequestError,
    NoAvailableKeyError,
    OperationCancelled,
    ProviderError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    TransientProviderError,
)
from mao.core.events import EventBus, LLMCallEvent, ProviderIssueEvent
from mao.core.text import one_line
from mao.core.types import CompletionRequest, CompletionResponse, Usage
from mao.models.catalog import CatalogEntry, ModelCatalog
from mao.orchestration.control import RunControl
from mao.providers.base import Provider
from mao.providers.keypool import KeyPool
from mao.providers.prompt_tools import parse_prompt_response, to_prompt_request
from mao.providers.ratelimit import ProviderLimiter
from mao.tokens.budget import BudgetGuard
from mao.tokens.estimator import ESTIMATOR
from mao.tokens.tracker import UsageRecord, UsageTracker, compute_cost, estimate_cost

Sleep = Callable[[float], Awaitable[None]]

MAX_KEY_WAITS = 3
MAX_ADJUSTMENTS = 4

_TOOLS_UNSUPPORTED = (
    "does not support tools",
    "tools is not supported",
    "tool use is not supported",
    "function calling is not enabled",
    "tool calling is not supported",
)
_HISTORY_PROBLEMS = ("thinking", "signature", "reasoning", "thought", "item with id", "tool_use", "tool_result", "function call")


def backoff_delay(attempt: int) -> float:
    base = min(30.0, 1.5 * 2 ** max(0, attempt - 1))
    return base * (0.8 + random.random() * 0.4)


@dataclass
class GatewayResult:
    response: CompletionResponse
    entry: CatalogEntry
    cost_usd: float


class LLMGateway:
    def __init__(
        self,
        *,
        providers: dict[str, Provider],
        catalog: ModelCatalog,
        keypools: dict[str, KeyPool],
        limiters: dict[str, ProviderLimiter],
        tracker: UsageTracker,
        bus: EventBus,
        budget: BudgetGuard | None = None,
        control: RunControl | None = None,
        sleep: Sleep = asyncio.sleep,
        max_wait_s: float = 90.0,
    ) -> None:
        self.providers = providers
        self.catalog = catalog
        self.keypools = keypools
        self.limiters = limiters
        self.tracker = tracker
        self.bus = bus
        self.budget = budget
        self.control = control
        self._sleep = sleep
        self.max_wait_s = max_wait_s
        self._prompt_tool_models: set[str] = set()

    # ------------------------------------------------------------------ public

    async def complete(
        self,
        request: CompletionRequest,
        *,
        model_ref: str,
        agent: str,
        purpose: str = "",
        fallbacks: list[str] | None = None,
    ) -> GatewayResult:
        chain = [model_ref, *(fallbacks or [])]
        attempts: list[str] = []
        for index, ref in enumerate(chain):
            next_ref = chain[index + 1] if index + 1 < len(chain) else None
            try:
                entry = self.catalog.resolve(ref)
            except ConfigError as exc:
                attempts.append(f"{ref}: {exc}")
                continue
            if not self.catalog.is_available(entry):
                attempts.append(f"{entry.ref}: {self.catalog.unavailable_reason(entry)}")
                continue
            try:
                return await self._call_with_retries(entry, request, agent=agent, purpose=purpose)
            except (ContextLengthError, BudgetExceededError, OperationCancelled):
                raise
            except ProviderError as exc:
                attempts.append(f"{entry.ref}: {exc}")
                self.bus.publish(
                    ProviderIssueEvent(
                        agent=agent,
                        provider=entry.provider,
                        model=entry.api_id,
                        error_type=type(exc).__name__,
                        message=one_line(str(exc), 200),
                        action="fallback" if next_ref else "give_up",
                        next_model=next_ref,
                    )
                )
        raise ProviderUnavailableError(
            f"No model could handle the request for '{agent}'", attempts=attempts
        )

    # ------------------------------------------------------------------ internals

    def _issue(self, entry: CatalogEntry, agent: str, error: Exception, action: str, attempt: int, max_attempts: int, wait: float = 0.0) -> None:
        self.bus.publish(
            ProviderIssueEvent(
                agent=agent,
                provider=entry.provider,
                model=entry.api_id,
                error_type=type(error).__name__,
                message=one_line(str(error), 200),
                attempt=attempt,
                max_attempts=max_attempts,
                action=action,
                wait_s=round(wait, 2),
            )
        )

    @staticmethod
    def adjust_request(request: CompletionRequest, error: ProviderError) -> CompletionRequest | None:
        """Derive a request that avoids a parameter the provider rejected."""
        message = (error.message or "").lower()
        meta = request.metadata

        def flag(**flags: bool) -> CompletionRequest:
            return request.model_copy(update={"metadata": {**meta, **flags}})

        if "temperature" in message and request.temperature is not None and not meta.get("omit_temperature"):
            return flag(omit_temperature=True)
        if ("include" in message or "encrypted_content" in message) and not meta.get("omit_include"):
            return flag(omit_include=True)
        if request.json_mode and not meta.get("omit_json_mode") and (
            "response_format" in message or "responsemimetype" in message or "json_object" in message or "text.format" in message
        ):
            return flag(omit_json_mode=True)
        if request.tools and not meta.get("prompt_tools") and any(m in message for m in _TOOLS_UNSUPPORTED):
            return flag(prompt_tools=True)
        if not meta.get("plain_history") and any(m in message for m in _HISTORY_PROBLEMS):
            return flag(plain_history=True)
        return None

    async def _call_with_retries(
        self, entry: CatalogEntry, template: CompletionRequest, *, agent: str, purpose: str
    ) -> GatewayResult:
        provider = self.providers[entry.provider]
        pool = self.keypools[entry.provider]
        limiter = self.limiters[entry.provider]
        provider_config = entry.provider_config
        max_attempts = provider_config.max_retries + 1

        request = template.model_copy(update={"model": entry.api_id, "metadata": dict(template.metadata)})
        cap = entry.config.max_output_tokens
        request.max_output_tokens = min(request.max_output_tokens, cap) if request.max_output_tokens else cap
        if request.tools and (entry.config.tool_calling == "prompt" or entry.ref in self._prompt_tool_models):
            request.metadata["prompt_tools"] = True
        if request.tools and entry.config.tool_calling == "none":
            raise InvalidRequestError("The model does not support tools", provider=entry.provider)

        attempt = 0
        key_waits = 0
        adjustments = 0
        rotations = 0
        while True:
            attempt += 1
            if self.control is not None:
                await self.control.checkpoint()
            prompt_tools = bool(request.tools and request.metadata.get("prompt_tools"))
            effective = to_prompt_request(request) if prompt_tools else request
            estimated_input = ESTIMATOR.count_request(effective.messages, effective.system, effective.tools)
            if self.budget is not None:
                await self.budget.check(
                    est_tokens=estimated_input,
                    est_cost=estimate_cost(estimated_input, 0, entry.config.pricing),
                )
            try:
                key = pool.acquire()
            except NoAvailableKeyError as exc:
                if exc.wait_seconds is not None and exc.wait_seconds <= self.max_wait_s and key_waits < MAX_KEY_WAITS:
                    key_waits += 1
                    attempt -= 1
                    self._issue(entry, agent, exc, "wait", attempt + 1, max_attempts, exc.wait_seconds)
                    await self._sleep(exc.wait_seconds + 0.1)
                    continue
                raise

            self.bus.publish(
                LLMCallEvent(
                    stage="start",
                    agent=agent,
                    provider=entry.provider,
                    model=entry.api_id,
                    purpose=purpose,
                    attempt=attempt,
                    input_tokens=estimated_input,
                    context_window=entry.config.context_window,
                    estimated=True,
                )
            )
            error: ProviderError | None = None
            response: CompletionResponse | None = None
            try:
                async with limiter.slot(estimated_input):
                    response = await asyncio.wait_for(
                        provider.complete(
                            effective,
                            api_key=key.key if key is not None else None,
                            model=entry.config,
                            timeout=provider_config.timeout_s,
                        ),
                        timeout=provider_config.timeout_s + 15,
                    )
            except asyncio.TimeoutError:
                error = TransientProviderError("Timeout", provider=entry.provider)
            except ProviderError as exc:
                error = exc
            finally:
                pool.release(key)

            if error is None and response is not None:
                if prompt_tools:
                    response = parse_prompt_response(response, {t.name for t in request.tools})
                pool.report_success(key)
                return self._record_success(entry, response, effective, agent, purpose, attempt, estimated_input)

            assert error is not None
            if isinstance(error, RateLimitError):
                pool.report_rate_limited(key, error.retry_after)
                if pool.has_other_ready_key(key) and rotations < pool.size:
                    rotations += 1
                    attempt -= 1
                    self._issue(entry, agent, error, "rotate_key", attempt + 1, max_attempts)
                    continue
                if attempt < max_attempts:
                    wait = min(error.retry_after or backoff_delay(attempt), self.max_wait_s)
                    self._issue(entry, agent, error, "retry", attempt + 1, max_attempts, wait)
                    await self._sleep(wait)
                    continue
                raise error
            if isinstance(error, (AuthenticationError, QuotaExceededError)) and key is not None:
                pool.report_invalid(key, type(error).__name__)
                if pool.healthy_count() > 0:
                    attempt -= 1
                    self._issue(entry, agent, error, "rotate_key", attempt + 1, max_attempts)
                    continue
                self.catalog.set_key_availability(entry.provider, False)
                raise error
            if isinstance(error, InvalidRequestError) and adjustments < MAX_ADJUSTMENTS:
                adjusted = self.adjust_request(request, error)
                if adjusted is not None:
                    adjustments += 1
                    attempt -= 1
                    if adjusted.metadata.get("prompt_tools"):
                        self._prompt_tool_models.add(entry.ref)
                    request = adjusted
                    self._issue(entry, agent, error, "adjust_request", attempt + 1, max_attempts)
                    continue
            if error.retryable and attempt < max_attempts:
                wait = min(error.retry_after or backoff_delay(attempt), self.max_wait_s)
                self._issue(entry, agent, error, "retry", attempt + 1, max_attempts, wait)
                await self._sleep(wait)
                continue
            raise error

    def _record_success(
        self,
        entry: CatalogEntry,
        response: CompletionResponse,
        effective: CompletionRequest,
        agent: str,
        purpose: str,
        attempt: int,
        estimated_input: int,
    ) -> GatewayResult:
        if response.usage.input_tokens == 0 and response.usage.output_tokens == 0:
            response.usage = Usage(
                input_tokens=estimated_input,
                output_tokens=ESTIMATOR.count_message(response.message),
                estimated=True,
            )
        cost = compute_cost(response.usage, entry.config.pricing)
        self.tracker.record(
            UsageRecord(
                agent=agent,
                provider=entry.provider,
                model=entry.api_id,
                purpose=purpose,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_input_tokens=response.usage.cached_input_tokens,
                reasoning_tokens=response.usage.reasoning_tokens,
                cost_usd=cost,
                cost_reported=response.usage.reported_cost_usd is not None,
                estimated_usage=response.usage.estimated,
                latency_s=response.latency_s,
            )
        )
        self.tracker.update_context(agent, response.usage.input_tokens, entry.config.context_window)
        self.bus.publish(
            LLMCallEvent(
                stage="end",
                agent=agent,
                provider=entry.provider,
                model=entry.api_id,
                purpose=purpose,
                attempt=attempt,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=cost,
                latency_s=response.latency_s,
                finish_reason=response.finish_reason.value,
                context_window=entry.config.context_window,
                estimated=response.usage.estimated,
            )
        )
        return GatewayResult(response=response, entry=entry, cost_usd=cost)
