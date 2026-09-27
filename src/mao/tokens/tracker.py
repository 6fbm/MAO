"""Records every model call and aggregates tokens and costs."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from pydantic import BaseModel, Field

from mao.config.schema import PricingConfig
from mao.core.types import Usage


def compute_cost(usage: Usage, pricing: PricingConfig) -> float:
    """Cost in USD. Provider-reported cost wins over the price list."""
    if usage.reported_cost_usd is not None:
        return usage.reported_cost_usd
    return estimate_cost(
        usage.input_tokens,
        usage.output_tokens,
        pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: PricingConfig,
    *,
    cached_input_tokens: int = 0,
) -> float:
    long_context = (
        pricing.long_context_threshold is not None and input_tokens > pricing.long_context_threshold
    )
    input_rate = pricing.input_per_mtok
    output_rate = pricing.output_per_mtok
    if long_context:
        if pricing.long_input_per_mtok is not None:
            input_rate = pricing.long_input_per_mtok
        if pricing.long_output_per_mtok is not None:
            output_rate = pricing.long_output_per_mtok
    cached = max(0, min(cached_input_tokens, input_tokens))
    cached_rate = pricing.cached_input_per_mtok if pricing.cached_input_per_mtok is not None else pricing.input_per_mtok
    if long_context and pricing.input_per_mtok > 0:
        cached_rate *= input_rate / pricing.input_per_mtok
    return ((input_tokens - cached) * input_rate + cached * cached_rate + output_tokens * output_rate) / 1_000_000


class UsageRecord(BaseModel):
    ts: float = Field(default_factory=time.time)
    agent: str
    provider: str
    model: str
    purpose: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_reported: bool = False
    estimated_usage: bool = False
    latency_s: float = 0.0


class UsageTotals(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, record: UsageRecord) -> None:
        self.calls += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cached_input_tokens += record.cached_input_tokens
        self.reasoning_tokens += record.reasoning_tokens
        self.cost_usd += record.cost_usd


class ContextUsage(BaseModel):
    used_tokens: int = 0
    window: int = 0

    @property
    def ratio(self) -> float:
        return self.used_tokens / self.window if self.window else 0.0


class UsageTracker:
    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._context: dict[str, ContextUsage] = {}

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)

    def extend(self, records: list[UsageRecord]) -> None:
        with self._lock:
            self._records.extend(records)

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def update_context(self, agent: str, used_tokens: int, window: int) -> None:
        with self._lock:
            self._context[agent] = ContextUsage(used_tokens=used_tokens, window=window)

    def context_usage(self) -> dict[str, ContextUsage]:
        with self._lock:
            return dict(self._context)

    def max_context_ratio(self) -> float:
        usage = self.context_usage()
        return max((c.ratio for c in usage.values()), default=0.0)

    def totals(self) -> UsageTotals:
        totals = UsageTotals()
        for record in self.records:
            totals.add(record)
        return totals

    def _group(self, attribute: str) -> dict[str, UsageTotals]:
        groups: dict[str, UsageTotals] = defaultdict(UsageTotals)
        for record in self.records:
            groups[getattr(record, attribute)].add(record)
        return dict(sorted(groups.items(), key=lambda item: -item[1].total_tokens))

    def by_agent(self) -> dict[str, UsageTotals]:
        return self._group("agent")

    def by_provider(self) -> dict[str, UsageTotals]:
        return self._group("provider")

    def by_model(self) -> dict[str, UsageTotals]:
        grouped: dict[str, UsageTotals] = defaultdict(UsageTotals)
        for record in self.records:
            grouped[f"{record.provider}/{record.model}"].add(record)
        return dict(sorted(grouped.items(), key=lambda item: -item[1].total_tokens))

    def any_estimated(self) -> bool:
        return any(r.estimated_usage for r in self.records)

    def snapshot(self) -> dict:
        def dump(groups: dict[str, UsageTotals]) -> dict:
            return {k: {**v.model_dump(), "total_tokens": v.total_tokens} for k, v in groups.items()}

        totals = self.totals()
        return {
            "totals": {**totals.model_dump(), "total_tokens": totals.total_tokens},
            "by_agent": dump(self.by_agent()),
            "by_provider": dump(self.by_provider()),
            "by_model": dump(self.by_model()),
            "context": {k: {**v.model_dump(), "ratio": round(v.ratio, 4)} for k, v in self.context_usage().items()},
            "cost_note": "Cost estimated from the price list (providers.yaml) unless cost_reported=true",
            "records": [r.model_dump() for r in self.records],
        }
