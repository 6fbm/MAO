"""Session budgets for tokens, cost and number of model calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum

from mao.config.schema import LimitsConfig
from mao.core.errors import BudgetExceededError
from mao.core.events import BudgetEvent, EventBus
from mao.tokens.tracker import UsageTracker


class LimitDecision(str, Enum):
    EXTEND = "extend"
    STOP = "stop"


LimitHandler = Callable[[str, float, float], Awaitable[LimitDecision]]

METRIC_LABELS = {"tokens": "Tokens", "cost": "Cost (USD)", "calls": "Model calls"}


class BudgetGuard:
    def __init__(
        self,
        limits: LimitsConfig,
        tracker: UsageTracker,
        bus: EventBus,
        handler: LimitHandler | None = None,
    ) -> None:
        self.limits = limits
        self.tracker = tracker
        self.bus = bus
        self.handler = handler
        self._extra = {"tokens": 0.0, "cost": 0.0, "calls": 0.0}
        self._warned: set[str] = set()
        self._declined: set[str] = set()
        self._lock = asyncio.Lock()

    def limit(self, metric: str) -> float | None:
        base = {
            "tokens": self.limits.max_tokens,
            "cost": self.limits.max_cost_usd,
            "calls": self.limits.max_llm_calls,
        }[metric]
        return None if base is None else float(base) + self._extra[metric]

    def usage(self) -> dict[str, float]:
        totals = self.tracker.totals()
        return {"tokens": float(totals.total_tokens), "cost": totals.cost_usd, "calls": float(totals.calls)}

    def reset_declines(self) -> None:
        """Called after the user raised a limit manually (e.g. /max-cost)."""
        self._declined.clear()
        self._warned.clear()

    async def check(self, *, est_tokens: int = 0, est_cost: float = 0.0) -> None:
        async with self._lock:
            used = self.usage()
            projected = {"tokens": used["tokens"] + est_tokens, "cost": used["cost"] + est_cost, "calls": used["calls"] + 1}
            for metric in ("tokens", "cost", "calls"):
                limit = self.limit(metric)
                if limit is None:
                    continue
                if projected[metric] > limit:
                    if metric in self._declined:
                        raise BudgetExceededError(METRIC_LABELS[metric], used[metric], limit)
                    self.bus.publish(BudgetEvent(metric=metric, used=used[metric], limit=limit, level="exceeded"))
                    if self.limits.on_limit == "ask" and self.handler is not None:
                        decision = await self.handler(metric, used[metric], limit)
                        if decision is LimitDecision.EXTEND:
                            increase = max(limit * 0.5, projected[metric] - limit)
                            self._extra[metric] += increase
                            self.bus.publish(
                                BudgetEvent(metric=metric, used=used[metric], limit=limit + increase, level="extended")
                            )
                            continue
                    self._declined.add(metric)
                    raise BudgetExceededError(METRIC_LABELS[metric], used[metric], limit)
                if projected[metric] >= limit * self.limits.soft_limit_ratio and metric not in self._warned:
                    self._warned.add(metric)
                    self.bus.publish(BudgetEvent(metric=metric, used=used[metric], limit=limit, level="warning"))
