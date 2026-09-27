"""Client-side rate limiting: concurrency, requests/minute and tokens/minute."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from mao.config.schema import RateLimitConfig


class SlidingWindow:
    """Allows at most ``limit`` units within any 60 second window."""

    def __init__(self, limit: int, *, clock: Callable[[], float] = time.monotonic, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    def _purge(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.window_s:
            self._events.popleft()

    def used(self) -> int:
        self._purge(self._clock())
        return sum(amount for _, amount in self._events)

    async def acquire(self, amount: int = 1) -> None:
        amount = max(1, min(amount, self.limit))
        while True:
            async with self._lock:
                now = self._clock()
                self._purge(now)
                if sum(a for _, a in self._events) + amount <= self.limit:
                    self._events.append((now, amount))
                    return
                wait = self.window_s - (now - self._events[0][0]) + 0.05
            # re-check periodically instead of sleeping the whole window at once
            await asyncio.sleep(min(max(0.05, wait), 0.5))


class ProviderLimiter:
    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._rpm = SlidingWindow(config.requests_per_minute) if config.requests_per_minute else None
        self._tpm = SlidingWindow(config.tokens_per_minute) if config.tokens_per_minute else None
        self.in_flight = 0
        self.waiting = 0

    @asynccontextmanager
    async def slot(self, estimated_tokens: int = 0) -> AsyncIterator[None]:
        self.waiting += 1
        try:
            if self._rpm is not None:
                await self._rpm.acquire(1)
            if self._tpm is not None and estimated_tokens > 0:
                await self._tpm.acquire(estimated_tokens)
            await self._semaphore.acquire()
        finally:
            self.waiting -= 1
        self.in_flight += 1
        try:
            yield
        finally:
            self.in_flight -= 1
            self._semaphore.release()
