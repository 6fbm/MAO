"""Pause / resume / stop for a running session."""

from __future__ import annotations

import asyncio

from mao.core.errors import OperationCancelled


class RunControl:
    def __init__(self) -> None:
        self._running = asyncio.Event()
        self._running.set()
        self._stopped = False
        self.stop_reason = ""

    @property
    def paused(self) -> bool:
        return not self._running.is_set() and not self._stopped

    @property
    def stopped(self) -> bool:
        return self._stopped

    def pause(self) -> None:
        if not self._stopped:
            self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self, reason: str = "stopped by the user") -> None:
        self._stopped = True
        self.stop_reason = reason
        self._running.set()  # wake up paused waiters so they can observe the stop

    async def checkpoint(self) -> None:
        if self._stopped:
            raise OperationCancelled(self.stop_reason)
        if not self._running.is_set():
            await self._running.wait()
            if self._stopped:
                raise OperationCancelled(self.stop_reason)
