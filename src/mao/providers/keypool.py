"""Multiple API keys per provider with rotation, cooldowns and invalidation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from mao.core.errors import NoAvailableKeyError
from mao.security.secrets import fingerprint

DEFAULT_RATE_LIMIT_COOLDOWN_S = 20.0
MAX_COOLDOWN_S = 600.0


@dataclass
class KeyState:
    key: str
    fingerprint: str
    source: str
    cooldown_until: float = 0.0
    invalid: bool = False
    invalid_reason: str = ""
    in_flight: int = 0
    uses: int = 0
    failures: int = 0
    last_used: float = 0.0


class KeyPool:
    def __init__(
        self,
        provider: str,
        keys: list[tuple[str, str]],
        *,
        requires_key: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.requires_key = requires_key
        self._clock = clock
        self._lock = threading.Lock()
        seen: set[str] = set()
        self._keys: list[KeyState] = []
        for key, source in keys:
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                self._keys.append(KeyState(key=key, fingerprint=fingerprint(key), source=source))

    @property
    def size(self) -> int:
        return len(self._keys)

    def healthy_count(self) -> int:
        with self._lock:
            return sum(1 for k in self._keys if not k.invalid)

    def is_usable(self) -> bool:
        return not self.requires_key or self.healthy_count() > 0

    def acquire(self) -> KeyState | None:
        if not self.requires_key:
            return None
        with self._lock:
            now = self._clock()
            candidates = [k for k in self._keys if not k.invalid]
            if not candidates:
                reason = "No API keys configured" if not self._keys else "All API keys are invalid or exhausted"
                raise NoAvailableKeyError(reason, provider=self.provider)
            ready = [k for k in candidates if k.cooldown_until <= now]
            if not ready:
                wait = min(k.cooldown_until for k in candidates) - now
                raise NoAvailableKeyError(
                    f"All API keys are in cooldown (free again in {wait:.0f}s)", provider=self.provider, wait_seconds=wait
                )
            chosen = min(ready, key=lambda k: (k.in_flight, k.last_used))
            chosen.in_flight += 1
            chosen.uses += 1
            chosen.last_used = now
            return chosen

    def release(self, state: KeyState | None) -> None:
        if state is None:
            return
        with self._lock:
            state.in_flight = max(0, state.in_flight - 1)

    def report_success(self, state: KeyState | None) -> None:
        if state is not None:
            with self._lock:
                state.failures = 0

    def report_rate_limited(self, state: KeyState | None, retry_after: float | None) -> None:
        if state is None:
            return
        cooldown = min(MAX_COOLDOWN_S, retry_after if retry_after and retry_after > 0 else DEFAULT_RATE_LIMIT_COOLDOWN_S)
        with self._lock:
            state.failures += 1
            state.cooldown_until = max(state.cooldown_until, self._clock() + cooldown)

    def report_invalid(self, state: KeyState | None, reason: str) -> None:
        if state is None:
            return
        with self._lock:
            state.invalid = True
            state.invalid_reason = reason

    def has_other_ready_key(self, state: KeyState | None) -> bool:
        if state is None:
            return False
        with self._lock:
            now = self._clock()
            return any(k is not state and not k.invalid and k.cooldown_until <= now for k in self._keys)

    def status(self) -> list[dict[str, str]]:
        with self._lock:
            now = self._clock()
            rows = []
            for k in self._keys:
                if k.invalid:
                    state = f"invalid ({k.invalid_reason})"
                elif k.cooldown_until > now:
                    state = f"Cooldown {k.cooldown_until - now:.0f}s"
                else:
                    state = "ready"
                rows.append({"fingerprint": k.fingerprint, "source": k.source, "state": state, "uses": str(k.uses)})
            return rows
