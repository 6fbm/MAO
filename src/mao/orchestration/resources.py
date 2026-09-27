"""System resource sampling (CPU, RAM, optional NVIDIA GPU) for display and throttling."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ResourceSnapshot:
    cpu_percent: float | None = None
    ram_percent: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    gpu_name: str | None = None
    gpu_used_mb: int | None = None
    gpu_total_mb: int | None = None
    gpu_util_percent: float | None = None
    ts: float = 0.0

    def render(self) -> str:
        parts = []
        if self.cpu_percent is not None:
            parts.append(f"CPU {self.cpu_percent:.0f} %")
        if self.ram_percent is not None:
            detail = f" ({self.ram_used_gb:.1f}/{self.ram_total_gb:.1f} GB)" if self.ram_total_gb else ""
            parts.append(f"RAM {self.ram_percent:.0f} %{detail}")
        if self.gpu_name:
            parts.append(f"GPU {self.gpu_name} {self.gpu_used_mb}/{self.gpu_total_mb} MB ({self.gpu_util_percent:.0f} %)")
        return " · ".join(parts) or "no measurements"


class ResourceMonitor:
    def __init__(self, *, interval_s: float = 2.0, gpu_interval_s: float = 15.0, enable_gpu: bool = True) -> None:
        self.interval_s = interval_s
        self.gpu_interval_s = gpu_interval_s
        self._snapshot = ResourceSnapshot()
        self._gpu: tuple[str, int, int, float] | None = None
        self._gpu_thread: threading.Thread | None = None
        self._last_gpu = -1e9
        self._lock = threading.Lock()
        self._psutil: Any = None
        try:
            import psutil

            self._psutil = psutil
            psutil.cpu_percent(interval=None)  # prime: the first reading is always 0
        except ImportError:
            self._psutil = None
        self._nvidia_smi = shutil.which("nvidia-smi") if enable_gpu else None

    def _refresh_gpu(self) -> None:
        kwargs: dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        try:
            completed = subprocess.run(
                [str(self._nvidia_smi), "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                **kwargs,
            )
            name, used, total, util = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
            value: tuple[str, int, int, float] | None = (name, int(float(used)), int(float(total)), float(util))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            value = None
        with self._lock:
            self._gpu = value

    def sample(self) -> ResourceSnapshot:
        now = time.monotonic()
        with self._lock:
            if self._snapshot.ts and now - self._snapshot.ts < self.interval_s:
                return self._snapshot
        snapshot = ResourceSnapshot(ts=now)
        if self._psutil is not None:
            try:
                snapshot.cpu_percent = float(self._psutil.cpu_percent(interval=None))
                memory = self._psutil.virtual_memory()
                snapshot.ram_percent = float(memory.percent)
                snapshot.ram_used_gb = (memory.total - memory.available) / 1024**3
                snapshot.ram_total_gb = memory.total / 1024**3
            except Exception:  # noqa: BLE001 - monitoring must never break a run
                pass
        if self._nvidia_smi and now - self._last_gpu >= self.gpu_interval_s and (self._gpu_thread is None or not self._gpu_thread.is_alive()):
            self._last_gpu = now
            self._gpu_thread = threading.Thread(target=self._refresh_gpu, name="mao-gpu", daemon=True)
            self._gpu_thread.start()
        with self._lock:
            if self._gpu is not None:
                snapshot.gpu_name, snapshot.gpu_used_mb, snapshot.gpu_total_mb, snapshot.gpu_util_percent = self._gpu
            self._snapshot = snapshot
        return snapshot

    def ram_pressure(self, threshold: float | None) -> bool:
        if threshold is None:
            return False
        snapshot = self.sample()
        return snapshot.ram_percent is not None and snapshot.ram_percent >= threshold
