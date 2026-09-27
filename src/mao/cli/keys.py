"""Non-blocking single-key input while the live dashboard is shown."""

from __future__ import annotations

import queue
import sys
import threading
import time


class KeyListener:
    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not sys.stdin or not sys.stdin.isatty():
            return
        self._stop.clear()
        target = self._run_windows if sys.platform == "win32" else self._run_posix
        self._thread = threading.Thread(target=target, name="mao-keys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def drain(self) -> list[str]:
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):  # function/arrow keys come as two codes
                    msvcrt.getwch()
                    continue
                self._queue.put(char.lower())
            else:
                time.sleep(0.05)

    def _run_posix(self) -> None:  # pragma: no cover - exercised on POSIX only
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    self._queue.put(sys.stdin.read(1).lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


class InterruptFlag:
    """Set from the SIGINT handler, consumed by the UI loop (thread-safe)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.last = 0.0

    def trigger(self) -> bool:
        """Return True when this is a quick second Ctrl+C (the caller may exit)."""
        now = time.monotonic()
        double = self._event.is_set() and now - self.last < 1.5
        self.last = now
        self._event.set()
        return double

    def consume(self) -> bool:
        if self._event.is_set():
            self._event.clear()
            return True
        return False
