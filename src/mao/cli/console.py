"""Console UI: output helpers, prompts, approval/budget dialogs and the live dashboard loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from mao.cli.dashboard import DashboardState, render_dashboard
from mao.cli.keys import InterruptFlag, KeyListener
from mao.core.errors import MaoError
from mao.core.text import fmt_cost, fmt_int
from mao.security.approval import ApprovalDecision, ApprovalRequest
from mao.security.risk import RiskLevel
from mao.tokens.budget import METRIC_LABELS, LimitDecision

if TYPE_CHECKING:
    from mao.app import AppContext

log = logging.getLogger("mao.cli")


class ConsoleUI:
    def __init__(self, app: AppContext, *, interactive: bool, debug: bool = False) -> None:
        self.app = app
        self.interactive = interactive
        # markup/emoji off: agent output, file content and git output must never be interpreted as rich markup
        self.console = Console(highlight=False, markup=False, emoji=False)
        self.state = DashboardState(app)
        self.state.debug = debug
        app.bus.subscribe(self.state.on_event)
        if not interactive:
            self.state.on_feed = self._print_feed_line
        self.interrupts = InterruptFlag()
        self.prompt_lock = asyncio.Lock()
        self.view = "events"
        self._live: Live | None = None
        self._keys: KeyListener | None = None
        self._suspended = False
        self._prompt = None
        if interactive:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory

            from mao.cli.repl import CommandCompleter

            history = app.paths.history_file
            history.parent.mkdir(parents=True, exist_ok=True)
            self._prompt = PromptSession(history=FileHistory(str(history)), completer=CommandCompleter(), complete_while_typing=False)

    # ------------------------------------------------------------------ output

    @property
    def debug(self) -> bool:
        return self.state.debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self.state.debug = value

    def print(self, renderable: RenderableType | str = "") -> None:
        self.console.print(renderable)

    def info(self, text: str) -> None:
        self.console.print(Text(text, style="cyan"))

    def success(self, text: str) -> None:
        self.console.print(Text(f"✓ {text}", style="green"))

    def warn(self, text: str) -> None:
        self.console.print(Text(f"! {text}", style="yellow"))

    def error(self, text: str) -> None:
        self.console.print(Text(f"✗ {text}", style="bold red"))

    def _print_feed_line(self, text: str, style: str) -> None:
        self.console.print(Text(text, style=style))

    # ------------------------------------------------------------------ input

    async def read_line(self, prompt: str, *, password: bool = False) -> str | None:
        if self._prompt is not None:
            try:
                return await self._prompt.prompt_async(prompt, is_password=password)
            except EOFError:
                return None
            except KeyboardInterrupt:
                return ""
        self.console.print(Text(prompt, style="bold"), end="")
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return None
        return line.rstrip("\r\n")

    async def confirm(self, question: str, *, default: bool = False) -> bool:
        suffix = " [Y/n] " if default else " [y/N] "
        answer = await self.read_line(question + suffix)
        if answer is None or not answer.strip():
            return default
        return answer.strip().lower() in ("y", "yes", "j", "ja")

    async def choose(self, question: str, options: dict[str, str], default: str) -> str:
        labels = " / ".join(f"[{key.upper() if key == default else key}] {text}" for key, text in options.items())
        while True:
            answer = await self.read_line(f"{question} {labels}: ")
            if answer is None:
                return default
            choice = answer.strip().lower()[:1] or default
            if choice in options:
                return choice
            self.warn("Invalid choice.")

    # ------------------------------------------------------------------ dialogs used by the core

    @contextlib.asynccontextmanager
    async def suspended(self) -> AsyncIterator[None]:
        live, keys = self._live, self._keys
        if live is None:
            yield
            return
        self._suspended = True
        if keys is not None:
            keys.stop()
        live.stop()
        try:
            yield
        finally:
            live.start()
            if keys is not None:
                keys.start()
            self._suspended = False

    async def approval_handler(self, request: ApprovalRequest) -> ApprovalDecision:
        async with self.prompt_lock:
            async with self.suspended():
                action = request.action
                body = Text()
                body.append("Agent ", style="bold")
                body.append(action.agent, style="bold cyan")
                body.append(" wants to:\n", style="bold")
                body.append(f"{action.kind.label} ", style="bold red")
                body.append(action.target + "\n")
                if action.detail:
                    body.append(f"{action.detail}\n", style="dim")
                body.append(f"Risk: {action.risk.label}", style="red" if action.risk >= RiskLevel.HIGH else "yellow")
                if action.reasons:
                    body.append("  (" + "; ".join(action.reasons) + ")", style="dim")
                self.console.print(Panel(body, title="WARNING", title_align="left", border_style="red" if action.risk >= RiskLevel.HIGH else "yellow"))
                critical = action.risk >= RiskLevel.CRITICAL
                options = "[y]es / [N]o" + ("" if critical else " / [a]lways (this session, same kind of action)")
                answer = await self.read_line(f"Allow? {options}: ")
                choice = (answer or "").strip().lower()
                if choice in ("y", "yes", "j", "ja"):
                    return ApprovalDecision.ALLOW
                if choice in ("a", "always") and not critical:
                    return ApprovalDecision.ALLOW_SESSION
                return ApprovalDecision.DENY

    async def limit_handler(self, metric: str, used: float, limit: float) -> LimitDecision:
        async with self.prompt_lock:
            async with self.suspended():
                label = METRIC_LABELS.get(metric, metric)
                value = f"{fmt_cost(used)} / {fmt_cost(limit)}" if metric == "cost" else f"{fmt_int(used)} / {fmt_int(limit)}"
                self.console.print(Panel(Text(f"Limit reached: {label} {value}"), title="BUDGET", border_style="red"))
                choice = await self.choose("What now?", {"c": "raise it by 50 % and continue", "s": "stop in a controlled way"}, default="s")
                return LimitDecision.EXTEND if choice == "c" else LimitDecision.STOP

    # ------------------------------------------------------------------ dashboard

    def _render(self, paused: bool = False) -> RenderableType:
        return render_dashboard(self.state, self.app, view=self.view, paused=paused, height=self.console.size.height)

    def status_snapshot(self) -> RenderableType:
        srt = self.app.orchestrator.active
        return self._render(paused=bool(srt and srt.control.paused))

    async def _handle_key(self, key: str) -> None:
        orchestrator = self.app.orchestrator
        if key == "p":
            with contextlib.suppress(MaoError):
                orchestrator.pause()
        elif key == "s":
            async with self.prompt_lock:
                async with self.suspended():
                    if await self.confirm("Really stop the task in a controlled way?", default=False):
                        with contextlib.suppress(MaoError):
                            orchestrator.stop()
        elif key == "d":
            self.debug = not self.debug
        elif key == "m":
            self.view = "messages" if self.view == "events" else "events"

    async def attach(self, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        """Show progress until the task finishes or the run is paused."""
        orchestrator = self.app.orchestrator
        if not self.interactive:
            while not task.done():
                if self.interrupts.consume():
                    with contextlib.suppress(MaoError):
                        orchestrator.pause()
                    return
                srt = orchestrator.active
                if srt is not None and srt.control.paused:
                    return
                await asyncio.wait({task}, timeout=0.25)
            return
        keys = KeyListener()
        live = Live(self._render(), console=self.console, auto_refresh=False, transient=False, vertical_overflow="crop")
        self._live, self._keys = live, keys
        live.start()
        keys.start()
        try:
            while not task.done():
                if not self._suspended:
                    for key in keys.drain():
                        await self._handle_key(key)
                    if self.interrupts.consume():
                        with contextlib.suppress(MaoError):
                            orchestrator.pause()
                    srt = orchestrator.active
                    if srt is not None and srt.control.paused:
                        live.update(self._render(paused=True), refresh=True)
                        break
                    live.update(self._render(), refresh=True)
                await asyncio.wait({task}, timeout=0.25)
            if task.done() and not self._suspended:
                live.update(self._render(), refresh=True)
        finally:
            keys.stop()
            live.stop()
            self._live = None
            self._keys = None
