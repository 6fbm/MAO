"""Interactive shell: dispatch, foreground task handling and the plan decision loop."""

from __future__ import annotations

import asyncio
import difflib
import logging
from collections.abc import Awaitable, Iterable
from typing import TYPE_CHECKING, Any

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from rich.text import Text

from mao.cli.commands import REGISTRY, CommandContext, split_args
from mao.cli.commands.base import preview_manager
from mao.cli.render import agent_overview, banner, plan_view, provider_overview, result_view
from mao.core.errors import MaoError
from mao.core.text import one_line
from mao.orchestration.plan import PlanDocument

if TYPE_CHECKING:
    from mao.app import AppContext
    from mao.cli.console import ConsoleUI

log = logging.getLogger("mao.cli")

# Command names that also work as the first word of a task ("run the tests",
# "stop guessing"). They are only treated as commands on their own or with one of
# their own subcommands; with any other text they stay a task. Prefix them with a
# slash to force the command.
AMBIGUOUS_COMMANDS = frozenset({"run", "go", "plan", "replan", "stop", "pause", "resume", "debate", "help", "changes"})


class CommandCompleter(Completer):
    """Completes commands typed with or without the leading slash."""

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor
        slashed = text.startswith("/")
        body = text[1:] if slashed else text
        if " " not in body:
            prefix = "/" if slashed else ""
            for name in REGISTRY.names():
                if name.startswith(body.lower()):
                    spec = REGISTRY.get(name)
                    yield Completion(prefix + name, start_position=-len(text), display_meta=spec.summary if spec else "")
            return
        head, _, rest = body.partition(" ")
        spec = REGISTRY.get(head)
        if spec and spec.subcommands and " " not in rest:
            for sub in spec.subcommands:
                if sub.startswith(rest.lower()):
                    yield Completion(sub, start_position=-len(rest))


class Repl:
    def __init__(self, app: AppContext, ui: ConsoleUI, *, auto_approve_plan: bool = False) -> None:
        self.app = app
        self.ui = ui
        self.auto_approve_plan = auto_approve_plan
        self.background: asyncio.Task | None = None  # type: ignore[type-arg]
        self.background_kind: str | None = None
        self.should_exit = False
        self.context = CommandContext(self)

    # ------------------------------------------------------------------ startup

    def show_welcome(self) -> None:
        app = self.app
        self.ui.print(banner(app.demo))
        self.ui.print(Text(f"Workspace: {app.config.settings.workspace or '(not set - /workspace <path>)'}", style="bold"))
        self.ui.print(provider_overview(app))
        manager = preview_manager(app)
        self.ui.print(agent_overview(manager, app))
        for name, message in app.discovery_messages.items():
            if "not reachable" in message:
                self.ui.print(Text(f"  {name}: {message}", style="dim"))
        for entry in app.hub.catalog.entries():
            if entry.note.startswith("broken"):
                self.ui.print(Text(f"  {entry.ref}: {one_line(entry.note, 140)}", style="yellow"))
        if app.hub.secret_store_error:
            self.ui.warn(app.hub.secret_store_error)
        if not manager.available():
            self.ui.warn("No agent has a usable model. Add an API key (/providers key add openai) or start a local model (/models discover ollama).")
        self.ui.print(Text("\nhelp lists every command · the leading / is optional · anything that is not a command is planned as a task", style="dim"))

    # ------------------------------------------------------------------ loop

    async def run(self) -> None:
        self.show_welcome()
        while not self.should_exit:
            prompt = self._prompt_text()
            line = await self.ui.read_line(prompt)
            if line is None:
                break
            await self.dispatch(line)

    def _prompt_text(self) -> str:
        srt = self.app.orchestrator.active
        suffix = ""
        if self.background is not None and not self.background.done():
            suffix = " ⏸"
        elif srt is not None:
            suffix = f" #{srt.session.id}"
        return f"mao{suffix} › "

    @staticmethod
    def as_command(text: str) -> str | None:
        """Return the "/command …" form when a slashless line clearly names a command.

        Typing "clear" should clear the screen, not plan a task called "clear". But
        "run the tests" is a task, even though "run" is also a command - so a command
        name followed by free text only counts when the name cannot start a sentence.
        """
        head, _, rest = text.partition(" ")
        spec = REGISTRY.get(head)
        if spec is None:
            return None
        rest = rest.strip()
        if not rest:
            return "/" + text
        first = rest.split(" ", 1)[0].lower()
        if first in spec.subcommands:
            return "/" + text
        if spec.name == "help" and REGISTRY.get(first) is not None:
            return "/" + text
        if head.lower() in AMBIGUOUS_COMMANDS:
            return None
        return "/" + text

    async def dispatch(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        if not text.startswith("/"):
            text = self.as_command(text) or "/plan " + text
        name, _, raw = text[1:].partition(" ")
        spec = REGISTRY.get(name)
        if spec is None:
            suggestions = difflib.get_close_matches(name.lower(), REGISTRY.names(), n=3)
            hint = f" Did you mean: {', '.join('/' + s for s in suggestions)}?" if suggestions else " /help lists all commands."
            self.ui.error(f"Unknown command: /{name}.{hint}")
            return
        try:
            await spec.handler(self.context, split_args(raw), raw)
        except MaoError as exc:
            self.ui.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - the shell must survive any bug
            log.exception("command /%s failed", spec.name)
            self.ui.error(f"Unexpected error in /{spec.name}: {type(exc).__name__}: {exc} (Details in errors.log)")

    # ------------------------------------------------------------------ foreground tasks

    def ensure_idle(self) -> None:
        if self.background is not None and not self.background.done():
            raise MaoError("A task is paused. Use /resume or /stop first.")

    async def run_foreground(self, coroutine: Awaitable[Any], kind: str) -> Any:
        self.ensure_idle()
        self.background = asyncio.ensure_future(coroutine)
        self.background_kind = kind
        return await self._wait_background()

    async def _wait_background(self) -> Any:
        task = self.background
        assert task is not None
        await self.ui.attach(task)
        if not task.done():
            self.ui.warn("⏸ Paused. /resume continues, /stop ends it in a controlled way. /status, /tokens and /messages are available.")
            return None
        self.background = None
        return task.result()

    async def continue_background(self) -> None:
        kind = self.background_kind
        result = await self._wait_background()
        if result is None:
            return
        if kind == "run":
            self.ui.print(result_view(result))
        elif kind == "plan" and isinstance(result, PlanDocument):
            await self.plan_decision(result)
        elif kind == "debate":
            from mao.cli.render import decision_view

            self.ui.print(decision_view(result))

    async def execute_and_report(self) -> None:
        result = await self.run_foreground(self.app.orchestrator.execute(), "run")
        if result is not None:
            self.ui.print(result_view(result))

    async def plan_decision(self, plan: PlanDocument) -> None:
        orchestrator = self.app.orchestrator
        while True:
            self.ui.print(plan_view(plan))
            if self.auto_approve_plan:
                choice = "y"
                self.ui.info("Plan confirmed automatically (--yes).")
            else:
                choice = await self.ui.choose(
                    "Run the plan?",
                    {"y": "yes", "n": "discard", "e": "change", "l": "later (/run)"},
                    default="l",
                )
            if choice == "y":
                orchestrator.approve()
                await self.execute_and_report()
                return
            if choice == "n":
                orchestrator.reject()
                self.ui.info("Plan discarded.")
                return
            if choice == "e":
                feedback = await self.ui.read_line("What should change in the plan? ")
                if not feedback or not feedback.strip():
                    continue
                revised = await self.run_foreground(orchestrator.revise(feedback.strip()), "plan")
                if revised is None:
                    return
                plan = revised
                continue
            self.ui.info("Plan saved. Run it with /run (later too, via /sessions load <id>).")
            return
