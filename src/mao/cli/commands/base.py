"""Command registry and helpers shared by all command modules."""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mao.agents.manager import AgentManager
from mao.agents.roles import RoleRegistry
from mao.core.errors import MaoError
from mao.core.events import EventBus
from mao.models.selector import ModelSelector

if TYPE_CHECKING:
    from mao.app import AppContext
    from mao.cli.console import ConsoleUI
    from mao.cli.repl import Repl

Handler = Callable[["CommandContext", list[str], str], Awaitable[None]]

GROUP_ORDER = ["Tasks", "Agents", "Models", "Usage", "Sessions", "Workspace & Git", "Configuration", "General"]


@dataclass
class CommandSpec:
    name: str
    handler: Handler
    summary: str
    usage: str
    group: str
    aliases: tuple[str, ...] = ()
    subcommands: tuple[str, ...] = ()


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandSpec] = {}
        self._aliases: dict[str, str] = {}

    def add(self, spec: CommandSpec) -> None:
        self._commands[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name

    def get(self, name: str) -> CommandSpec | None:
        name = name.lower()
        return self._commands.get(name) or self._commands.get(self._aliases.get(name, ""))

    def all(self) -> list[CommandSpec]:
        return sorted(self._commands.values(), key=lambda c: (GROUP_ORDER.index(c.group) if c.group in GROUP_ORDER else 99, c.name))

    def names(self) -> list[str]:
        return sorted([*self._commands, *self._aliases])


REGISTRY = CommandRegistry()


def command(
    name: str,
    *,
    summary: str,
    usage: str = "",
    group: str = "General",
    aliases: tuple[str, ...] = (),
    subcommands: tuple[str, ...] = (),
) -> Callable[[Handler], Handler]:
    def decorator(func: Handler) -> Handler:
        REGISTRY.add(CommandSpec(name, func, summary, usage or f"/{name}", group, aliases, subcommands))
        return func

    return decorator


@dataclass
class CommandContext:
    repl: Repl

    @property
    def app(self) -> AppContext:
        return self.repl.app

    @property
    def ui(self) -> ConsoleUI:
        return self.repl.ui


def split_args(raw: str) -> list[str]:
    try:
        parts = shlex.split(raw, posix=False)
    except ValueError:
        parts = raw.split()
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


def preview_manager(app: AppContext) -> AgentManager:
    """The active session's agents, or a fresh preview from the current configuration."""
    if app.orchestrator.active is not None:
        return app.orchestrator.active.manager
    manager = AgentManager(
        agents_config=app.config.agents,
        roles=RoleRegistry(app.config.roles.roles),
        catalog=app.hub.catalog,
        selector=ModelSelector(app.hub.catalog, app.config.settings.orchestration.model_policy),
        settings=app.config.settings,
        permissions=app.config.permissions,
        bus=EventBus(),
    )
    manager.build()
    return manager


def require_args(args: list[str], count: int, usage: str) -> None:
    if len(args) < count:
        raise MaoError(f"Usage: {usage}")


def parse_limit(value: str, *, as_float: bool = False) -> float | int | None:
    text = value.strip().lower().replace("_", "").replace(",", "")
    if text in ("none", "off", "unlimited", "null"):
        return None
    try:
        number = float(text) if as_float else int(float(text))
    except ValueError as exc:
        raise MaoError(f"Invalid number: {value}") from exc
    if number < 0:
        raise MaoError("The value must not be negative")
    return number
