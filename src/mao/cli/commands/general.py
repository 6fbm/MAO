"""General commands: help, status, debug, clear, exit, doctor."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys
import tempfile

from rich import box
from rich.table import Table
from rich.text import Text

from mao import __version__
from mao.cli.commands.base import REGISTRY, CommandContext, command, preview_manager
from mao.cli.render import agent_overview, provider_overview
from mao.core.errors import ConfigError
from mao.tools.terminal import resolve_shell


@command("help", summary="Help on all commands or on one command", usage="/help [command]", aliases=("?",))
async def help_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    if args:
        spec = REGISTRY.get(args[0].lstrip("/"))
        if spec is None:
            ctx.ui.error(f"Unknown command: {args[0]}")
            return
        ctx.ui.print(Text(f"/{spec.name}", style="bold cyan"))
        ctx.ui.print(spec.summary)
        ctx.ui.print(Text(f"Usage: {spec.usage}", style="dim"))
        if spec.aliases:
            ctx.ui.print(Text("Aliases: " + ", ".join("/" + a for a in spec.aliases), style="dim"))
        return
    current = None
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    for spec in REGISTRY.all():
        if spec.group != current:
            current = spec.group
            table.add_row(Text(f"\n{current}", style="bold white"), "")
        aliases = f" ({', '.join('/' + a for a in spec.aliases)})" if spec.aliases else ""
        table.add_row(f"/{spec.name}{aliases}", spec.summary)
    ctx.ui.print(table)
    ctx.ui.print(Text("\nThe leading / is optional: 'clear' and '/clear' both work. Anything that is not\n"
                      "a command is planned as a task - use /plan <text> to force that. Details: help <command>",
                      style="dim"))


@command("status", summary="Status of the current session (agents, progress, tokens)")
async def status_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    if app.orchestrator.active is None:
        ctx.ui.print(Text(f"Workspace: {app.config.settings.workspace or '(not set - /workspace <path>)'}", style="bold"))
        ctx.ui.print(provider_overview(app))
        ctx.ui.print(agent_overview(preview_manager(app), app))
        limits = app.config.settings.limits
        ctx.ui.print(
            Text(
                f"Limits: tokens {limits.max_tokens or 'unlimited'} · cost {limits.max_cost_usd or 'unlimited'} USD · "
                f"agents {limits.max_agents} · parallel {limits.max_parallel_agents} · rounds {limits.max_rounds}",
                style="dim",
            )
        )
        ctx.ui.print(Text("No active session.", style="dim"))
        return
    ctx.ui.print(ctx.ui.status_snapshot())
    if ctx.repl.background is not None and not ctx.repl.background.done():
        ctx.ui.warn("Task paused - /resume to continue, /stop to end it.")


@command("debug", summary="Turn the detailed debug view on or off", usage="/debug on|off", subcommands=("on", "off"))
async def debug_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    if args:
        ctx.ui.debug = args[0].lower() in ("on", "1", "true", "yes")
    else:
        ctx.ui.debug = not ctx.ui.debug
    ctx.ui.info(f"Debug mode {'on' if ctx.ui.debug else 'off'}")


@command("clear", summary="Clear the screen", aliases=("cls",))
async def clear_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    ctx.ui.console.clear()


@command("exit", summary="Exit the program", aliases=("quit", "q"))
async def exit_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    background = ctx.repl.background
    if background is not None and not background.done():
        if not await ctx.ui.confirm("A task is paused. Stop it and exit?", default=False):
            return
        ctx.app.orchestrator.stop("Program exited")
        await ctx.ui.attach(background)
    ctx.repl.should_exit = True


@command("doctor", summary="Check the environment, the configuration and the providers")
async def doctor_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    rows: list[tuple[str, bool | None, str]] = []
    rows.append(("mao version", True, __version__))
    rows.append(("Python", sys.version_info >= (3, 11), f"{platform.python_version()} ({sys.executable})"))
    rows.append(("Program folder", app.paths.config_dir.exists(), str(app.paths.home)))
    try:
        app.config_manager.load()
        rows.append(("Configuration", True, str(app.paths.config_dir)))
    except ConfigError as exc:
        rows.append(("Configuration", False, str(exc)))
    workspace = app.config.settings.workspace
    rows.append(("Workspace", os.path.isdir(workspace) if workspace else None, workspace or "not set (/workspace <path>)"))
    try:
        stored = app.secrets.providers()
        rows.append(("Secret store", True, f"{app.secrets.backend}; keys for: {', '.join(stored) or 'none'}"))
    except ConfigError as exc:
        rows.append(("Secret store", False, str(exc)))
    logs = app.paths.logs_dir(app.config.settings.logs_dir)
    try:
        logs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=logs):
            pass
        rows.append(("Logs folder", True, str(logs)))
    except OSError as exc:
        rows.append(("Logs folder", False, f"{logs}: {exc}"))
    git = shutil.which("git")
    rows.append(("Git", git is not None, git or "not found (git features disabled)"))
    rows.append(("Terminal shell", True, resolve_shell(app.config.tools.terminal.shell)))
    rows.append(("Console", True if sys.stdout.isatty() else None, f"encoding {sys.stdout.encoding}, interactive={ctx.ui.interactive}"))
    with ctx.ui.console.status("Checking providers …"):
        try:
            health = await asyncio.wait_for(app.hub.check_health(), 30)
        except asyncio.TimeoutError:
            health = {}
    for name, provider in app.config.providers.providers.items():
        if not provider.enabled:
            continue
        pool = app.hub.keypools[name]
        if pool.requires_key and pool.size == 0:
            rows.append((f"Provider {name}", None, "no API key (/providers key add " + name + ")"))
            continue
        result = health.get(name)
        if result is None:
            rows.append((f"Provider {name}", None, "not checked"))
        else:
            detail = f"{result.message}" + (f", {result.models} models" if result.models is not None else "")
            rows.append((f"Provider {name}", result.ok, detail))
    broken = [e for e in app.hub.catalog.entries() if e.note.startswith("broken")]
    for entry in broken:
        rows.append((f"Model {entry.ref}", False, entry.note))
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    for label, ok, detail in rows:
        state = Text("OK", style="green") if ok else (Text("–", style="yellow") if ok is None else Text("ERROR", style="red"))
        table.add_row(label, state, detail)
    ctx.ui.print(table)
