"""Configuration commands."""

from __future__ import annotations

from rich.syntax import Syntax
from rich.text import Text

from mao.agents.roles import RoleRegistry
from mao.cli.commands.base import CommandContext, command, require_args
from mao.config.loader import CONFIG_FILES, parse_scalar
from mao.core.errors import ConfigError, MaoError


def cross_check(ctx: CommandContext) -> list[str]:
    app = ctx.app
    config = app.config
    warnings: list[str] = []
    roles = RoleRegistry(config.roles.roles)
    groups = set(config.tools.enabled_groups)
    for agent in config.agents.agents:
        try:
            roles.get(agent.role)
        except ConfigError as exc:
            warnings.append(f"Agent {agent.name}: {exc}")
        for ref in [agent.model, *agent.fallback_models]:
            if ref.lower() == "auto":
                if not config.settings.orchestration.auto_model_selection:
                    warnings.append(f"Agent {agent.name}: model 'auto', but auto_model_selection is disabled")
                continue
            try:
                app.hub.catalog.resolve(ref)
            except ConfigError as exc:
                warnings.append(f"Agent {agent.name}: {exc}")
        for group in agent.tools or []:
            if group not in groups:
                warnings.append(f"Agent {agent.name}: tool group '{group}' is not enabled")
    return warnings


@command(
    "config",
    summary="Show, change, check or reload the configuration",
    usage="/config | /config show <section> | /config set <section.path> <value> | /config validate | /config reload",
    group="Configuration",
    subcommands=("show", "set", "validate", "reload", "path"),
)
async def config_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    settings = app.config.settings
    if not args or args[0].lower() == "path":
        ctx.ui.print(Text("Configuration", style="bold"))
        ctx.ui.print(f"Program folder: {app.paths.home}")
        ctx.ui.print(f"Configuration:  {app.paths.config_dir}  ({', '.join(f for f, _ in CONFIG_FILES.values())})")
        ctx.ui.print(f"Sessions/Logs:  {app.paths.logs_dir(settings.logs_dir)}")
        ctx.ui.print(f"Secret-Store:   {app.secrets.path} ({app.secrets.backend})")
        ctx.ui.print(f"Workspace:      {settings.workspace or '-'}")
        ctx.ui.print(f"Model choice:   {'automatic' if settings.orchestration.auto_model_selection else 'manual'} ({settings.orchestration.model_policy}), local_only={settings.privacy.local_only}")
        limits = settings.limits
        ctx.ui.print(f"Limits:         tokens {limits.max_tokens}, cost {limits.max_cost_usd} USD, agents {limits.max_agents}, rounds {limits.max_rounds}, parallel {limits.max_parallel_agents}")
        ctx.ui.print(Text("Sections: " + ", ".join(CONFIG_FILES) + "   Example: /config set settings.limits.max_rounds 5", style="dim"))
        return
    sub = args[0].lower()
    if sub == "show":
        require_args(args, 2, "/config show <section>")
        section = args[1]
        text = app.config_manager.read_text(section)
        ctx.ui.print(Syntax(text or "# (file missing - defaults are active)", "yaml", word_wrap=True, theme="ansi_dark"))
        return
    if sub == "set":
        require_args(args, 3, "/config set <section.path> <value>")
        path = args[1]
        value_text = raw.split(None, 2)[2]
        value = parse_scalar(value_text)
        app.config_manager.set_value(path, value)
        live_applied = False
        if path.startswith("settings."):
            node = app.config.settings
            parts = path.split(".")[1:]
            try:
                for part in parts[:-1]:
                    node = getattr(node, part)
                setattr(node, parts[-1], value)
                live_applied = True
            except (AttributeError, ValueError):
                live_applied = False
        if not live_applied:
            app.config = app.config_manager.load()
        ctx.ui.success(f"{path} = {value!r} saved." + ("" if live_applied else " Takes effect from the next session (or /config reload)."))
        return
    if sub == "validate":
        try:
            app.config_manager.load()
        except ConfigError as exc:
            ctx.ui.error(str(exc))
            return
        warnings = cross_check(ctx)
        if warnings:
            for warning in warnings:
                ctx.ui.warn(warning)
        else:
            ctx.ui.success("The configuration is valid.")
        return
    if sub == "reload":
        ctx.repl.ensure_idle()
        with ctx.ui.console.status("Reloading the configuration …"):
            await app.orchestrator.close_active()
            await app.reload_config()
        ctx.ui.success("Configuration reloaded.")
        for name, message in app.discovery_messages.items():
            ctx.ui.info(f"{name}: {message}")
        return
    raise MaoError(f"Unknown subcommand: {sub}")
