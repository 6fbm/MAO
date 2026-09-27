"""Agent management commands."""

from __future__ import annotations

import re

from rich import box
from rich.table import Table
from rich.text import Text

from mao.agents.roles import RoleRegistry
from mao.cli.commands.base import CommandContext, command, preview_manager, require_args
from mao.cli.render import agents_table
from mao.config.loader import parse_scalar
from mao.core.errors import ConfigError, MaoError
from mao.core.text import one_line

_COUNT_RE = re.compile(r"^(\d+)x$", re.I)
SETTABLE = {"model", "role", "count", "enabled", "temperature", "max_steps", "max_output_tokens", "system_prompt", "extra_instructions", "fallback_models", "tools", "capabilities"}


def _sanitize(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.\-]+", "-", name).strip("-.")
    return clean[:40] or "agent"


def _reload_agents(ctx: CommandContext) -> None:
    ctx.app.config.agents = ctx.app.config_manager.load_section("agents")  # type: ignore[assignment]


@command(
    "agents",
    summary="Show, add, remove and configure agents",
    usage="/agents | /agents add [Nx] <model> <role> [--name NAME] | /agents remove <name> | /agents show <name> | /agents set <name> <field> <value> | /agents roles",
    group="Agents",
    subcommands=("add", "remove", "show", "set", "roles"),
)
async def agents_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    if not args:
        manager = preview_manager(app)
        ctx.ui.print(agents_table(manager))
        for warning in manager.warnings:
            ctx.ui.warn(warning)
        if app.orchestrator.active is None:
            ctx.ui.print(Text("Preview from agents.yaml - models are reassigned for every session.", style="dim"))
        return
    sub = args[0].lower()
    roles = RoleRegistry(app.config.roles.roles)

    if sub == "roles":
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Role", style="bold")
        table.add_column("Description")
        table.add_column("Capabilities", style="dim")
        table.add_column("Tools", style="dim")
        table.add_column("Tier")
        for role in roles.all():
            table.add_row(role.name, role.description, ", ".join(role.capabilities), ", ".join(role.tools), role.preferred_tier)
        ctx.ui.print(table)
        return

    if sub == "add":
        tokens = list(args[1:])
        name: str | None = None
        if "--name" in tokens:
            index = tokens.index("--name")
            if index + 1 >= len(tokens):
                raise MaoError("--name needs a value")
            name = tokens[index + 1]
            del tokens[index : index + 2]
        count = 1
        if tokens and _COUNT_RE.match(tokens[0]):
            count = int(_COUNT_RE.match(tokens[0]).group(1))  # type: ignore[union-attr]
            tokens = tokens[1:]
        if len(tokens) < 2:
            raise MaoError("Usage: /agents add [Nx] <model> <role> [--name NAME]   e.g. /agents add 3x gemini coder")
        model, role_name = tokens[0], tokens[1]
        role = roles.get(role_name)
        if model.lower() != "auto":
            try:
                entry = app.hub.catalog.resolve(model)
                if not app.hub.catalog.is_available(entry):
                    ctx.ui.warn(f"Model {entry.ref} is currently unavailable: {app.hub.catalog.unavailable_reason(entry)}")
            except ConfigError as exc:
                raise MaoError(str(exc)) from exc
        existing = {a.name.lower() for a in app.config.agents.agents}
        base = _sanitize(name or f"{model.split('/')[-1]}-{role.name}")
        final = base
        suffix = 2
        while final.lower() in existing:
            final = f"{base}-{suffix}"
            suffix += 1

        def mutate(data: dict) -> None:
            entry: dict = {"name": final, "role": role.name, "model": model}
            if count > 1:
                entry["count"] = count
            data.setdefault("agents", []).append(entry)

        app.config_manager.update_raw("agents", mutate)
        _reload_agents(ctx)
        ctx.ui.success(f"Agent '{final}' added: {count}x {role.name} with model {model}. Takes effect from the next session.")
        return

    if sub == "remove":
        require_args(args, 2, "/agents remove <name>")
        target = args[1].lower()
        if not any(a.name.lower() == target for a in app.config.agents.agents):
            raise MaoError(f"No agent '{args[1]}' in agents.yaml")
        if not await ctx.ui.confirm(f"Agent '{args[1]}' from agents.yaml?", default=False):
            return

        def remove(data: dict) -> None:
            data["agents"] = [a for a in data.get("agents", []) if str(a.get("name", "")).lower() != target]

        app.config_manager.update_raw("agents", remove)
        _reload_agents(ctx)
        ctx.ui.success(f"Agent '{args[1]}' removed.")
        return

    if sub == "show":
        require_args(args, 2, "/agents show <name>")
        manager = preview_manager(app)
        agent = manager.get(args[1]) or manager.find(args[1])
        if agent is None:
            raise MaoError(f"No agent '{args[1]}'")
        lines = [
            f"Name:          {agent.name}  (config: {agent.config_name})",
            f"Role:          {agent.role.name} – {agent.role.description}",
            f"Model:         {agent.model_ref or 'offline: ' + agent.offline_reason}",
            f"Fallbacks:     {', '.join(agent.fallback_refs) or '-'}",
            f"Rights:        {agent.permissions.describe()}",
            f"Tools:         {', '.join(agent.tool_groups)}",
            f"Capabilities: {', '.join(agent.capabilities)}",
            f"Max. steps: {agent.max_steps}   Temperature: {agent.temperature if agent.temperature is not None else 'default'}",
            f"System-Prompt: {one_line(agent.system_prompt, 400)}",
        ]
        for line in lines:
            ctx.ui.print(line)
        return

    if sub == "set":
        require_args(args, 4, "/agents set <name> <field> <value>")
        target, field_name = args[1].lower(), args[2]
        value_text = raw.split(None, 3)[3] if len(raw.split(None, 3)) > 3 else args[3]
        top = field_name.split(".", 1)[0]
        if top not in SETTABLE and top != "permissions":
            raise MaoError(f"Field cannot be changed. Allowed: {', '.join(sorted(SETTABLE))}, permissions.<right>")
        if top in ("fallback_models", "tools", "capabilities"):
            value: object = [v.strip() for v in value_text.split(",") if v.strip()]
        elif top in ("system_prompt", "extra_instructions", "model", "role"):
            value = value_text
        else:
            value = parse_scalar(value_text)

        def update(data: dict) -> None:
            for agent in data.get("agents", []):
                if str(agent.get("name", "")).lower() == target:
                    if top == "permissions":
                        key = field_name.split(".", 1)[1] if "." in field_name else ""
                        if not key:
                            raise MaoError("Usage: /agents set <name> permissions.<filesystem|terminal|internet|git|…> <value>")
                        agent.setdefault("permissions", {})[key] = value
                    else:
                        agent[field_name] = value
                    return
            raise MaoError(f"No agent '{args[1]}' in agents.yaml")

        app.config_manager.update_raw("agents", update)
        _reload_agents(ctx)
        ctx.ui.success(f"{args[1]}.{field_name} = {value!r} saved. Takes effect from the next session.")
        return
    raise MaoError(f"Unknown subcommand: {sub}")
