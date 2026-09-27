"""Token, cost and limit commands."""

from __future__ import annotations

from mao.cli.commands.base import CommandContext, command, parse_limit, require_args
from mao.cli.render import cost_view, tokens_table
from mao.core.errors import MaoError
from mao.core.text import fmt_cost, fmt_int

LIMIT_COMMANDS = {
    "max-tokens": ("max_tokens", False, "Tokens"),
    "max-cost": ("max_cost_usd", True, "Cost (USD)"),
    "max-agents": ("max_agents", False, "Agents"),
    "max-rounds": ("max_rounds", False, "Discussion rounds"),
    "max-parallel": ("max_parallel_agents", False, "Parallel agents"),
}


@command("tokens", summary="Token usage per agent, provider and model", group="Usage")
async def tokens_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    ctx.ui.print(tokens_table(srt.tracker))


@command("cost", summary="Cost of the active session and budget usage", group="Usage")
async def cost_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    ctx.ui.print(cost_view(srt.tracker, srt.budget.limits))


def _make_limit_command(name: str, field: str, as_float: bool, label: str) -> None:
    async def handler(ctx: CommandContext, args: list[str], raw: str) -> None:
        app = ctx.app
        limits = app.config.settings.limits
        if not args:
            value = getattr(limits, field)
            shown = "unlimited" if value is None else (fmt_cost(value) if as_float else fmt_int(value))
            ctx.ui.info(f"{label}: {shown}")
            return
        require_args(args, 1, f"/{name} <value>")
        value = parse_limit(args[0], as_float=as_float)
        if value is None and field in ("max_agents", "max_rounds", "max_parallel_agents"):
            raise MaoError("This value needs a number")
        try:
            setattr(limits, field, value)
        except ValueError as exc:
            raise MaoError(f"Invalid value: {exc}") from exc
        app.config_manager.set_value(f"settings.limits.{field}", value)
        srt = app.orchestrator.active
        if srt is not None:
            if srt.budget.limits is not limits:
                setattr(srt.budget.limits, field, value)
            srt.budget.reset_declines()
        shown = "unlimited" if value is None else (fmt_cost(value) if as_float else fmt_int(value))
        ctx.ui.success(f"{label} set to: {shown}")

    command(name, summary=f"Show or set the limit for {label}", usage=f"/{name} [value|none]", group="Usage")(handler)


for _name, (_field, _float, _label) in LIMIT_COMMANDS.items():
    _make_limit_command(_name, _field, _float, _label)
