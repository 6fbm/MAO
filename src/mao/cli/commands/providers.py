"""Provider, API key and model commands."""

from __future__ import annotations

import asyncio

from rich import box
from rich.table import Table
from rich.text import Text

from mao.cli.commands.base import CommandContext, command, require_args
from mao.core.errors import ConfigError, MaoError, ProviderError
from mao.core.text import fmt_int


@command(
    "providers",
    summary="Manage providers, API keys and reachability",
    usage="/providers | /providers test [name] | /providers key add|list|remove <provider> [fingerprint|nr] | /providers enable|disable <name>",
    group="Models",
    subcommands=("test", "key", "enable", "disable"),
)
async def providers_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    if not args:
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Provider", style="bold")
        table.add_column("Type", style="dim")
        table.add_column("Enabled")
        table.add_column("Keys")
        table.add_column("Status")
        table.add_column("Models", justify="right")
        for status in app.hub.catalog.statuses():
            config = app.config.providers.providers[status.name]
            pool = app.hub.keypools[status.name]
            if not config.requires_api_key:
                keys = Text("not needed (local)", style="dim")
            elif pool.size:
                keys = Text(", ".join(f"{row['fingerprint']} ({row['source']}, {row['state']})" for row in pool.status()))
            else:
                keys = Text("none", style="yellow")
            if not config.enabled:
                state = Text("disabled", style="dim")
            elif status.reachable is False:
                state = Text(f"offline: {status.message[:60]}", style="red")
            elif pool.is_usable():
                state = Text("ready", style="green")
            else:
                state = Text("no key", style="yellow")
            models = len(app.hub.catalog.entries(provider=status.name, include_disabled=False))
            table.add_row(status.name, config.type, "yes" if config.enabled else "no", keys, state, str(models))
        ctx.ui.print(table)
        ctx.ui.print(Text(f"Secret-Store: {app.secrets.backend}. Keys are never shown or logged in clear text.", style="dim"))
        return

    sub = args[0].lower()
    if sub == "test":
        names = args[1:] or None
        with ctx.ui.console.status("Checking reachability …"):
            try:
                results = await asyncio.wait_for(app.hub.check_health(names), 60)
            except asyncio.TimeoutError:
                ctx.ui.error("Timeout during the check")
                return
        if not results:
            ctx.ui.warn("No checkable providers (enabled and with a key, or local).")
        for name, health in results.items():
            text = f"{name}: {health.message}" + (f" - {health.models} models" if health.models is not None else "") + (f" ({health.latency_s:.2f}s)" if health.latency_s else "")
            (ctx.ui.success if health.ok else ctx.ui.error)(text)
        return

    if sub == "key":
        require_args(args, 3, "/providers key add|list|remove <provider> [fingerprint|nr]")
        action, provider = args[1].lower(), args[2]
        if provider not in app.config.providers.providers:
            raise MaoError(f"Unknown provider: {provider}")
        if action == "add":
            ctx.ui.info(f"API key for '{provider}' (it is stored encrypted, input is hidden).")
            key = await ctx.ui.read_line("API-Key: ", password=True)
            if not key or not key.strip():
                ctx.ui.warn("Cancelled.")
                return
            fingerprint = app.secrets.add(provider, key)
            pool = app.hub.reload_keys(provider)
            ctx.ui.success(f"Key {fingerprint} saved ({app.secrets.backend}). {pool.size} key(s) active for {provider}.")
            ctx.ui.info("New keys take effect from the next session. Tip: /providers test " + provider)
        elif action == "list":
            pool = app.hub.keypools[provider]
            if not pool.size:
                ctx.ui.warn("No keys (neither environment variables nor the secret store).")
            for index, row in enumerate(pool.status(), 1):
                ctx.ui.print(f"{index}. {row['fingerprint']}  source: {row['source']}  status: {row['state']}  uses: {row['uses']}")
            env_names = app.config.providers.providers[provider].api_keys.env
            ctx.ui.print(Text(f"Environment variables: {', '.join(env_names) or '-'}", style="dim"))
        elif action == "remove":
            require_args(args, 4, "/providers key remove <provider> <fingerprint|nr>")
            if app.secrets.remove(provider, args[3]):
                app.hub.reload_keys(provider)
                ctx.ui.success("Key removed from the secret store.")
            else:
                ctx.ui.error("No matching key in the secret store (remove keys from environment variables there).")
        else:
            raise MaoError("Action must be add, list or remove")
        return

    if sub in ("enable", "disable"):
        require_args(args, 2, f"/providers {sub} <name>")
        name = args[1]
        if name not in app.config.providers.providers:
            raise MaoError(f"Unknown provider: {name}")

        def mutate(data: dict) -> None:
            data.setdefault("providers", {}).setdefault(name, {})["enabled"] = sub == "enable"

        app.config_manager.update_raw("providers", mutate)
        with ctx.ui.console.status("Reloading the configuration …"):
            await app.reload_config()
        ctx.ui.success(f"Provider {name} {'enabled' if sub == 'enable' else 'disabled'}.")
        return
    raise MaoError(f"Unknown subcommand: {sub}")


@command(
    "models",
    summary="Show models, detect them, or show details",
    usage="/models [provider] | /models discover [provider] | /models info <provider/model>",
    group="Models",
    subcommands=("discover", "info"),
)
async def models_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    catalog = app.hub.catalog
    if args and args[0].lower() == "discover":
        targets = args[1:] or [n for n, c in app.config.providers.providers.items() if c.enabled and app.hub.keypools[n].is_usable()]
        for name in targets:
            try:
                with ctx.ui.console.status(f"Detecting models at {name} …"):
                    added = await asyncio.wait_for(app.hub.discover(name), 60)
                catalog.set_reachability(name, True)
                ctx.ui.success(f"{name}: {len(added)} new models detected" + (": " + ", ".join(e.key for e in added[:12]) if added else ""))
            except (ProviderError, ConfigError, asyncio.TimeoutError) as exc:
                ctx.ui.error(f"{name}: {exc or 'Timeout'}")
        return
    if args and args[0].lower() == "info":
        require_args(args, 2, "/models info <provider/model>")
        entry = catalog.resolve(args[1])
        cfg = entry.config
        lines = [
            f"Reference:     {entry.ref}  (API ID: {entry.api_id}, source: {entry.source})",
            f"Available:     {'yes' if catalog.is_available(entry) else 'no - ' + catalog.unavailable_reason(entry)}",
            f"Tier:          {cfg.tier}",
            f"Context:       {fmt_int(cfg.context_window)} tokens, max. output {fmt_int(cfg.max_output_tokens)}",
            f"Capabilities: {', '.join(cfg.capabilities)}  (Tool-Calling: {cfg.tool_calling})",
            f"Prices (USD/1M): input {cfg.pricing.input_per_mtok}, Output {cfg.pricing.output_per_mtok}, Cached {cfg.pricing.cached_input_per_mtok}",
        ]
        if cfg.pricing.long_context_threshold:
            lines.append(f"Long Context ab {fmt_int(cfg.pricing.long_context_threshold)}: Input {cfg.pricing.long_input_per_mtok}, Output {cfg.pricing.long_output_per_mtok}")
        if entry.note:
            lines.append(f"Note:          {entry.note}")
        for line in lines:
            ctx.ui.print(line)
        return
    provider_filter = args[0] if args else None
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Model", style="bold")
    table.add_column("Tier")
    table.add_column("Context", justify="right")
    table.add_column("$ In/Out", justify="right")
    table.add_column("Capabilities", style="dim")
    table.add_column("Tools")
    table.add_column("Available")
    entries = catalog.entries(provider=provider_filter)
    for entry in sorted(entries, key=lambda e: (e.provider, e.key)):
        available = catalog.is_available(entry)
        reason = catalog.unavailable_reason(entry)
        price = "local" if entry.local else f"{entry.config.pricing.input_per_mtok:g}/{entry.config.pricing.output_per_mtok:g}"
        table.add_row(
            entry.ref,
            entry.config.tier,
            fmt_int(entry.config.context_window),
            price,
            ", ".join(entry.config.capabilities),
            entry.config.tool_calling,
            Text("yes", style="green") if available else Text(f"no ({reason})", style="red" if "broken" in reason else "yellow"),
        )
    ctx.ui.print(table)
    ctx.ui.print(Text("New models of a provider: /models discover <provider>   Details: /models info <ref>", style="dim"))
