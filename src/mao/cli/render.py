"""Rich renderables for plans, results, tables and overviews."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mao.core.text import fmt_cost, fmt_duration, fmt_int, one_line
from mao.orchestration.debate import Decision
from mao.orchestration.plan import PlanDocument
from mao.sessions.store import SessionMeta, SessionResult
from mao.tokens.tracker import UsageTracker

if TYPE_CHECKING:
    from mao.agents.manager import AgentManager
    from mao.app import AppContext
    from mao.config.schema import LimitsConfig

NODE_ICONS = {
    "pending": ("○", "dim"),
    "running": ("●", "cyan"),
    "done": ("✓", "green"),
    "failed": ("✗", "red"),
    "skipped": ("⊘", "yellow"),
}
AGENT_ICONS = {
    "idle": ("●", "green"),
    "thinking": ("●", "cyan"),
    "tool": ("●", "magenta"),
    "waiting": ("◌", "yellow"),
    "done": ("✓", "green"),
    "failed": ("✗", "red"),
    "offline": ("●", "red"),
}
STATUS_STYLE = {
    "completed": "green",
    "planned": "green",
    "running": "cyan",
    "planning": "cyan",
    "paused": "yellow",
    "stopped": "yellow",
    "failed": "red",
    "created": "dim",
}


def bar(ratio: float, width: int = 20) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def banner(demo: bool = False) -> RenderableType:
    title = Text("MULTI AI ORCHESTRATOR", style="bold white")
    panel = Panel(Align.center(title), box=box.DOUBLE, width=62, padding=(0, 1))
    if not demo:
        return panel
    from mao.demo import DEMO_NOTICE

    return Group(panel, Text(DEMO_NOTICE, style="bold yellow"))


def agent_overview(manager: AgentManager, app: AppContext) -> RenderableType:
    lines = [Text("Agents:", style="bold")]
    agents = manager.all()
    if not agents:
        lines.append(Text("  (no agents configured - /agents add …)", style="yellow"))
    width = max((len(a.name) for a in agents), default=10) + 2
    for agent in agents:
        online = agent.available
        line = Text("● ", style="green" if online else "red")
        line.append(agent.name.ljust(width))
        line.append(("ONLINE " if online else "OFFLINE").ljust(8), style="green" if online else "red")
        if online:
            line.append(f" {agent.model_ref}", style="dim")
        else:
            line.append(f" {one_line(agent.offline_reason, 90)}", style="dim red")
        lines.append(line)
    return Group(*lines)


def provider_overview(app: AppContext) -> RenderableType:
    catalog = app.hub.catalog
    line = Text("Provider: ", style="bold")
    for status in catalog.statuses():
        if not status.enabled:
            continue
        pool = app.hub.keypools[status.name]
        usable = pool.is_usable() and status.reachable is not False
        working = [e for e in catalog.entries(provider=status.name) if catalog.is_available(e)]
        detail = ""
        if pool.requires_key and not pool.size:
            detail = " (no key)"
        elif status.reachable is False:
            detail = " (offline)"
        elif not working:
            detail = " (no working models)"
            usable = False
        elif status.discovered:
            detail = f" ({len(working)} models)"
        line.append("● ", style="green" if usable else "red")
        line.append(f"{status.name}{detail}  ")
    return line


def plan_view(plan: PlanDocument) -> RenderableType:
    parts: list[RenderableType] = []
    head = Text("Plan created", style="bold green")
    head.append(f"  v{plan.version}  ", style="dim")
    head.append(plan.title, style="bold")
    parts.append(head)
    if plan.summary:
        parts.append(Text(plan.summary))
    consensus = Text("Team review: ", style="bold")
    if plan.rounds:
        consensus.append("consensus ✓" if plan.consensus else "no consensus - judge decision", style="green" if plan.consensus else "yellow")
        consensus.append(f" after {plan.rounds} round(s), {len(plan.critiques)} critiques", style="dim")
    else:
        consensus.append("no critique round (simple task or disabled)", style="dim")
    parts.append(consensus)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=False)
    table.add_column("#", style="bold", no_wrap=True)
    table.add_column("Step")
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("depends on", style="dim")
    index = {node.id: i for i, node in enumerate(plan.graph.nodes, 1)}
    for i, node in enumerate(plan.graph.nodes, 1):
        title = Text(node.title)
        if node.auto_added:
            title.append(" (auto)", style="dim")
        if node.modifies_files:
            title.append(" ✎", style="yellow")
        deps = ", ".join(str(index.get(d, d)) for d in node.depends_on) or "–"
        table.add_row(f"[{i}]", title, node.kind.value, plan.assignments.get(node.id, "–"), deps)
    parts.append(table)

    if plan.risks:
        parts.append(Text("Risks:", style="bold"))
        for risk in plan.risks:
            line = Text(f"- [{risk.severity}] ", style="red" if risk.severity == "high" else "yellow")
            line.append(risk.risk)
            if risk.mitigation:
                line.append(f" → {risk.mitigation}", style="dim")
            parts.append(line)
    for label, items in (("Assumptions", plan.assumptions), ("Open questions", plan.open_questions)):
        if items:
            parts.append(Text(f"{label}:", style="bold"))
            parts.extend(Text(f"- {item}") for item in items)
    if plan.warnings:
        parts.append(Text("Notes:", style="bold yellow"))
        parts.extend(Text(f"- {w}", style="yellow") for w in plan.warnings)
    if plan.estimate:
        est = plan.estimate
        parts.append(Text(""))
        cost = Text("Estimated API cost: ", style="bold")
        cost.append(f"{fmt_cost(est.cost_usd)}  (range {fmt_cost(est.low_cost_usd)} - {fmt_cost(est.high_cost_usd)})")
        parts.append(cost)
        tokens = Text("Estimated tokens:      ", style="bold")
        tokens.append(f"~{fmt_int(est.total_tokens)}  (range {fmt_int(est.low_tokens)} - {fmt_int(est.high_tokens)})")
        parts.append(tokens)
        used = Text("So far (planning):     ", style="bold")
        used.append(f"{fmt_int(est.planning_tokens_used)} Tokens, {fmt_cost(est.planning_cost_used)}")
        parts.append(used)
        if est.unpriced_models:
            parts.append(Text("Without a price (cost incomplete): " + ", ".join(est.unpriced_models), style="yellow"))
        parts.append(Text("Estimated from typical step sizes and the prices in providers.yaml.", style="dim"))
    return Group(*parts)


def result_view(result: SessionResult) -> RenderableType:
    titles = {"completed": ("TASK COMPLETED", "green"), "stopped": ("TASK STOPPED", "yellow"), "failed": ("TASK FAILED", "red")}
    title, style = titles.get(result.status, (f"TASK {result.status.upper()}", "cyan"))
    parts: list[RenderableType] = [Panel(Align.center(Text(title, style=f"bold {style}")), box=box.DOUBLE, width=40, border_style=style)]

    def check(text: str, ok: bool = True) -> Text:
        line = Text("✓ " if ok else "✗ ", style="green" if ok else "red")
        line.append(text)
        return line

    parts.append(Text("Changes:", style="bold"))
    parts.append(check(f"{len(result.files_modified)} files modified"))
    parts.append(check(f"{len(result.files_created)} files created"))
    if result.files_deleted:
        parts.append(check(f"{len(result.files_deleted)} files deleted"))
    parts.append(check(f"{result.bugs_fixed} bugs fixed"))
    if result.tests:
        tests_ok = result.tests.get("failed", 0) == 0 and result.tests.get("errors", 0) == 0 and not result.tests.get("timed_out")
        parts.append(check(f"{result.tests.get('passed', 0)} tests passed", tests_ok))
        if not tests_ok:
            parts.append(check(f"{result.tests.get('failed', 0)} tests failed, {result.tests.get('errors', 0)} errors", False))
    else:
        parts.append(Text("- no automatic tests were run", style="dim"))
    for failed in result.nodes_failed:
        parts.append(check(f"Step failed: {failed}", False))
    for skipped in result.nodes_skipped:
        parts.append(Text(f"⊘ skipped: {skipped}", style="yellow"))
    parts.append(Text(""))
    rows = [
        ("Agents", f"{len(result.agents_used)} used"),
        ("Tokens", f"{fmt_int(result.total_tokens)} (Input {fmt_int(result.input_tokens)} / Output {fmt_int(result.output_tokens)})"),
        ("Estimated cost" if result.cost_estimated else "Cost", fmt_cost(result.cost_usd)),
        ("Duration", fmt_duration(result.duration_s)),
    ]
    git = result.git or {}
    if git.get("repository"):
        git_text = f"Branch {git.get('branch', '?')}"
        git_text += " - changes not committed (commit available: /git commit <message>)" if git.get("dirty") else " - clean"
        if git.get("checkpoint"):
            git_text += f" – Checkpoint {str(git['checkpoint'])[:10]}"
    else:
        git_text = "no git repository (roll back via backups: /rollback)"
    rows.append(("Git", git_text))
    for label, value in rows:
        line = Text(f"{label}:".ljust(16), style="bold")
        line.append(value)
        parts.append(line)
    if result.error:
        parts.append(Text(f"Reason: {result.error}", style=style))
    parts.append(Text(""))
    parts.append(Text("Summary:", style="bold"))
    parts.append(Text(result.summary or "(no summary)"))
    for label, items in (("Highlights", result.highlights), ("Open", result.remaining_issues), ("Recommendations", result.recommendations)):
        if items:
            parts.append(Text(f"{label}:", style="bold"))
            parts.extend(Text(f"- {item}") for item in items)
    return Group(*parts)


def tokens_table(tracker: UsageTracker) -> RenderableType:
    table = Table(title="Tokens pro Agent", box=box.SIMPLE_HEAD)
    table.add_column("Agent")
    table.add_column("Calls", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Context", justify="right")
    context = tracker.context_usage()
    for agent, totals in tracker.by_agent().items():
        ctx = context.get(agent)
        table.add_row(
            agent,
            str(totals.calls),
            fmt_int(totals.input_tokens),
            fmt_int(totals.output_tokens),
            fmt_int(totals.total_tokens),
            f"{ctx.ratio:.0%}" if ctx and ctx.window else "–",
        )
    totals = tracker.totals()
    table.add_section()
    table.add_row("Total", str(totals.calls), fmt_int(totals.input_tokens), fmt_int(totals.output_tokens), fmt_int(totals.total_tokens), "")
    providers = Table(title="Per provider / model", box=box.SIMPLE_HEAD)
    providers.add_column("Model")
    providers.add_column("Calls", justify="right")
    providers.add_column("Input", justify="right")
    providers.add_column("Output", justify="right")
    providers.add_column("Cached", justify="right")
    providers.add_column("Reasoning", justify="right")
    providers.add_column("Cost", justify="right")
    for model, t in tracker.by_model().items():
        providers.add_row(model, str(t.calls), fmt_int(t.input_tokens), fmt_int(t.output_tokens), fmt_int(t.cached_input_tokens), fmt_int(t.reasoning_tokens), fmt_cost(t.cost_usd))
    return Group(table, providers)


def cost_view(tracker: UsageTracker, limits: LimitsConfig) -> RenderableType:
    totals = tracker.totals()
    table = Table(title="Cost", box=box.SIMPLE_HEAD)
    table.add_column("Provider")
    table.add_column("Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="bold")
    for provider, t in tracker.by_provider().items():
        table.add_row(provider, str(t.calls), fmt_int(t.total_tokens), fmt_cost(t.cost_usd))
    table.add_section()
    table.add_row("Total", str(totals.calls), fmt_int(totals.total_tokens), fmt_cost(totals.cost_usd))
    lines: list[RenderableType] = [table]
    if limits.max_cost_usd:
        ratio = totals.cost_usd / limits.max_cost_usd
        lines.append(Text(f"Cost limit:   {bar(ratio)} {ratio:.0%}  ({fmt_cost(totals.cost_usd)} / {fmt_cost(limits.max_cost_usd)})"))
    if limits.max_tokens:
        ratio = totals.total_tokens / limits.max_tokens
        lines.append(Text(f"Token limit:  {bar(ratio)} {ratio:.0%}  ({fmt_int(totals.total_tokens)} / {fmt_int(limits.max_tokens)})"))
    reported = sum(1 for r in tracker.records if r.cost_reported)
    note = "Cost figures are estimates based on the prices in providers.yaml"
    if reported:
        note += f"; {reported} call(s) with cost reported by the provider"
    lines.append(Text(note + ".", style="dim"))
    return Group(*lines)


def agents_table(manager: AgentManager) -> RenderableType:
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Agent", style="bold")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Rights", style="dim")
    table.add_column("Tools", style="dim")
    for agent in manager.all(include_orchestrator=True):
        icon, style = AGENT_ICONS.get(agent.state.value, ("●", "white"))
        status = Text(f"{icon} {agent.state.value}", style=style)
        if not agent.available:
            status = Text(f"● offline: {one_line(agent.offline_reason, 60)}", style="red")
        model = Text(agent.model_ref or "–")
        if agent.fallback_refs:
            model.append(f"\n  Fallback: {', '.join(agent.fallback_refs)}", style="dim")
        name = agent.name + (" (internal)" if agent.internal else "")
        table.add_row(name, agent.role.name, model, status, agent.permissions.describe(), ", ".join(agent.tool_groups))
    return table


def sessions_table(metas: list[SessionMeta]) -> RenderableType:
    from datetime import datetime

    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("ID", style="bold")
    table.add_column("DATE")
    table.add_column("STATUS")
    table.add_column("TYPE", style="dim")
    table.add_column("TASK")
    table.add_column("TOKENS", justify="right")
    table.add_column("COST", justify="right")
    for meta in metas:
        table.add_row(
            meta.id,
            datetime.fromtimestamp(meta.created_at).strftime("%Y-%m-%d %H:%M"),
            Text(meta.status.value, style=STATUS_STYLE.get(meta.status.value, "white")),
            meta.kind,
            one_line(meta.task, 60),
            fmt_int(meta.total_tokens),
            fmt_cost(meta.cost_usd),
        )
    return table


def decision_view(decision: Decision) -> RenderableType:
    parts: list[RenderableType] = [Text(f"Decision ({decision.method}{', consensus' if decision.consensus else ''}, confidence {decision.confidence:.0%})", style="bold green")]
    parts.append(Text(decision.decision or "(no decision)"))
    if decision.rationale:
        parts.append(Text("Rationale:", style="bold"))
        parts.append(Text(decision.rationale))
    if decision.proposals:
        table = Table(title="Proposals", box=box.SIMPLE_HEAD)
        table.add_column("ID")
        table.add_column("Author")
        table.add_column("Title")
        table.add_column("Votes", justify="right")
        for proposal in decision.proposals:
            chosen = proposal.id == decision.chosen_proposal
            table.add_row(
                Text(proposal.id, style="bold green" if chosen else ""),
                proposal.author,
                one_line(proposal.title, 60) + (" (withdrawn)" if proposal.withdrawn else ""),
                f"{decision.vote_totals.get(proposal.id, 0):.1f}",
            )
        parts.append(table)
    for label, items in (("Evidence checked", decision.evidence_checked), ("Open risks", decision.open_risks)):
        if items:
            parts.append(Text(f"{label}:", style="bold"))
            parts.extend(Text(f"- {i}") for i in items)
    return Group(*parts)
