"""Live dashboard: event-driven state plus rendering."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mao.cli.render import AGENT_ICONS, NODE_ICONS, bar
from mao.core.events import (
    AgentMessageEvent,
    AgentStatusEvent,
    ApprovalEvent,
    BudgetEvent,
    ContextEvent,
    DebateEvent,
    DecisionEvent,
    Event,
    LLMCallEvent,
    LogEvent,
    PhaseEvent,
    ProviderIssueEvent,
    SessionStatusEvent,
    TaskNodeEvent,
    ToolEvent,
)
from mao.core.text import fmt_cost, fmt_duration, fmt_int, one_line

if TYPE_CHECKING:
    from mao.app import AppContext

ACTIVITY_LABEL = {"thinking": "is thinking", "tool": "using tools", "waiting": "waiting", "idle": "ready", "done": "done", "failed": "error"}
MESSAGE_STYLE = {"critique": "yellow", "decision": "bold green", "result": "green", "question": "blue", "answer": "blue", "proposal": "cyan", "warning": "red", "finding": "magenta"}
ISSUE_TEXT = {
    "retry": "Retry - attempt {attempt}/{max} in {wait}s",
    "rotate_key": "switching API key",
    "fallback": "switching to fallback model {next}",
    "give_up": "no further fallback models",
    "adjust_request": "adjusting the request",
    "wait": "all keys in cooldown - waiting {wait}s",
}


class DashboardState:
    def __init__(self, app: AppContext, max_feed: int = 300) -> None:
        self.app = app
        self.session_id: str | None = None
        self.on_feed: Callable[[str, str], None] | None = None
        self.debug = False
        self.reset(None)
        self._max_feed = max_feed

    def reset(self, session_id: str | None) -> None:
        self.session_id = session_id
        self.phase = "–"
        self.phase_detail = ""
        self.phases: list[tuple[str, str]] = []
        self.status = "–"
        self.agents: dict[str, dict[str, str]] = {}
        self.nodes: dict[str, dict[str, str | None]] = {}
        self.feed: deque[tuple[float, str, str]] = deque(maxlen=300)
        self.messages: deque[tuple[float, str, str]] = deque(maxlen=200)
        self.started = time.monotonic()
        self.llm_calls_running = 0

    def _add(self, text: str, style: str = "", *, debug_only: bool = False) -> None:
        if debug_only and not self.debug:
            return
        self.feed.append((time.time(), style, text))
        if self.on_feed is not None:
            self.on_feed(text, style)

    def on_event(self, event: Event) -> None:
        if event.session_id and event.session_id != self.session_id:
            self.reset(event.session_id)
        if isinstance(event, PhaseEvent):
            self.phase, self.phase_detail = event.phase, event.detail
            self.phases.append((event.phase, event.detail))
            self._add(f"▶ {event.phase} {event.detail}".strip(), "bold cyan")
        elif isinstance(event, SessionStatusEvent):
            self.status = event.status
            if event.status in ("planning", "running"):
                self.started = time.monotonic()
            self._add(f"Session: {event.status} {one_line(event.message, 100)}", "bold")
        elif isinstance(event, AgentStatusEvent):
            self.agents[event.agent] = {"status": event.status, "activity": event.activity, "model": event.model or ""}
        elif isinstance(event, LLMCallEvent):
            if event.stage == "start":
                self.llm_calls_running += 1
                self._add(f"{event.agent} → {event.provider}/{event.model} ({event.purpose}, ~{fmt_int(event.input_tokens)} Tokens)", "dim", debug_only=True)
            else:
                self.llm_calls_running = max(0, self.llm_calls_running - 1)
                self._add(
                    f"{event.agent} ← {fmt_int(event.input_tokens)}/{fmt_int(event.output_tokens)} Tokens, {fmt_cost(event.cost_usd)}, {event.latency_s:.1f}s",
                    "dim",
                    debug_only=True,
                )
        elif isinstance(event, ToolEvent):
            if event.stage == "start":
                self._add(f"{event.agent} ⚙ {event.tool}({event.summary})", "magenta")
            elif not event.ok:
                self._add(f"{event.agent} ✗ {event.tool}: {event.summary}", "red")
            else:
                detail = f"\n{one_line(event.detail, 300)}" if event.detail else ""
                self._add(f"{event.agent} ✓ {event.tool}: {event.summary}{detail}", "dim", debug_only=True)
        elif isinstance(event, AgentMessageEvent):
            recipients = ", ".join(event.recipients)
            text = f"{event.sender} → {recipients} [{event.msg_kind}]: {event.summary}"
            self.messages.append((time.time(), MESSAGE_STYLE.get(event.msg_kind, ""), f"{event.sender} → {recipients} [{event.msg_kind}]\n{event.content}"))
            self._add(text, MESSAGE_STYLE.get(event.msg_kind, "white"))
            if self.debug:
                self._add("   " + one_line(event.content, 400), "dim")
        elif isinstance(event, TaskNodeEvent):
            self.nodes[event.node_id] = {"title": event.title, "status": event.status, "agent": event.agent, "outcome": event.outcome}
            style = {"done": "green", "failed": "red", "skipped": "yellow", "running": "cyan"}.get(event.status, "")
            self._add(f"Step {event.node_id} {event.status}: {event.title} ({event.agent or '-'}) {one_line(event.detail, 120)}", style)
        elif isinstance(event, ProviderIssueEvent):
            action = ISSUE_TEXT.get(event.action, event.action).format(attempt=event.attempt, max=event.max_attempts, wait=event.wait_s, next=event.next_model or "-")
            style = "red" if event.action in ("give_up", "fallback") else "yellow"
            self._add(f"{event.provider}/{event.model} API error ({event.error_type}): {one_line(event.message, 120)} – {action}", style)
        elif isinstance(event, ApprovalEvent):
            self._add(f"Approval '{event.decision}': {event.agent} {event.action} {one_line(event.target, 100)} (risk {event.risk})", "yellow")
        elif isinstance(event, BudgetEvent):
            style = {"warning": "yellow", "exceeded": "bold red", "extended": "green"}[event.level]
            self._add(f"Budget {event.metric}: {event.level} ({event.used:,.2f} / {event.limit:,.2f})", style)
        elif isinstance(event, DebateEvent):
            self._add(f"Debate {event.stage} (round {event.round}): {one_line(event.detail, 140)}", "cyan")
        elif isinstance(event, DecisionEvent):
            consensus = ", consensus" if event.consensus else ""
            self._add(f"Decision ({event.method}{consensus}): {event.summary}", "bold green")
        elif isinstance(event, ContextEvent):
            self._add(f"{event.agent}: context compacted {fmt_int(event.before_tokens)} → {fmt_int(event.after_tokens)} tokens", "blue")
        elif isinstance(event, LogEvent):
            style = {"warning": "yellow", "error": "red", "debug": "dim"}.get(event.level, "")
            self._add(f"[{event.source}] {event.message}", style, debug_only=event.level == "debug")


def render_dashboard(state: DashboardState, app: AppContext, *, view: str = "events", paused: bool = False, height: int = 40) -> RenderableType:
    srt = app.orchestrator.active
    header = Text("MULTI AI ORCHESTRATOR", style="bold white")
    if srt is not None:
        header.append(f"   Session {srt.session.id}", style="bold")
        header.append(f"   {srt.mode.value.upper()}", style="bold magenta")
    header.append(f"   Phase: {state.phase}", style="cyan")
    header.append(f"   {fmt_duration(time.monotonic() - state.started)}", style="dim")
    if paused:
        header.append("   ⏸ PAUSED", style="bold yellow")
    if app.demo:
        header.append("   DEMO (simuliert)", style="bold yellow")
    parts: list[RenderableType] = [Panel(header, box=box.DOUBLE, padding=(0, 1))]
    if srt is not None:
        task = Text("Task: ", style="bold")
        task.append(f"> {one_line(srt.session.meta.task, 110)}")
        parts.append(task)

    agents_table = Table(box=None, show_header=True, header_style="bold", pad_edge=False, expand=True)
    agents_table.add_column("ACTIVE AGENTS", ratio=3)
    agents_table.add_column("Tokens", justify="right", ratio=1)
    usage = srt.tracker.by_agent() if srt is not None else {}
    rows = 0
    manager_agents = srt.manager.all(include_orchestrator=True) if srt is not None else []
    ordered = sorted(
        manager_agents,
        key=lambda a: (state.agents.get(a.name, {}).get("status", "idle") not in ("thinking", "tool"), a.name not in usage, a.name),
    )
    max_rows = 10
    for agent in ordered:
        info = state.agents.get(agent.name, {})
        status = info.get("status") or ("offline" if not agent.available else "idle")
        if status == "idle" and agent.name not in usage and rows >= 6:
            continue
        if rows >= max_rows:
            break
        icon, style = AGENT_ICONS.get(status, ("●", "white"))
        name = Text(f"{icon} ", style=style)
        name.append(agent.name, style="bold")
        name.append(f"  {agent.model_ref or 'offline'}", style="dim")
        activity = info.get("activity") or ""
        label = ACTIVITY_LABEL.get(status, status)
        name.append(f"\n  └─ {label}{': ' + one_line(activity, 70) if activity else ''}", style=style if status != "idle" else "dim")
        tokens = usage.get(agent.name)
        agents_table.add_row(name, fmt_int(tokens.total_tokens) if tokens else "–")
        rows += 1
    hidden = len(manager_agents) - rows
    if hidden > 0:
        agents_table.add_row(Text(f"… {hidden} more agents", style="dim"), "")

    right: RenderableType
    plan = srt.plan if srt is not None else None
    if plan is not None and plan.graph.nodes and srt is not None and srt.mode.value == "run":
        nodes_table = Table(box=None, show_header=True, header_style="bold", pad_edge=False, expand=True)
        nodes_table.add_column("TASKS", ratio=3)
        nodes_table.add_column("Agent", ratio=1, style="dim")
        for node in plan.graph.nodes[:14]:
            icon, style = NODE_ICONS.get(node.status.value, ("○", ""))
            title = Text(f"{icon} ", style=style)
            title.append(f"{node.id} {one_line(node.title, 46)}", style=style if node.status.value != "pending" else "")
            if node.outcome == "issues":
                title.append(" (open points)", style="yellow")
            nodes_table.add_row(title, node.assigned_agent or plan.assignments.get(node.id, ""))
        right = nodes_table
    else:
        phases_table = Table(box=None, show_header=True, header_style="bold", pad_edge=False, expand=True)
        phases_table.add_column("PHASEN")
        for index, (phase, detail) in enumerate(state.phases[-10:]):
            last = index == len(state.phases[-10:]) - 1
            phases_table.add_row(Text(f"{'●' if last else '✓'} {phase} {one_line(detail, 60)}", style="cyan" if last else "green"))
        right = phases_table
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(agents_table, right)
    parts.append(grid)

    if srt is not None:
        totals = srt.tracker.totals()
        limits = srt.budget.limits
        stats = Text()
        if plan is not None and srt.mode.value == "run" and plan.graph.nodes:
            done, total = plan.graph.progress()
            counts = plan.graph.counts()
            stats.append(f"Status:  [{bar(done / total if total else 0)}] {done / total:.0%}   " if total else "")
            stats.append(f"Active: {counts['running']}  Completed: {counts['done']}  Waiting: {counts['pending']}  Failed: {counts['failed']}\n")
        stats.append(f"Tokens:  Input {fmt_int(totals.input_tokens)}   Output {fmt_int(totals.output_tokens)}   Total {fmt_int(totals.total_tokens)}", style="bold")
        if limits.max_tokens:
            stats.append(f" / {fmt_int(limits.max_tokens)}", style="dim")
        stats.append(f"   Cost {fmt_cost(totals.cost_usd)}", style="bold")
        if limits.max_cost_usd:
            stats.append(f" / {fmt_cost(limits.max_cost_usd)}", style="dim")
        ratio = srt.tracker.max_context_ratio()
        stats.append(f"\nContext: {bar(ratio)} {ratio:.0%}   LLM calls: {totals.calls} (active {state.llm_calls_running})")
        stats.append(f"\nSystem:  {app.resources.sample().render()}", style="dim")
        parts.append(stats)

    feed_height = max(4, min(app.config.settings.ui.max_event_lines, height - 24))
    if view == "messages":
        title = "AGENT COMMUNICATION"
        items = list(state.messages)[-max(2, feed_height // 3):]
        body = Text()
        for ts, style, text in items:
            body.append(datetime.fromtimestamp(ts).strftime("%H:%M:%S "), style="dim")
            body.append(one_line(text.split("\n", 1)[0], 160) + "\n", style=style or "bold")
            if "\n" in text:
                body.append("   " + one_line(text.split("\n", 1)[1], 300) + "\n")
    else:
        title = "EVENTS" + (" (debug)" if state.debug else "")
        body = Text()
        for ts, style, text in list(state.feed)[-feed_height:]:
            body.append(datetime.fromtimestamp(ts).strftime("%H:%M:%S "), style="dim")
            body.append(one_line(text, 170) + "\n", style=style)
    parts.append(Panel(body, title=title, title_align="left", box=box.ROUNDED, padding=(0, 1)))
    footer = Text("[P] Pause  [S] Stop  [D] Debug  [M] Messages/events  [Ctrl+C] Pause", style="dim")
    parts.append(footer)
    return Group(*parts)
