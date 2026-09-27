"""Writes every session event to categorized log files (secrets redacted)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from mao.core.events import (
    AgentMessageEvent,
    AgentStatusEvent,
    ApprovalEvent,
    BudgetEvent,
    ContextEvent,
    DebateEvent,
    DecisionEvent,
    Event,
    EventBus,
    LLMCallEvent,
    LogEvent,
    PhaseEvent,
    ProviderIssueEvent,
    SessionStatusEvent,
    TaskNodeEvent,
    ToolEvent,
)
from mao.core.text import one_line
from mao.security.redaction import RedactingFilter, Redactor
from mao.tokens.tracker import UsageTracker

LOG_FILES = ("orchestration", "agents", "tools", "errors")


def describe_event(event: Event) -> tuple[str, str, int]:
    """Return (log file, message, level) for an event."""
    if isinstance(event, PhaseEvent):
        return "orchestration", f"PHASE {event.phase} {event.detail}".strip(), logging.INFO
    if isinstance(event, SessionStatusEvent):
        return "orchestration", f"SESSION {event.status} {event.message}".strip(), logging.INFO
    if isinstance(event, LogEvent):
        level = {"debug": logging.DEBUG, "warning": logging.WARNING, "error": logging.ERROR}.get(event.level, logging.INFO)
        return ("errors" if level >= logging.ERROR else "orchestration"), f"[{event.source}] {event.message}", level
    if isinstance(event, TaskNodeEvent):
        level = logging.WARNING if event.status in ("failed", "skipped") else logging.INFO
        return "orchestration", f"NODE {event.node_id} {event.status} agent={event.agent or '-'} {event.title} {event.detail}".strip(), level
    if isinstance(event, DebateEvent):
        return "orchestration", f"DEBATE {event.debate_id} {event.stage} r{event.round}: {event.detail}", logging.INFO
    if isinstance(event, DecisionEvent):
        return "orchestration", f"DECISION {event.decision_id} ({event.method}, consensus={event.consensus}): {event.summary}", logging.INFO
    if isinstance(event, BudgetEvent):
        return "orchestration", f"BUDGET {event.metric} {event.level}: {event.used:,.2f}/{event.limit:,.2f}", logging.WARNING
    if isinstance(event, AgentStatusEvent):
        return "agents", f"{event.agent} -> {event.status} {event.activity}".strip(), logging.INFO
    if isinstance(event, LLMCallEvent):
        if event.stage == "start":
            return "agents", f"{event.agent} CALL {event.provider}/{event.model} purpose={event.purpose} attempt={event.attempt} est_in={event.input_tokens}", logging.DEBUG
        return (
            "agents",
            f"{event.agent} DONE {event.provider}/{event.model} in={event.input_tokens} out={event.output_tokens} cost=${event.cost_usd:.4f} {event.latency_s:.1f}s finish={event.finish_reason}",
            logging.INFO,
        )
    if isinstance(event, AgentMessageEvent):
        return "agents", f"MSG {event.sender} -> {','.join(event.recipients)} [{event.msg_kind}] {one_line(event.content, 600)}", logging.INFO
    if isinstance(event, ContextEvent):
        return "agents", f"{event.agent} CONTEXT {event.action} {event.before_tokens} -> {event.after_tokens} tokens", logging.INFO
    if isinstance(event, ToolEvent):
        if event.stage == "start":
            return "tools", f"{event.agent} {event.tool}({event.summary})", logging.INFO
        level = logging.INFO if event.ok else logging.WARNING
        return "tools", f"{event.agent} {event.tool} -> {'ok' if event.ok else 'ERROR'} ({event.duration_s}s): {event.summary}", level
    if isinstance(event, ApprovalEvent):
        return "tools", f"APPROVAL {event.agent} {event.action} {event.target} risk={event.risk} -> {event.decision}", logging.WARNING
    if isinstance(event, ProviderIssueEvent):
        return (
            "errors",
            f"PROVIDER {event.provider}/{event.model} ({event.agent}) {event.error_type}: {event.message} -> {event.action} "
            f"attempt {event.attempt}/{event.max_attempts} wait={event.wait_s}s next={event.next_model or '-'}",
            logging.ERROR if event.action in ("give_up", "fallback") else logging.WARNING,
        )
    return "orchestration", f"{event.kind}: {event.model_dump_json()}", logging.DEBUG


class SessionLogger:
    def __init__(self, session_dir: Path, session_id: str, bus: EventBus, redactor: Redactor, tracker: UsageTracker, *, level: str = "INFO") -> None:
        self.dir = session_dir
        self.bus = bus
        self.redactor = redactor
        self.tracker = tracker
        self._lock = threading.Lock()
        self._last_tokens_write = 0.0
        self._loggers: dict[str, logging.Logger] = {}
        self._handlers: list[tuple[logging.Logger, logging.Handler]] = []
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
        redacting = RedactingFilter(redactor)
        for name in LOG_FILES:
            logger = logging.getLogger(f"mao.session.{session_id}.{name}")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            handler = logging.FileHandler(self.dir / f"{name}.log", encoding="utf-8")
            handler.setFormatter(formatter)
            handler.addFilter(redacting)
            handler.setLevel(getattr(logging, level, logging.INFO) if name != "errors" else logging.WARNING)
            logger.addHandler(handler)
            self._loggers[name] = logger
            self._handlers.append((logger, handler))
        # application exceptions logged anywhere under "mao" also land in errors.log
        app_errors = logging.FileHandler(self.dir / "errors.log", encoding="utf-8")
        app_errors.setLevel(logging.ERROR)
        app_errors.setFormatter(formatter)
        app_errors.addFilter(redacting)
        root = logging.getLogger("mao")
        root.addHandler(app_errors)
        self._handlers.append((root, app_errors))
        self._events_file = open(self.dir / "events.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        self._unsubscribe = bus.subscribe(self.handle)

    def handle(self, event: Event) -> None:
        with self._lock:
            if self._events_file.closed:
                return
            self._events_file.write(self.redactor.redact(event.model_dump_json()) + "\n")
            self._events_file.flush()
        target, message, level = describe_event(event)
        self._loggers[target].log(level, message)
        if target != "errors" and level >= logging.WARNING and isinstance(event, (ToolEvent, TaskNodeEvent)):
            self._loggers["errors"].log(level, message)
        if isinstance(event, LLMCallEvent) and event.stage == "end" and time.monotonic() - self._last_tokens_write > 2:
            self.write_tokens()

    def write_tokens(self) -> None:
        self._last_tokens_write = time.monotonic()
        path = self.dir / "tokens.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.tracker.snapshot(), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def close(self) -> None:
        self._unsubscribe()
        self.write_tokens()
        with self._lock:
            self._events_file.close()
        for logger, handler in self._handlers:
            logger.removeHandler(handler)
            handler.close()
