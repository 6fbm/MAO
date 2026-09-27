"""Typed events and a synchronous in-process event bus.

The orchestration core never talks to the terminal directly. It publishes
events; the dashboard, the session logger and tests subscribe to them.
Handlers must be fast and must not raise – exceptions are logged and
swallowed so a broken subscriber can never break a run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger("mao.events")


class Event(BaseModel):
    kind: str
    ts: float = Field(default_factory=time.time)
    session_id: str | None = None


class LogEvent(Event):
    kind: Literal["log"] = "log"
    level: str = "info"
    source: str = "orchestrator"
    message: str


class PhaseEvent(Event):
    kind: Literal["phase"] = "phase"
    phase: str
    detail: str = ""


class SessionStatusEvent(Event):
    kind: Literal["session_status"] = "session_status"
    status: str
    message: str = ""


class AgentStatusEvent(Event):
    kind: Literal["agent_status"] = "agent_status"
    agent: str
    status: str
    activity: str = ""
    model: str | None = None


class LLMCallEvent(Event):
    kind: Literal["llm_call"] = "llm_call"
    stage: Literal["start", "end"]
    agent: str
    provider: str
    model: str
    purpose: str = ""
    attempt: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    finish_reason: str | None = None
    context_window: int | None = None
    estimated: bool = False


class ProviderIssueEvent(Event):
    kind: Literal["provider_issue"] = "provider_issue"
    agent: str
    provider: str
    model: str
    error_type: str
    message: str
    attempt: int = 1
    max_attempts: int = 1
    # retry | rotate_key | fallback | adjust_request | give_up | wait
    action: str
    wait_s: float = 0.0
    next_model: str | None = None


class ToolEvent(Event):
    kind: Literal["tool"] = "tool"
    stage: Literal["start", "end"]
    agent: str
    tool: str
    summary: str = ""
    ok: bool | None = None
    duration_s: float | None = None
    detail: str | None = None


class AgentMessageEvent(Event):
    kind: Literal["agent_message"] = "agent_message"
    message_id: str
    sender: str
    recipients: list[str]
    msg_kind: str
    summary: str
    content: str
    topic: str | None = None


class TaskNodeEvent(Event):
    kind: Literal["task_node"] = "task_node"
    node_id: str
    title: str
    status: str
    agent: str | None = None
    outcome: str | None = None
    detail: str = ""


class ApprovalEvent(Event):
    kind: Literal["approval"] = "approval"
    agent: str
    action: str
    target: str
    risk: str
    decision: str


class BudgetEvent(Event):
    kind: Literal["budget"] = "budget"
    metric: str
    used: float
    limit: float
    level: Literal["warning", "exceeded", "extended"]


class DebateEvent(Event):
    kind: Literal["debate"] = "debate"
    debate_id: str
    stage: str
    round: int = 0
    detail: str = ""


class DecisionEvent(Event):
    kind: Literal["decision"] = "decision"
    decision_id: str
    question: str
    summary: str
    method: str
    consensus: bool


class ContextEvent(Event):
    kind: Literal["context"] = "context"
    agent: str
    action: str
    before_tokens: int
    after_tokens: int


class ModeChangedEvent(Event):
    kind: Literal["mode_changed"] = "mode_changed"
    # global | agent | orchestrator | temporary | temporary_active | restored | reset
    scope: str
    target: str | None = None
    modes: list[str] = Field(default_factory=list)
    label: str = ""


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[tuple[EventHandler, frozenset[str] | None]] = []
        self._lock = threading.Lock()
        self.session_id: str | None = None

    def subscribe(self, handler: EventHandler, kinds: Iterable[str] | None = None) -> Callable[[], None]:
        entry = (handler, frozenset(kinds) if kinds is not None else None)
        with self._lock:
            self._subscribers.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._subscribers:
                    self._subscribers.remove(entry)

        return unsubscribe

    def publish(self, event: Event) -> None:
        if event.session_id is None:
            event.session_id = self.session_id
        with self._lock:
            subscribers = list(self._subscribers)
        for handler, kinds in subscribers:
            if kinds is not None and event.kind not in kinds:
                continue
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - subscribers must never break the core
                log.exception("event handler failed for %s", event.kind)

    def log(self, message: str, *, level: str = "info", source: str = "orchestrator") -> None:
        self.publish(LogEvent(message=message, level=level, source=source))
