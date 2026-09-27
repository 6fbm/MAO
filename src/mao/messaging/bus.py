"""Structured, persisted agent-to-agent messages."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mao.core.events import AgentMessageEvent, EventBus
from mao.core.text import one_line
from mao.core.types import new_id
from mao.security.redaction import Redactor

BROADCAST = {"*", "all"}


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg_"))
    ts: float = Field(default_factory=time.time)
    sender: str
    recipients: list[str]
    kind: str = "info"
    content: str
    summary: str = ""
    topic: str | None = None
    reply_to: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_broadcast(self) -> bool:
        return any(r.lower() in BROADCAST for r in self.recipients)


def base_name(agent_name: str) -> str:
    """'coder-2' -> 'coder'."""
    head, _, tail = agent_name.rpartition("-")
    return head if head and tail.isdigit() else agent_name


class MessageBus:
    def __init__(self, events: EventBus, redactor: Redactor, store_path: Path | None = None) -> None:
        self.events = events
        self.redactor = redactor
        self.store_path = store_path
        self._messages: list[AgentMessage] = []
        self._read: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def post(
        self,
        sender: str,
        recipients: list[str],
        kind: str,
        content: str,
        *,
        topic: str | None = None,
        reply_to: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        clean = self.redactor.redact(content)
        message = AgentMessage(
            sender=sender,
            recipients=recipients or ["*"],
            kind=kind,
            content=clean,
            summary=one_line(summary or clean, 180),
            topic=topic,
            reply_to=reply_to,
            metadata=metadata or {},
        )
        with self._lock:
            self._messages.append(message)
            if self.store_path is not None:
                with open(self.store_path, "a", encoding="utf-8") as handle:
                    handle.write(message.model_dump_json() + "\n")
        self.events.publish(
            AgentMessageEvent(
                message_id=message.id,
                sender=sender,
                recipients=message.recipients,
                msg_kind=kind,
                summary=message.summary,
                content=clean,
                topic=topic,
            )
        )
        return message

    def _addressed_to(self, message: AgentMessage, agent_name: str, role: str) -> bool:
        if message.sender == agent_name:
            return False
        if message.is_broadcast:
            return True
        wanted = {agent_name.lower(), role.lower(), base_name(agent_name).lower()}
        return any(r.lower() in wanted for r in message.recipients)

    def inbox(self, agent_name: str, role: str, *, unread_only: bool = True, direct_only: bool = False) -> list[AgentMessage]:
        with self._lock:
            read = self._read.get(agent_name, set())
            return [
                m
                for m in self._messages
                if self._addressed_to(m, agent_name, role)
                and (not unread_only or m.id not in read)
                and (not direct_only or not m.is_broadcast)
            ]

    def mark_read(self, agent_name: str, messages: list[AgentMessage]) -> None:
        with self._lock:
            self._read.setdefault(agent_name, set()).update(m.id for m in messages)

    def all(self) -> list[AgentMessage]:
        with self._lock:
            return list(self._messages)

    def thread(self, topic: str) -> list[AgentMessage]:
        return [m for m in self.all() if m.topic == topic]

    def load(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        loaded = []
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    loaded.append(AgentMessage.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue
        with self._lock:
            self._messages = loaded
