"""Agent identity, state and task data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from mao.agents.roles import RoleDefinition
from mao.config.schema import PermissionSet
from mao.core.types import Usage, new_id


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    TOOL = "tool"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"
    OFFLINE = "offline"


@dataclass
class AgentStats:
    tasks: int = 0
    failures: int = 0
    llm_calls: int = 0
    tool_calls: int = 0


@dataclass
class Agent:
    name: str
    config_name: str
    role: RoleDefinition
    model_ref: str | None
    fallback_refs: list[str]
    permissions: PermissionSet
    tool_groups: list[str]
    capabilities: list[str]
    system_prompt: str
    extra_instructions: str | None = None
    temperature: float | None = None
    max_steps: int = 30
    max_output_tokens: int | None = None
    offline_reason: str = ""
    internal: bool = False
    state: AgentState = AgentState.IDLE
    activity: str = ""
    busy: bool = False
    stats: AgentStats = field(default_factory=AgentStats)
    notes: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.model_ref is not None

    def remember(self, note: str, limit: int = 8) -> None:
        self.notes.append(note)
        del self.notes[:-limit]


class AgentTask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task_"))
    title: str
    instructions: str
    purpose: str = "task"
    node_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    output: Literal["text", "json"] = "text"
    output_schema: str | None = None
    tool_groups: list[str] | None = None
    max_steps: int | None = None
    include_shared_context: bool = True
    shared_context_query: str = ""
    readonly: bool = False
    consult_depth: int = 0


class AgentTaskResult(BaseModel):
    task_id: str
    agent: str
    status: Literal["ok", "incomplete", "failed"]
    text: str = ""
    data: Any = None
    steps: int = 0
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"
