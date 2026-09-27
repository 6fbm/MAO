"""Tool interface and the execution context handed to tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import httpx
from pydantic import BaseModel, Field

from mao.config.schema import PermissionSet, PermissionsConfig, ToolsConfig
from mao.core.events import EventBus
from mao.core.types import RunMode, ToolSpec
from mao.security.approval import ApprovalGateway
from mao.security.permissions import Capability
from mao.security.sandbox import WorkspaceSandbox

if TYPE_CHECKING:
    from mao.tools.tests_tool import TestRunSummary
    from mao.tools.web import WebSearcher
    from mao.workspace.changes import ChangeTracker
    from mao.workspace.git import GitRepo


class ToolResult(BaseModel):
    ok: bool = True
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **data: Any) -> ToolResult:
        return cls(ok=True, content=content, data=data)

    @classmethod
    def failure(cls, content: str, **data: Any) -> ToolResult:
        return cls(ok=False, content=content, data=data)


class CollaborationPort(Protocol):
    def send_message(self, sender: str, recipients: list[str], kind: str, content: str, topic: str | None = None) -> str: ...

    def post_finding(
        self,
        author: str,
        title: str,
        content: str,
        *,
        kind: str = "finding",
        sources: list[str] | None = None,
        files: list[str] | None = None,
        importance: int = 3,
        node_id: str | None = None,
    ) -> str: ...

    def read_board(self, query: str | None, kind: str | None, limit: int) -> str: ...

    def record_sources(self, author: str, query: str, hits: list[dict[str, str]], node_id: str | None = None) -> None: ...

    async def consult(self, requester: str, target: str, question: str, node_id: str | None = None, depth: int = 0) -> str: ...


@dataclass
class ToolContext:
    agent_name: str
    agent_role: str
    permissions: PermissionSet
    mode: RunMode
    sandbox: WorkspaceSandbox
    approval: ApprovalGateway
    changes: ChangeTracker
    bus: EventBus
    tools_config: ToolsConfig
    permissions_config: PermissionsConfig
    http: httpx.AsyncClient | None = None
    collaboration: CollaborationPort | None = None
    git: GitRepo | None = None
    web: WebSearcher | None = None
    node_id: str | None = None
    consult_depth: int = 0
    on_test_result: Callable[[TestRunSummary], None] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def object_schema(properties: dict[str, dict[str, Any]], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    group: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    required: ClassVar[frozenset[Capability]] = frozenset({Capability.READ})
    # mutating tools are hidden in PLAN mode unless plan_allowed (then they check at runtime)
    mutating: ClassVar[bool] = False
    plan_allowed: ClassVar[bool] = True
    # read-only tools may run concurrently within one agent step
    parallel_safe: ClassVar[bool] = True

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def available_in(self, mode: RunMode) -> bool:
        return mode is RunMode.RUN or not self.mutating or self.plan_allowed


AsyncCallback = Callable[..., Awaitable[Any]]
