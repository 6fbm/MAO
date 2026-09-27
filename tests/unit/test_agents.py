"""Agent runtime (real tool loop against the mock provider), manager, task graph and assignment."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from mao.agents.agent import AgentTask
from mao.agents.manager import AgentManager
from mao.agents.roles import RoleRegistry
from mao.agents.runtime import AgentRuntime, RuntimeServices
from mao.config.schema import (
    AgentConfig,
    AgentsFile,
    ModelConfig,
    PermissionsConfig,
    ProviderConfig,
    RateLimitConfig,
    Settings,
    ToolsConfig,
)
from mao.context.compaction import ConversationCompactor
from mao.core.errors import TransientProviderError
from mao.core.events import ContextEvent, EventBus
from mao.core.types import ChatMessage, CompletionRequest, CompletionResponse, MessageRole, RunMode, ToolCall
from mao.messaging.blackboard import Blackboard
from mao.messaging.bus import MessageBus
from mao.messaging.hub import CollaborationHub
from mao.messaging.router import ContextRouter
from mao.models.catalog import ModelCatalog
from mao.models.selector import ModelSelector
from mao.orchestration.assignment import AgentAssigner
from mao.orchestration.control import RunControl
from mao.orchestration.taskgraph import NodeKind, NodeStatus, TaskGraph, TaskNode, parse_kind
from mao.providers.base import create_provider
from mao.providers.gateway import LLMGateway
from mao.providers.keypool import KeyPool
from mao.providers.mock import MockProvider
from mao.providers.ratelimit import ProviderLimiter
from mao.security.approval import ApprovalGateway, allow_all
from mao.security.redaction import Redactor
from mao.security.sandbox import WorkspaceSandbox
from mao.tools.executor import ToolExecutor
from mao.tools.registry import build_default_registry
from mao.tokens.tracker import UsageTracker
from mao.workspace.changes import ChangeTracker


def tool_call(name: str, **args) -> CompletionResponse:  # type: ignore[no-untyped-def]
    return CompletionResponse(message=ChatMessage.assistant("", [ToolCall(id=f"call_{name}", name=name, arguments=args)]))


class RuntimeHarness:
    def __init__(self, workspace: Path, tmp: Path, responder, *, model: ModelConfig | None = None, agents: list[AgentConfig] | None = None, mode: RunMode = RunMode.RUN, max_retries: int = 1) -> None:  # type: ignore[no-untyped-def]
        self.settings = Settings()
        providers = {
            "mock": ProviderConfig(
                type="mock",
                requires_api_key=False,
                selectable=True,
                max_retries=max_retries,
                models={"scripted": model or ModelConfig(context_window=64_000, max_output_tokens=4_096, tier="strong", capabilities=["tools", "coding", "reasoning"])},
            )
        }
        http = httpx.AsyncClient()
        self.provider = create_provider("mock", providers["mock"], http)
        assert isinstance(self.provider, MockProvider)
        self.provider.set_responder(responder)
        self.catalog = ModelCatalog(providers, self.settings)
        self.bus = EventBus()
        self.events: list = []
        self.bus.subscribe(self.events.append)
        self.tracker = UsageTracker()
        control = RunControl()

        async def no_sleep(_s: float) -> None:
            return None

        gateway = LLMGateway(
            providers={"mock": self.provider},
            catalog=self.catalog,
            keypools={"mock": KeyPool("mock", [], requires_key=False)},
            limiters={"mock": ProviderLimiter(RateLimitConfig())},
            tracker=self.tracker,
            bus=self.bus,
            control=control,
            sleep=no_sleep,
        )
        selector = ModelSelector(self.catalog)
        self.manager = AgentManager(
            agents_config=AgentsFile(agents=agents or [AgentConfig(name="coder", role="coder"), AgentConfig(name="reviewer", role="reviewer")]),
            roles=RoleRegistry(),
            catalog=self.catalog,
            selector=selector,
            settings=self.settings,
            permissions=PermissionsConfig(),
            bus=self.bus,
        )
        self.warnings = self.manager.build()
        redactor = Redactor()
        sandbox = WorkspaceSandbox(workspace, protected_patterns=[".git"], sensitive_patterns=[".env"])
        self.board = Blackboard(redactor)
        self.messages = MessageBus(self.bus, redactor)
        self.services = RuntimeServices(
            gateway=gateway,
            executor=ToolExecutor(build_default_registry(), redactor=redactor),
            router=ContextRouter(self.board, self.messages),
            compactor=ConversationCompactor(gateway=gateway, catalog=self.catalog, selector=selector, config=self.settings.context, bus=self.bus),
            manager=self.manager,
            bus=self.bus,
            control=control,
            settings=self.settings,
            tools_config=ToolsConfig(),
            permissions_config=PermissionsConfig(),
            sandbox=sandbox,
            approval=ApprovalGateway(PermissionsConfig().approval, allow_all, self.bus),
            changes=ChangeTracker(sandbox, tmp / "backups"),
            collaboration=CollaborationHub(self.messages, self.board),
            mode=mode,
        )
        self.runtime = AgentRuntime(self.services)


async def test_tool_loop_reads_file_and_returns_json(workspace: Path, tmp_path: Path) -> None:
    def responder(request: CompletionRequest):  # type: ignore[no-untyped-def]
        last = request.messages[-1]
        if last.role is MessageRole.USER:
            return tool_call("read_file", path="src/app.py")
        assert last.role is MessageRole.TOOL and "return a + b" in last.content
        return json.dumps({"status": "done", "summary": "add() adds two numbers", "files_changed": []})

    h = RuntimeHarness(workspace, tmp_path, responder)
    coder = h.manager.get("coder")
    assert coder is not None and coder.model_ref == "mock/scripted"
    result = await h.runtime.run(coder, AgentTask(title="Analysis", instructions="What does add do?", output="json", output_schema="{}"))
    assert result.ok and result.data["summary"] == "add() adds two numbers"
    assert result.tool_calls == 1 and result.steps == 2
    assert h.tracker.totals().calls == 2
    system = h.provider.calls[0].system or ""
    assert '"coder"' in system and "RUN mode" in system and "English" in system
    assert {t.name for t in h.provider.calls[0].tools} >= {"read_file", "write_file", "run_command"}


async def test_json_repair_prompt(workspace: Path, tmp_path: Path) -> None:
    replies = iter(["I am done, all good.", '{"status": "done", "summary": "ok"}'])
    h = RuntimeHarness(workspace, tmp_path, lambda _r: next(replies))
    coder = h.manager.get("coder")
    assert coder is not None
    result = await h.runtime.run(coder, AgentTask(title="T", instructions="I", output="json"))
    assert result.ok and result.data == {"status": "done", "summary": "ok"}
    assert "valid JSON" in h.provider.calls[1].messages[-1].content


async def test_step_limit_returns_incomplete(workspace: Path, tmp_path: Path) -> None:
    def responder(request: CompletionRequest):  # type: ignore[no-untyped-def]
        if request.metadata.get("final"):
            return "Did not finish."
        return tool_call("list_directory", path=".")

    h = RuntimeHarness(workspace, tmp_path, responder)
    coder = h.manager.get("coder")
    assert coder is not None
    result = await h.runtime.run(coder, AgentTask(title="Endless", instructions="…", max_steps=3))
    assert result.status == "incomplete" and result.tool_calls == 3
    assert "Did not finish" in result.text


async def test_provider_failure_marks_task_failed(workspace: Path, tmp_path: Path) -> None:
    def responder(_request: CompletionRequest):  # type: ignore[no-untyped-def]
        raise TransientProviderError("overloaded", provider="mock")

    h = RuntimeHarness(workspace, tmp_path, responder, max_retries=0)
    coder = h.manager.get("coder")
    assert coder is not None
    coder.fallback_refs = []
    result = await h.runtime.run(coder, AgentTask(title="T", instructions="I"))
    assert result.status == "failed" and "overloaded" in (result.error or "")
    assert coder.stats.failures == 1


async def test_readonly_task_hides_write_tools(workspace: Path, tmp_path: Path) -> None:
    h = RuntimeHarness(workspace, tmp_path, lambda _r: "done")
    coder = h.manager.get("coder")
    assert coder is not None
    await h.runtime.run(coder, AgentTask(title="Read only", instructions="…", readonly=True))
    names = {t.name for t in h.provider.calls[0].tools}
    assert "read_file" in names and not names & {"write_file", "run_command", "run_tests", "delete_path"}
    assert "PLAN mode" in (h.provider.calls[0].system or "")


async def test_compaction_rebases_long_conversation(workspace: Path, tmp_path: Path) -> None:
    (workspace / "big.txt").write_text("x" * 9_000, encoding="utf-8")
    state = {"reads": 0}

    def responder(request: CompletionRequest):  # type: ignore[no-untyped-def]
        if request.metadata.get("purpose") == "compaction":
            return "- big.txt was read (x characters only)"
        if any("Progress so far" in m.content for m in request.messages if m.role is MessageRole.USER):
            return "done after compaction"
        state["reads"] += 1
        return tool_call("read_file", path="big.txt")

    model = ModelConfig(context_window=4_096, max_output_tokens=512, tier="strong", capabilities=["tools", "coding"])
    h = RuntimeHarness(workspace, tmp_path, responder, model=model)
    coder = h.manager.get("coder")
    assert coder is not None
    coder.max_output_tokens = 512
    result = await h.runtime.run(coder, AgentTask(title="Large file", instructions="Read big.txt", include_shared_context=False))
    assert result.ok and result.text == "done after compaction"
    assert any(isinstance(e, ContextEvent) for e in h.events)
    rebased = [c for c in h.provider.calls if c.metadata.get("purpose") != "compaction" and len(c.messages) == 1 and "Progress so far" in c.messages[0].content]
    assert rebased and "big.txt was read" in rebased[0].messages[0].content


def test_manager_counts_limits_and_offline(workspace: Path, tmp_path: Path) -> None:
    agents = [AgentConfig(name="coder", role="coder", count=3), AgentConfig(name="ghost", role="reviewer", model="openai/gpt-x"), AgentConfig(name="bad", role="nonexistent")]
    h = RuntimeHarness(workspace, tmp_path, lambda _r: "x", agents=agents)
    names = [a.name for a in h.manager.all()]
    assert names == ["coder-1", "coder-2", "coder-3", "ghost"]
    ghost = h.manager.get("ghost")
    assert ghost is not None and not ghost.available and "openai" in ghost.offline_reason
    assert any("nonexistent" in w for w in h.warnings)
    assert h.manager.find("coder") is not None and h.manager.find("coder").name.startswith("coder-")  # type: ignore[union-attr]
    assert h.manager.orchestrator is not None and h.manager.orchestrator.available
    h.settings.limits.max_agents = 2
    h.manager.build()
    assert len(h.manager.all()) == 2 and any("max_agents" in w for w in h.manager.warnings)


# ------------------------------------------------------------------ task graph


def test_taskgraph_repair_and_scheduling() -> None:
    graph = TaskGraph(
        nodes=[
            TaskNode(id="s1", title="A"),
            TaskNode(id="s2", title="B", depends_on=["s1", "zz", "s2"]),
            TaskNode(id="s3", title="C", depends_on=["s4"]),
            TaskNode(id="s4", title="D", depends_on=["s3"]),
            TaskNode(id="s1", title="dup"),
        ]
    )
    warnings = graph.validate_and_repair()
    assert any("zz" in w for w in warnings) and any("Cyclic" in w for w in warnings) and any("Duplicate" in w for w in warnings)
    assert len(set(graph.ids())) == 5
    assert not graph._cycle_members()
    ready = {n.id for n in graph.ready()}
    assert "s1" in ready and "s2" not in ready
    graph.get("s1").mark_finished(NodeStatus.FAILED)  # type: ignore[union-attr]
    skipped = graph.skip_blocked()
    assert [n.id for n in skipped] == ["s2"]
    assert graph.progress()[0] == 2
    order = [n.id for n in graph.topological_order()]
    assert order.index("s1") < order.index("s2")


def test_parse_kind_aliases() -> None:
    assert parse_kind("Implement") is NodeKind.IMPLEMENTATION
    assert parse_kind("docs") is NodeKind.DOCUMENTATION
    assert parse_kind("security-review") is NodeKind.SECURITY
    assert parse_kind("something") is NodeKind.ANALYSIS


def test_assigner_respects_permissions_and_roles(workspace: Path, tmp_path: Path) -> None:
    agents = [AgentConfig(name="reviewer", role="reviewer"), AgentConfig(name="coder", role="coder"), AgentConfig(name="researcher", role="researcher")]
    h = RuntimeHarness(workspace, tmp_path, lambda _r: "x", agents=agents)
    assigner = AgentAssigner()
    available = h.manager.available()
    impl = TaskNode(id="i", title="Fix", kind=NodeKind.IMPLEMENTATION, modifies_files=True)
    assert assigner.assign(impl, available, busy=set()).name == "coder"  # type: ignore[union-attr]
    assert assigner.assign(impl, available, busy={"coder"}) is None
    research = TaskNode(id="r", title="Search the docs", kind=NodeKind.RESEARCH)
    assert assigner.assign(research, available, busy=set()).name == "researcher"  # type: ignore[union-attr]
    only_readers = [a for a in available if a.name != "coder"]
    assert "Write files" in (assigner.explain_unassignable(impl, only_readers) or "")
    review = TaskNode(id="v", title="Review", kind=NodeKind.REVIEW, role="reviewer")
    assert assigner.assign(review, available, busy=set()).name == "reviewer"  # type: ignore[union-attr]


@pytest.mark.parametrize("count", [1, 5])
def test_many_instances_of_same_model(workspace: Path, tmp_path: Path, count: int) -> None:
    agents = [AgentConfig(name="gpt", role=role, model="mock/scripted") for role in ["architect"]]
    agents += [AgentConfig(name="worker", role="coder", model="mock/scripted", count=count)]
    h = RuntimeHarness(workspace, tmp_path, lambda _r: "x", agents=agents)
    assert all(a.model_ref == "mock/scripted" for a in h.manager.all())
    assert len(h.manager.all()) == 1 + count
