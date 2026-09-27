"""End-to-end pipeline tests: real orchestration, tools, files and tests with scripted model answers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from mao.app import AppContext
from mao.core.errors import BudgetExceededError
from mao.core.events import AgentMessageEvent, LogEvent, TaskNodeEvent
from mao.demo import create_demo_workspace
from mao.orchestration.resources import ResourceSnapshot
from mao.orchestration.taskgraph import NodeKind, NodeStatus, TaskNode
from mao.paths import AppPaths
from mao.security.approval import ApprovalDecision, ApprovalRequest
from mao.sessions.store import SessionStatus


@pytest.fixture
async def demo(tmp_path: Path):  # type: ignore[no-untyped-def]
    paths = AppPaths(home=tmp_path / "home", user_data=tmp_path / "userdata")
    app = await AppContext.create(paths, demo=True, discover=False)
    workspace = create_demo_workspace(tmp_path / "workspace")
    app.config.settings.workspace = str(workspace)
    app.config.tools.tests.command = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.ALLOW

    app.set_handlers(approve, None)
    yield app, workspace, approvals
    await app.aclose()


async def test_plan_then_run_fixes_bug(demo) -> None:  # type: ignore[no-untyped-def]
    app, workspace, _approvals = demo
    messages: list[AgentMessageEvent] = []
    app.bus.subscribe(lambda e: messages.append(e) if isinstance(e, AgentMessageEvent) else None)

    plan = await app.orchestrator.plan("Analyse the project and fix all errors.")
    assert [n.kind for n in plan.graph.nodes][:3] == [NodeKind.ANALYSIS, NodeKind.IMPLEMENTATION, NodeKind.TEST]
    assert plan.estimate is not None and plan.estimate.total_tokens > 0
    assert plan.critiques, "critique round must have happened"
    assert plan.assignments["s2"].startswith("coder")
    assert "return a - b" in (workspace / "calculator.py").read_text(encoding="utf-8"), "PLAN must not change files"
    assert app.orchestrator.active is not None
    assert app.orchestrator.active.session.meta.status is SessionStatus.PLANNED

    app.orchestrator.approve()
    result = await app.orchestrator.execute()

    assert result.status == "completed", result
    assert "return a + b" in (workspace / "calculator.py").read_text(encoding="utf-8")
    assert result.files_modified == ["calculator.py"]
    assert result.tests is not None and result.tests["passed"] == 2 and result.tests["failed"] == 0
    assert result.bugs_fixed == 1
    assert result.total_tokens > 0 and len(result.agents_used) >= 5
    assert result.nodes_failed == [] and result.nodes_done == len(plan.graph.nodes)
    assert all(n.status is NodeStatus.DONE for n in plan.graph.nodes)

    session_dir = app.orchestrator.active.session.dir
    for name in ("orchestration.log", "agents.log", "tools.log", "errors.log", "tokens.json", "events.jsonl", "messages.jsonl", "blackboard.json", "plan.json", "result.json", "session.json", "changes.json"):
        assert (session_dir / name).exists(), name
    kinds = {json.loads(line)["kind"] for line in (session_dir / "messages.jsonl").read_text(encoding="utf-8").splitlines()}
    assert {"critique", "finding", "result", "info"} <= kinds
    assert any(m.sender.startswith("coder") and "tester" in m.recipients for m in messages)
    tokens = json.loads((session_dir / "tokens.json").read_text(encoding="utf-8"))
    assert tokens["totals"]["calls"] > 5 and tokens["by_agent"]

    restored = await app.orchestrator.rollback()
    assert restored == ["calculator.py"]
    assert "return a - b" in (workspace / "calculator.py").read_text(encoding="utf-8")


async def test_stop_and_resume_run(demo) -> None:  # type: ignore[no-untyped-def]
    app, workspace, _ = demo
    await app.orchestrator.plan("Fix the error.")
    app.orchestrator.approve()
    stopped = asyncio.Event()

    def on_event(event: object) -> None:
        if isinstance(event, TaskNodeEvent) and event.status == "running" and not stopped.is_set():
            stopped.set()
            app.orchestrator.stop("test abort")

    unsubscribe = app.bus.subscribe(on_event)
    first = await app.orchestrator.execute()
    unsubscribe()
    assert first.status == "stopped"
    plan = app.orchestrator.active.plan  # type: ignore[union-attr]
    assert any(n.status is NodeStatus.PENDING for n in plan.graph.nodes)

    second = await app.orchestrator.execute()
    assert second.status == "completed"
    assert "return a + b" in (workspace / "calculator.py").read_text(encoding="utf-8")


async def test_resume_session_from_disk(demo) -> None:  # type: ignore[no-untyped-def]
    app, workspace, _ = demo
    await app.orchestrator.plan("Fix the error.")
    app.orchestrator.approve()
    session_id = app.orchestrator.active.session.id  # type: ignore[union-attr]
    await app.orchestrator.close_active()

    loaded = await app.orchestrator.load_session(session_id)
    assert loaded.plan is not None and loaded.plan.approved
    result = await app.orchestrator.execute()
    assert result.status == "completed"
    assert app.store.load(session_id).meta.status is SessionStatus.COMPLETED


async def test_debate_produces_decision(demo) -> None:  # type: ignore[no-untyped-def]
    app, _workspace, _ = demo
    decision = await app.orchestrator.debate("How should divide() handle division by zero?")
    assert decision.decision and decision.proposals and decision.participants
    assert decision.method in ("consensus", "vote", "judge")
    session_dir = app.orchestrator.active.session.dir  # type: ignore[union-attr]
    stored = json.loads((session_dir / "decisions.json").read_text(encoding="utf-8"))
    assert stored[0]["decision"] == decision.decision
    kinds = {json.loads(line)["kind"] for line in (session_dir / "messages.jsonl").read_text(encoding="utf-8").splitlines()}
    assert {"proposal", "critique", "decision"} <= kinds


async def test_budget_limit_stops_planning(demo) -> None:  # type: ignore[no-untyped-def]
    app, _workspace, _ = demo
    app.config.settings.limits.max_tokens = 3_000
    app.config.settings.limits.on_limit = "stop"
    with pytest.raises(BudgetExceededError):
        await app.orchestrator.plan("Analyse everything.")
    assert app.orchestrator.active.session.meta.status is SessionStatus.STOPPED  # type: ignore[union-attr]


async def test_ram_pressure_defers_parallel_steps(demo) -> None:  # type: ignore[no-untyped-def]
    app, _workspace, _ = demo

    class FullMemory:
        def sample(self) -> ResourceSnapshot:
            return ResourceSnapshot(ram_percent=99.0, ts=1.0)

        def ram_pressure(self, threshold: float | None) -> bool:
            return threshold is not None

    app.resources = FullMemory()
    warnings: list[LogEvent] = []
    app.bus.subscribe(lambda e: warnings.append(e) if isinstance(e, LogEvent) and e.source == "resources" else None)
    plan = await app.orchestrator.plan("Fix the error.")
    plan.graph.add(TaskNode(id="s1b", title="Parallel analysis of the tests", kind=NodeKind.ANALYSIS))
    app.orchestrator.approve()
    result = await app.orchestrator.execute()
    assert result.status == "completed"
    assert warnings, "a parallel start must have been deferred"
    assert plan.graph.get("s1b").status is NodeStatus.DONE  # type: ignore[union-attr]


async def test_high_risk_command_needs_approval_and_denial_is_respected(demo, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app, workspace, _ = demo
    asked: list[ApprovalRequest] = []

    async def deny(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision.DENY

    app.set_handlers(deny, None)
    app.config.tools.tests.command = "del /q calculator.py"
    await app.orchestrator.plan("Fix the error.")
    app.orchestrator.approve()
    result = await app.orchestrator.execute()
    assert (workspace / "calculator.py").exists()
    assert any(r.action.target == "del /q calculator.py" for r in asked)
    assert any("s3" in f for f in result.nodes_failed)
