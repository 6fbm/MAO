"""UI: renderers and the interactive live-dashboard loop against an in-memory terminal."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from rich.console import Console, RenderableType

from mao.app import AppContext
from mao.cli.commands.base import preview_manager
from mao.cli.console import ConsoleUI
from mao.cli.dashboard import render_dashboard
from mao.cli.render import (
    agent_overview,
    agents_table,
    banner,
    cost_view,
    decision_view,
    plan_view,
    provider_overview,
    result_view,
    sessions_table,
    tokens_table,
)
from mao.core.events import TaskNodeEvent
from mao.demo import create_demo_workspace
from mao.paths import AppPaths
from mao.security.approval import ApprovalDecision, ApprovalRequest


@pytest.fixture
async def ui_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    paths = AppPaths(home=tmp_path / "home", user_data=tmp_path / "userdata")
    app = await AppContext.create(paths, demo=True, discover=False)
    workspace = create_demo_workspace(tmp_path / "ws")
    app.config.settings.workspace = str(workspace)
    ui = ConsoleUI(app, interactive=False)
    ui.interactive = True  # exercise the Live loop against an in-memory terminal
    ui.state.on_feed = None
    buffer = io.StringIO()
    ui.console = Console(file=buffer, force_terminal=True, width=150, height=50, color_system=None)

    async def allow(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.ALLOW

    app.set_handlers(allow, None)
    yield app, ui, buffer, workspace
    await app.aclose()


def render_text(renderable: RenderableType) -> str:
    console = Console(file=io.StringIO(), width=150, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


async def test_live_dashboard_plan_and_run(ui_app) -> None:  # type: ignore[no-untyped-def]
    app, ui, buffer, workspace = ui_app
    plan_task = asyncio.ensure_future(app.orchestrator.plan("Fix the error"))
    await asyncio.wait_for(ui.attach(plan_task), 120)
    plan = plan_task.result()
    app.orchestrator.approve()
    run_task = asyncio.ensure_future(app.orchestrator.execute())
    await asyncio.wait_for(ui.attach(run_task), 180)
    result = run_task.result()
    assert result.status == "completed"
    assert "return a + b" in (workspace / "calculator.py").read_text(encoding="utf-8")

    output = buffer.getvalue()
    assert "MULTI AI ORCHESTRATOR" in output and "ACTIVE AGENTS" in output and "TASKS" in output

    srt = app.orchestrator.active
    assert srt is not None
    snapshot = render_text(render_dashboard(ui.state, app))
    assert "Session" in snapshot and "Tokens:" in snapshot and "s2" in snapshot and "Context:" in snapshot
    for renderable in (
        plan_view(plan),
        result_view(result),
        tokens_table(srt.tracker),
        cost_view(srt.tracker, app.config.settings.limits),
        agents_table(srt.manager),
        sessions_table(app.store.list()),
        banner(True),
        agent_overview(preview_manager(app), app),
        provider_overview(app),
    ):
        assert render_text(renderable).strip()
    assert "TASK COMPLETED" in render_text(result_view(result))
    assert "Estimated API cost" in render_text(plan_view(plan))


async def test_dashboard_detaches_on_pause_and_resumes(ui_app) -> None:  # type: ignore[no-untyped-def]
    app, ui, buffer, _workspace = ui_app
    await app.orchestrator.plan("Fix the error")
    app.orchestrator.approve()
    paused: list[TaskNodeEvent] = []

    def on_event(event: object) -> None:
        if isinstance(event, TaskNodeEvent) and event.status == "running" and not paused:
            paused.append(event)
            app.orchestrator.pause()

    unsubscribe = app.bus.subscribe(on_event)
    task = asyncio.ensure_future(app.orchestrator.execute())
    await asyncio.wait_for(ui.attach(task), 60)
    unsubscribe()
    assert not task.done()
    assert app.orchestrator.active is not None and app.orchestrator.active.control.paused
    assert "PAUSED" in buffer.getvalue()

    app.orchestrator.resume()
    await asyncio.wait_for(ui.attach(task), 180)
    assert task.result().status == "completed"


async def test_debate_rendering(ui_app) -> None:  # type: ignore[no-untyped-def]
    app, _ui, _buffer, _workspace = ui_app
    decision = await app.orchestrator.debate("How should divide() handle division by zero?")
    text = render_text(decision_view(decision))
    assert "Decision" in text and "P1" in text
