"""Live tests with a REAL local model (no simulation).

Run explicitly:  pytest -m ollama
Requires a running Ollama server with at least one tool-capable model, e.g. ``ollama pull granite4.1:3b``.
Small models may fail individual steps; the tests check that real model calls, tool loops and the
pipeline work end to end, not the quality of the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mao.agents.agent import AgentTask
from mao.app import AppContext
from mao.config.schema import AgentConfig, AgentsFile
from mao.demo import create_demo_workspace
from mao.paths import AppPaths
from mao.security.approval import ApprovalDecision, ApprovalRequest

pytestmark = pytest.mark.ollama


@pytest.fixture
async def live_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    paths = AppPaths(home=tmp_path / "home", user_data=tmp_path / "userdata")
    app = await AppContext.create(paths, discover=True)
    catalog = app.hub.catalog
    tool_models = [e for e in catalog.entries(provider="ollama") if catalog.is_available(e) and e.config.tool_calling == "native"]
    if not tool_models:
        await app.aclose()
        pytest.skip("No tool-capable Ollama model available (e.g. 'ollama pull granite4.1:3b')")
    model = tool_models[0].ref
    settings = app.config.settings
    settings.privacy.local_only = True
    settings.orchestration.orchestrator_model = model
    settings.orchestration.investigation_max_agents = 1
    settings.limits.max_rounds = 1
    settings.limits.max_fix_iterations = 1
    settings.limits.max_review_iterations = 0
    app.config.agents = AgentsFile(
        agents=[
            AgentConfig(name="planner", role="planner", model=model),
            AgentConfig(name="coder", role="coder", model=model),
            AgentConfig(name="reviewer", role="reviewer", model=model),
            AgentConfig(name="pm", role="project_manager", model=model),
        ]
    )
    workspace = create_demo_workspace(tmp_path / "workspace")
    settings.workspace = str(workspace)
    app.config.tools.tests.command = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'

    async def allow(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.ALLOW

    app.set_handlers(allow, None)
    yield app, workspace, model
    await app.aclose()


async def test_real_model_uses_tools(live_app) -> None:  # type: ignore[no-untyped-def]
    app, _workspace, model = live_app
    srt = await app.orchestrator._open_new("Live-Test: Tool-Loop")
    agent = srt.manager.get("coder")
    assert agent is not None and agent.model_ref == model
    result = await srt.runtime.run(
        agent,
        AgentTask(
            title="Read a file",
            instructions="Use read_file to read calculator.py and report what add() returns.",
            output="json",
            output_schema='{"status": "done", "summary": "..."}',
            readonly=True,
            max_steps=6,
        ),
    )
    assert result.status != "failed", result.error
    assert result.tool_calls >= 1
    totals = srt.tracker.totals()
    assert totals.calls >= 2 and totals.input_tokens > 0 and not srt.tracker.any_estimated()


async def test_real_model_plan_and_run(live_app) -> None:  # type: ignore[no-untyped-def]
    app, _workspace, _model = live_app
    plan = await app.orchestrator.plan("In calculator.py returns add() a - b, but the tests expect the sum. Fix the bug.")
    assert plan.graph.nodes
    app.orchestrator.approve()
    result = await app.orchestrator.execute()
    assert result.status in ("completed", "failed")
    assert result.total_tokens > 0
