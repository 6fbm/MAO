"""Task commands: plan, run/go, replan, pause, resume, stop, debate, workspace."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text

from mao.cli.commands.base import CommandContext, command
from mao.cli.render import decision_view, plan_view
from mao.core.errors import MaoError
from mao.workspace.workspace import scan_workspace


@command("plan", summary="Analyse and plan a task in PLAN mode (no changes)", usage="/plan <task> | /plan show", group="Tasks")
async def plan_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    text = raw.strip()
    if not text:
        raise MaoError("Usage: /plan <task>")
    if text.lower() == "show":
        srt = ctx.app.orchestrator.require_active()
        if srt.plan is None:
            raise MaoError("The active session has no plan.")
        ctx.ui.print(plan_view(srt.plan))
        return
    ctx.repl.ensure_idle()
    plan = await ctx.repl.run_foreground(ctx.app.orchestrator.plan(text), "plan")
    if plan is not None:
        await ctx.repl.plan_decision(plan)


@command("replan", summary="Revise the current plan with feedback", usage="/replan <feedback>", group="Tasks")
async def replan_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    if not raw.strip():
        raise MaoError("Usage: /replan <feedback>")
    ctx.repl.ensure_idle()
    plan = await ctx.repl.run_foreground(ctx.app.orchestrator.revise(raw.strip()), "plan")
    if plan is not None:
        await ctx.repl.plan_decision(plan)


@command("run", summary="Run the plan in RUN mode (optionally a stored session)", usage="/run [session-id]", aliases=("go",), group="Tasks")
async def run_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    ctx.repl.ensure_idle()
    orchestrator = ctx.app.orchestrator
    if args:
        await orchestrator.load_session(args[0])
    srt = orchestrator.require_active()
    if srt.plan is None:
        raise MaoError("No plan available. Start with /plan <task>.")
    if not srt.plan.approved:
        ctx.ui.print(plan_view(srt.plan))
        if not await ctx.ui.confirm("Run this plan now?", default=True):
            return
        orchestrator.approve()
    await ctx.repl.execute_and_report()


@command("pause", summary="Pause the running task (also with key P or Ctrl+C)", group="Tasks")
async def pause_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    background = ctx.repl.background
    if background is None or background.done():
        raise MaoError("No task is running. (During a run: key P or Ctrl+C)")
    ctx.app.orchestrator.pause()
    ctx.ui.info("Paused.")


@command("resume", summary="Resume the paused task", group="Tasks")
async def resume_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    background = ctx.repl.background
    if background is None or background.done():
        raise MaoError("No paused task. Resume a stored session with: /run <session-id>")
    ctx.app.orchestrator.resume()
    await ctx.repl.continue_background()


@command("stop", summary="Stop the task in a controlled way (resumable with /run)", group="Tasks")
async def stop_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    background = ctx.repl.background
    if background is None or background.done():
        raise MaoError("No task is running.")
    if not await ctx.ui.confirm("Stop the task in a controlled way?", default=True):
        return
    ctx.app.orchestrator.stop()
    await ctx.repl.continue_background()


@command("debate", summary="Let the agents debate a question (proposals, critique, vote, judge)", usage="/debate <question>", group="Tasks")
async def debate_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    if not raw.strip():
        raise MaoError("Usage: /debate <question>")
    ctx.repl.ensure_idle()
    decision = await ctx.repl.run_foreground(ctx.app.orchestrator.debate(raw.strip()), "debate")
    if decision is not None:
        ctx.ui.print(decision_view(decision))


@command("workspace", summary="Show or set the workspace", usage="/workspace [path]", group="Workspace & Git")
async def workspace_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    if not raw.strip():
        current = app.config.settings.workspace
        ctx.ui.print(Text(f"Workspace: {current or '(not set)'}", style="bold"))
        if current and Path(current).is_dir():
            profile = await asyncio.to_thread(scan_workspace, Path(current), app.config.tools.filesystem.ignore_patterns, tree_entries=40)
            ctx.ui.print(profile.render())
        return
    ctx.repl.ensure_idle()
    path = Path(raw.strip().strip('"')).expanduser()
    if not path.is_dir():
        raise MaoError(f"Directory does not exist: {path}")
    resolved = str(path.resolve())
    app.config.settings.workspace = resolved
    app.config_manager.set_value("settings.workspace", resolved)
    await app.orchestrator.close_active()
    profile = await asyncio.to_thread(scan_workspace, path.resolve(), app.config.tools.filesystem.ignore_patterns, tree_entries=30)
    ctx.ui.success(f"Workspace set: {resolved}")
    languages = ", ".join(f"{k} ({v})" for k, v in profile.languages.items()) or "–"
    ctx.ui.print(Text(f"{profile.file_count} files · languages: {languages} · test command: {profile.test_command or 'not detected'}", style="dim"))
