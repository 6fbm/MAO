"""Orchestrator facade used by the CLI: plan, revise, approve, execute, pause/resume/stop, debate."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mao.core.errors import BudgetExceededError, MaoError, OperationCancelled, ToolError
from mao.core.events import SessionStatusEvent
from mao.core.text import one_line
from mao.core.types import RunMode
from mao.orchestration.debate import Decision
from mao.orchestration.plan import PlanDocument
from mao.orchestration.session import SessionRuntime
from mao.orchestration.taskgraph import NodeStatus
from mao.sessions.store import SessionResult, SessionStatus

if TYPE_CHECKING:
    from mao.app import AppContext

log = logging.getLogger("mao.orchestrator")


class Orchestrator:
    def __init__(self, app: AppContext) -> None:
        self.app = app
        self.active: SessionRuntime | None = None
        self.last_result: SessionResult | None = None

    # ------------------------------------------------------------------ session handling

    def workspace_path(self) -> Path:
        configured = self.app.config.settings.workspace
        if not configured:
            raise MaoError("No workspace set. Use /workspace <path>.")
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise MaoError(f"Workspace does not exist: {path}")
        return path.resolve()

    def _status(self, srt: SessionRuntime, status: SessionStatus, message: str = "", error: str | None = None) -> None:
        srt.session.set_status(status, error=error)
        self.app.bus.publish(SessionStatusEvent(status=status.value, message=message))

    async def close_active(self) -> None:
        if self.active is not None:
            await self.active.close()
            self.active = None

    async def _open_new(self, task: str, kind: str = "task") -> SessionRuntime:
        workspace = self.workspace_path()
        await self.close_active()
        session = self.app.store.create(task, str(workspace), kind=kind)
        srt = await SessionRuntime.open(
            self.app, session, workspace, approval_handler=self.app.approval_handler, limit_handler=self.app.limit_handler
        )
        self.active = srt
        return srt

    async def load_session(self, session_id: str) -> SessionRuntime:
        session = self.app.store.load(session_id)
        if not session.meta.workspace or not Path(session.meta.workspace).is_dir():
            raise MaoError(f"The workspace of the session no longer exists: {session.meta.workspace}")
        await self.close_active()
        srt = await SessionRuntime.open(
            self.app, session, Path(session.meta.workspace), approval_handler=self.app.approval_handler, limit_handler=self.app.limit_handler
        )
        if srt.plan is not None:
            srt.plan.graph.reset_interrupted()
        self.active = srt
        return srt

    def require_active(self) -> SessionRuntime:
        if self.active is None:
            raise MaoError("No active session. Start with /plan <task> or load one with /sessions load <id>.")
        return self.active

    # ------------------------------------------------------------------ PLAN

    async def plan(self, task: str) -> PlanDocument:
        srt = await self._open_new(task)
        return await self._planning(srt, lambda: srt.planner.create_plan(task))

    async def revise(self, feedback: str) -> PlanDocument:
        srt = self.require_active()
        if srt.plan is None:
            raise MaoError("No plan to revise.")
        plan = srt.plan
        return await self._planning(srt, lambda: srt.planner.revise_with_feedback(plan, feedback))

    async def _planning(self, srt: SessionRuntime, factory) -> PlanDocument:  # type: ignore[no-untyped-def]
        srt.set_mode(RunMode.PLAN)
        srt.reset_control()
        self._status(srt, SessionStatus.PLANNING, one_line(srt.session.meta.task, 120))
        try:
            plan = await factory()
        except (OperationCancelled, BudgetExceededError) as exc:
            self._status(srt, SessionStatus.STOPPED, str(exc), error=str(exc))
            raise
        except Exception as exc:
            log.exception("planning failed")
            self._status(srt, SessionStatus.FAILED, str(exc), error=str(exc))
            raise
        finally:
            srt.save()
        srt.plan = plan
        srt.session.save_plan(plan)
        self._status(srt, SessionStatus.PLANNED, plan.title)
        return plan

    def approve(self) -> None:
        srt = self.require_active()
        if srt.plan is None:
            raise MaoError("No plan available.")
        srt.plan.approved = True
        srt.session.meta.plan_approved = True
        srt.save()

    def reject(self) -> None:
        srt = self.require_active()
        if srt.plan is not None:
            srt.plan.approved = False
        srt.session.meta.plan_approved = False
        self._status(srt, SessionStatus.STOPPED, "Plan discarded", error="Plan discarded by the user")

    # ------------------------------------------------------------------ RUN

    async def execute(self) -> SessionResult:
        srt = self.require_active()
        plan = srt.plan
        if plan is None:
            raise MaoError("No plan available. Run /plan <task> first.")
        if not plan.approved and self.app.config.settings.orchestration.require_plan_approval:
            raise MaoError("The plan is not confirmed yet.")
        srt.set_mode(RunMode.RUN)
        srt.reset_control()
        self._status(srt, SessionStatus.RUNNING, plan.title)
        started = time.monotonic()
        status, error = SessionStatus.COMPLETED, None
        try:
            await self._prepare_git(srt)
            await srt.scheduler.run(plan)
            nodes = plan.graph.nodes
            if nodes and all(n.status in (NodeStatus.FAILED, NodeStatus.SKIPPED) for n in nodes):
                status, error = SessionStatus.FAILED, "All steps failed"
        except OperationCancelled as exc:
            status, error = SessionStatus.STOPPED, str(exc)
        except BudgetExceededError as exc:
            status, error = SessionStatus.STOPPED, str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("run failed")
            status, error = SessionStatus.FAILED, f"{type(exc).__name__}: {exc}"
        duration = srt.session.meta.duration_s + (time.monotonic() - started)
        srt.session.meta.duration_s = duration
        result = await self._build_result(srt, plan, status, error, duration)
        srt.session.meta.summary = one_line(result.summary, 300)
        srt.session.save_result(result)
        self._status(srt, status, error or "done", error=error)
        srt.save()
        self.last_result = result
        return result

    async def _prepare_git(self, srt: SessionRuntime) -> None:
        git_config = self.app.config.tools.git
        if srt.git is None:
            return
        meta = srt.session.meta
        try:
            if git_config.auto_checkpoint and not meta.git_checkpoint:
                meta.git_checkpoint = await srt.git.checkpoint(f"session_{meta.id}")
                srt.bus.log(f"Git checkpoint created: refs/mao/checkpoints/session_{meta.id} ({(meta.git_checkpoint or '')[:10]})", source="git")
            if git_config.auto_branch and not meta.git_branch:
                if await srt.git.is_clean():
                    name = f"{git_config.branch_prefix}session-{meta.id}"
                    await srt.git.create_branch(name)
                    meta.git_branch = name
                    srt.bus.log(f"Switched to a new branch: {name}", source="git")
                else:
                    srt.bus.log("Working tree not clean - no automatic branch (backups are active)", level="warning", source="git")
        except ToolError as exc:
            srt.bus.log(f"Git preparation failed: {exc}", level="warning", source="git")
        srt.session.save_meta()

    async def _build_result(self, srt: SessionRuntime, plan: PlanDocument, status: SessionStatus, error: str | None, duration: float) -> SessionResult:
        changes = srt.changes.summary()
        totals = srt.tracker.totals()
        synthesis = srt.final_synthesis or {}
        git_info: dict = {"repository": srt.git is not None}
        if srt.git is not None:
            try:
                git_info.update(
                    branch=await srt.git.current_branch(),
                    dirty=not await srt.git.is_clean(),
                    checkpoint=srt.session.meta.git_checkpoint,
                )
            except ToolError as exc:
                git_info["error"] = str(exc)
        summary = str(synthesis.get("summary") or "")
        if not summary:
            lines = [f"- {n.title}: {n.status.value} – {one_line(n.result_summary or n.error or '', 200)}" for n in plan.graph.nodes]
            summary = "\n".join(lines)
        try:
            bugs_fixed = int(synthesis.get("bugs_fixed") or 0)
        except (TypeError, ValueError):
            bugs_fixed = 0
        tests = srt.test_summaries[-1].model_dump(exclude={"output_tail"}) if srt.test_summaries else None
        return SessionResult(
            status=status.value,
            files_created=changes["created"],
            files_modified=changes["modified"],
            files_deleted=changes["deleted"],
            bugs_fixed=bugs_fixed,
            tests=tests,
            agents_used=[a for a in srt.tracker.by_agent() if a not in ("web_search",)],
            input_tokens=totals.input_tokens,
            output_tokens=totals.output_tokens,
            total_tokens=totals.total_tokens,
            cost_usd=totals.cost_usd,
            cost_estimated=not all(r.cost_reported for r in srt.tracker.records),
            duration_s=duration,
            nodes_done=sum(1 for n in plan.graph.nodes if n.status is NodeStatus.DONE),
            nodes_failed=[f"{n.id}: {n.title} ({one_line(n.error, 120)})" for n in plan.graph.nodes if n.status is NodeStatus.FAILED],
            nodes_skipped=[f"{n.id}: {n.title}" for n in plan.graph.nodes if n.status is NodeStatus.SKIPPED],
            git=git_info,
            summary=summary,
            highlights=[str(h) for h in synthesis.get("highlights") or []],
            remaining_issues=[str(i) for i in synthesis.get("remaining_issues") or []],
            recommendations=[str(r) for r in synthesis.get("recommendations") or []],
            error=error,
        )

    # ------------------------------------------------------------------ control

    def pause(self) -> None:
        srt = self.require_active()
        srt.control.pause()
        self._status(srt, SessionStatus.PAUSED, "paused")

    def resume(self) -> None:
        srt = self.require_active()
        srt.control.resume()
        self._status(srt, SessionStatus.RUNNING if srt.mode is RunMode.RUN else SessionStatus.PLANNING, "resumed")

    def stop(self, reason: str = "stopped by the user") -> None:
        srt = self.require_active()
        srt.control.stop(reason)

    # ------------------------------------------------------------------ extras

    async def debate(self, question: str) -> Decision:
        srt = await self._open_new(question, kind="debate")
        srt.set_mode(RunMode.PLAN)
        self._status(srt, SessionStatus.RUNNING, "debate")
        try:
            await srt.ensure_profile()
            decision = await srt.debate.debate(question)
        except (OperationCancelled, BudgetExceededError) as exc:
            self._status(srt, SessionStatus.STOPPED, str(exc), error=str(exc))
            raise
        srt.session.meta.summary = one_line(decision.decision, 300)
        self._status(srt, SessionStatus.COMPLETED, decision.method)
        srt.save()
        return decision

    async def rollback(self, paths: list[str] | None = None) -> list[str]:
        srt = self.require_active()
        return srt.changes.rollback(paths)

    async def commit(self, message: str) -> str:
        srt = self.require_active()
        if srt.git is None:
            raise MaoError("The workspace is not a git repository.")
        return await srt.git.commit(message)
