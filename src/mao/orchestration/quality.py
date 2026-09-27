"""Quality loops: test -> debug -> fix -> test, and review -> fix -> re-review."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mao.agents.agent import Agent, AgentState, AgentTask, AgentTaskResult
from mao.core.errors import PermissionDeniedError, ToolError
from mao.core.events import PhaseEvent
from mao.core.text import one_line
from mao.orchestration import prompts
from mao.orchestration.taskgraph import NodeKind, NodeStatus, TaskNode
from mao.security.permissions import Capability, granted_capabilities
from mao.tools.tests_tool import TestRunSummary, execute_tests

if TYPE_CHECKING:
    from mao.orchestration.session import SessionRuntime


class NodeOutcome(BaseModel):
    status: NodeStatus
    outcome: str = "success"  # success | issues | failed
    summary: str = ""
    details: str = ""
    files_changed: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    agent: str | None = None
    error: str | None = None

    def render(self) -> str:
        parts = [self.summary or self.error or "(no result)"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.files_changed:
            parts.append("Changed files: " + ", ".join(self.files_changed))
        if self.issues:
            parts.append("Open points:\n" + "\n".join(f"- {i}" for i in self.issues))
        if self.error and self.summary:
            parts.append(f"Error: {self.error}")
        return "\n".join(parts)


def outcome_from_result(result: AgentTaskResult) -> NodeOutcome:
    if result.status == "failed":
        return NodeOutcome(status=NodeStatus.FAILED, outcome="failed", error=result.error or "unknown error", agent=result.agent)
    data = result.data if isinstance(result.data, dict) else {}
    state = str(data.get("status") or "done").lower()
    summary = str(data.get("summary") or result.text or "").strip()
    issues = [str(i) for i in data.get("issues") or [] if i]
    files = [str(f) for f in data.get("files_changed") or [] if f]
    details = str(data.get("details") or "")
    if state in ("failed", "blocked"):
        return NodeOutcome(
            status=NodeStatus.FAILED, outcome="failed", summary=summary, details=details, files_changed=files,
            issues=issues, agent=result.agent, error=f"Agent reports '{state}': {one_line(summary, 300)}",
        )
    outcome = "success" if result.status == "ok" and not issues else "issues"
    return NodeOutcome(status=NodeStatus.DONE, outcome=outcome, summary=summary, details=details, files_changed=files, issues=issues, agent=result.agent)


class QualityLoops:
    def __init__(self, srt: SessionRuntime) -> None:
        self.srt = srt

    def _find(self, role: str, capability: Capability | None = None, exclude: set[str] | None = None) -> Agent | None:
        exclude = exclude or set()
        manager = self.srt.manager
        agent = manager.find(role)
        if agent is not None and agent.name not in exclude and (capability is None or capability in granted_capabilities(agent.permissions)):
            return agent
        try:
            wanted = set(self.srt.roles.get(role).capabilities)
        except Exception:  # noqa: BLE001
            wanted = set()
        candidates = [
            a
            for a in manager.available()
            if a.name not in exclude and (capability is None or capability in granted_capabilities(a.permissions))
        ]
        candidates.sort(key=lambda a: (-len(wanted & set(a.capabilities)), a.stats.tasks))
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------ tests

    async def run_tests(self, node: TaskNode, lead: Agent) -> NodeOutcome:
        srt = self.srt
        limits = srt.settings.limits
        coder = self._find("coder", Capability.WRITE)
        debugger = self._find("debugger") or coder
        ctx = srt.system_tool_context()
        last: TestRunSummary | None = None
        fixes = 0
        for iteration in range(limits.max_fix_iterations + 1):
            await srt.control.checkpoint()
            srt.bus.publish(PhaseEvent(phase="Tests", detail=f"{node.id} - run {iteration + 1}"))
            srt.manager.set_state(lead, AgentState.TOOL, f"Tests running (run {iteration + 1})")
            try:
                last = await execute_tests(ctx)
            except ToolError as exc:
                return await self._agent_verification(node, lead, str(exc))
            except PermissionDeniedError as exc:
                return NodeOutcome(status=NodeStatus.FAILED, outcome="failed", error=f"Running tests is not allowed: {exc}", agent=lead.name)
            finally:
                srt.manager.set_state(lead, AgentState.IDLE, "")
            srt.bus.log(last.headline(), source="tests")
            if last.success:
                suffix = f" (after {fixes} fix cycle(s))" if fixes else ""
                return NodeOutcome(status=NodeStatus.DONE, outcome="success", summary=last.headline() + suffix, details=f"$ {last.command}", agent=lead.name)
            if iteration >= limits.max_fix_iterations or coder is None:
                break
            diagnosis_text = ""
            if debugger is not None:
                diagnosis = await srt.runtime.run(
                    debugger,
                    AgentTask(
                        title=f"Analyse the test failures ({iteration + 1})",
                        instructions=prompts.diagnosis_instructions(last.command, last.output_tail, iteration + 1),
                        purpose="execute.debug",
                        output="json",
                        output_schema=prompts.DIAGNOSIS_SCHEMA,
                        readonly=True,
                        node_id=node.id,
                        max_steps=15,
                        shared_context_query="test failures " + last.output_tail[-400:],
                    ),
                )
                diagnosis_text = json.dumps(diagnosis.data, ensure_ascii=False, indent=1) if isinstance(diagnosis.data, dict) else diagnosis.text
                if diagnosis_text:
                    srt.messages.post(debugger.name, [coder.name], "info", f"Diagnosis of the test failures:\n{diagnosis_text}", topic=node.id)
            fix = await srt.runtime.run(
                coder,
                AgentTask(
                    title=f"Fix the test failures ({iteration + 1})",
                    instructions=prompts.fix_instructions(last.command, last.output_tail, diagnosis_text),
                    purpose="execute.fix",
                    output="json",
                    output_schema=prompts.NODE_RESULT_SCHEMA,
                    node_id=node.id,
                    max_steps=25,
                ),
            )
            fixes += 1
            if fix.status == "failed":
                srt.bus.log(f"Fix attempt {fixes} failed: {fix.error}", level="warning", source="tests")
        headline = last.headline() if last else "Tests not run"
        return NodeOutcome(
            status=NodeStatus.DONE,
            outcome="issues",
            summary=f"{headline} - after {fixes} fix cycle(s) did not resolve it completely",
            issues=[last.output_tail[-1_500:]] if last else [],
            agent=lead.name,
        )

    async def _agent_verification(self, node: TaskNode, lead: Agent, reason: str) -> NodeOutcome:
        srt = self.srt
        srt.bus.log(f"No automatic test run possible ({reason}) – {lead.name} checks manually", level="warning", source="tests")
        result = await srt.runtime.run(
            lead,
            AgentTask(
                title=node.title,
                instructions=(
                    f"{node.description}\n\nNo automatic test command was found ({reason}). Check this session's "
                    "changes by suitable means (syntax check, running the program or script, targeted checks). "
                    "Report status 'failed' if you find errors you are not allowed to fix."
                ),
                purpose="execute.test",
                output="json",
                output_schema=prompts.NODE_RESULT_SCHEMA,
                node_id=node.id,
                depends_on=node.depends_on,
                max_steps=20,
            ),
        )
        outcome = outcome_from_result(result)
        if outcome.status is NodeStatus.DONE and outcome.outcome == "success":
            outcome.outcome = "issues"
            outcome.issues.append("No automatic test suite - checked manually only")
        return outcome

    # ------------------------------------------------------------------ reviews

    async def run_review(self, node: TaskNode, lead: Agent) -> NodeOutcome:
        srt = self.srt
        limits = srt.settings.limits
        authors = {change.agent for change in srt.changes.changes()}
        capabilities = ["security"] if node.kind is NodeKind.SECURITY else ["review", "quality", "security"]
        count = 1 if node.kind is NodeKind.SECURITY else 2
        reviewers = srt.assigner.pick_reviewers(srt.manager.available(), capabilities=capabilities, count=count, exclude=authors)
        if not reviewers:
            reviewers = [lead]
        coder = self._find("coder", Capability.WRITE)
        previous: str | None = None
        blocking: list[str] = []
        summaries: list[str] = []
        for iteration in range(limits.max_review_iterations + 1):
            await srt.control.checkpoint()
            diff = srt.changes.diff(max_chars=40_000)
            if not diff.strip():
                return NodeOutcome(status=NodeStatus.DONE, outcome="success", summary="No file changes to check.", agent=lead.name)
            srt.bus.publish(PhaseEvent(phase="Review", detail=f"{node.id} - round {iteration + 1}: {', '.join(r.name for r in reviewers)}"))
            results = await asyncio.gather(
                *(
                    srt.runtime.run(
                        reviewer,
                        AgentTask(
                            title=f"{node.title} (round {iteration + 1})",
                            instructions=prompts.review_instructions(node.title, node.description, diff, previous),
                            purpose=f"execute.{node.kind.value}",
                            output="json",
                            output_schema=prompts.REVIEW_SCHEMA,
                            readonly=True,
                            node_id=node.id,
                            max_steps=12,
                        ),
                    )
                    for reviewer in reviewers
                )
            )
            blocking, summaries = [], []
            for reviewer, result in zip(reviewers, results, strict=True):
                data = result.data if isinstance(result.data, dict) else {}
                if not data:
                    summaries.append(f"{reviewer.name}: no usable review ({result.error or 'invalid format'})")
                    continue
                verdict = str(data.get("verdict") or "approve").lower()
                summaries.append(f"{reviewer.name}: {verdict} – {one_line(str(data.get('summary') or ''), 200)}")
                for issue in data.get("issues") or []:
                    if not isinstance(issue, dict):
                        continue
                    severity = str(issue.get("severity") or "minor").lower()
                    text = f"[{severity}] {issue.get('file') or '?'}:{issue.get('line') or '-'} {issue.get('issue', '')} -> {issue.get('suggestion', '')} ({reviewer.name})"
                    if verdict == "request_changes" and severity in ("critical", "major"):
                        blocking.append(text)
                lines = [f"Review ({verdict}): {data.get('summary') or ''}".strip()]
                for issue in data.get("issues") or []:
                    if isinstance(issue, dict):
                        lines.append(
                            f"- [{issue.get('severity', '?')}] {issue.get('file') or '?'}:{issue.get('line') or '-'} "
                            f"{issue.get('issue', '')} -> {issue.get('suggestion', '')}"
                        )
                srt.messages.post(reviewer.name, [coder.name] if coder else ["*"], "critique", "\n".join(lines), topic=node.id)
            if not blocking:
                return NodeOutcome(status=NodeStatus.DONE, outcome="success", summary=" | ".join(summaries), agent=lead.name)
            if iteration >= limits.max_review_iterations or coder is None:
                break
            previous = "\n".join(blocking)
            await srt.runtime.run(
                coder,
                AgentTask(
                    title=f"Review-Befunde beheben ({iteration + 1})",
                    instructions=prompts.review_fix_instructions(previous),
                    purpose="execute.fix",
                    output="json",
                    output_schema=prompts.NODE_RESULT_SCHEMA,
                    node_id=node.id,
                    max_steps=25,
                ),
            )
            if srt.test_summaries:
                try:
                    rerun = await execute_tests(srt.system_tool_context())
                    srt.bus.log(f"Regression test after the review fix: {rerun.headline()}", source="tests")
                except (ToolError, PermissionDeniedError) as exc:
                    srt.bus.log(f"Regression test not possible: {exc}", level="warning", source="tests")
        return NodeOutcome(
            status=NodeStatus.DONE,
            outcome="issues",
            summary=f"{len(blocking)} blocking review findings open | " + " | ".join(summaries),
            issues=blocking,
            agent=lead.name,
        )
