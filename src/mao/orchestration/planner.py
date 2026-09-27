"""PLAN mode: triage -> parallel investigation -> draft -> critique rounds -> judge -> final plan."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from mao.agents.agent import Agent, AgentTask
from mao.core.errors import ConfigError, MaoError
from mao.core.events import PhaseEvent
from mao.core.text import one_line
from mao.orchestration import prompts
from mao.orchestration.plan import CritiqueRecord, PlanDocument
from mao.orchestration.taskgraph import NodeKind, TaskNode

if TYPE_CHECKING:
    from mao.orchestration.session import SessionRuntime


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_critiques(records: list[CritiqueRecord]) -> str:
    lines = []
    for record in records:
        lines.append(f"### {record.reviewer}: {record.verdict} (Score {record.score if record.score is not None else '-'})")
        for issue in record.issues:
            lines.append(
                f"- [{issue.get('severity', '?')}] {issue.get('step') or 'general'}: {issue.get('issue', '')}"
                f" -> {issue.get('suggestion', '')} (evidence: {issue.get('evidence', '-')})"
            )
        for missing in record.missing_steps:
            lines.append(f"- missing step: {missing}")
    return "\n".join(lines)


class PlanningPipeline:
    def __init__(self, srt: SessionRuntime) -> None:
        self.srt = srt

    # ------------------------------------------------------------------ helpers

    def _phase(self, phase: str, detail: str = "") -> None:
        self.srt.bus.publish(PhaseEvent(phase=phase, detail=detail))

    def _provider(self, agent: Agent) -> str:
        return (agent.model_ref or "").split("/", 1)[0]

    def lead_planner(self) -> Agent:
        manager = self.srt.manager
        planner = manager.find("planner")
        if planner is None:
            candidates = [a for a in manager.available() if "planning" in a.capabilities]
            planner = candidates[0] if candidates else None
        if planner is None and manager.orchestrator is not None and manager.orchestrator.available:
            planner = manager.orchestrator
        if planner is None:
            raise MaoError(
                "No agent available for planning. Check /providers (API keys) and /agents, or start a local model."
            )
        return planner

    def _triage_agent(self, planner: Agent) -> Agent:
        orchestrator = self.srt.manager.orchestrator
        return orchestrator if orchestrator is not None and orchestrator.available else planner

    def pick_agent(self, role: str, used: set[str]) -> Agent | None:
        candidates = [a for a in self.srt.manager.available() if a.name not in used]
        exact = [a for a in candidates if role.lower() in (a.role.name.lower(), a.config_name.lower())]
        if exact:
            return min(exact, key=lambda a: a.stats.tasks)
        try:
            capabilities = set(self.srt.roles.get(role).capabilities)
        except ConfigError:
            capabilities = {"analysis"}
        ranked = sorted(candidates, key=lambda a: -len(capabilities & set(a.capabilities)))
        if ranked and capabilities & set(ranked[0].capabilities):
            return ranked[0]
        return None

    # ------------------------------------------------------------------ pipeline

    async def create_plan(self, task: str) -> PlanDocument:
        srt = self.srt
        settings = srt.settings
        if not srt.manager.available():
            raise MaoError(
                "No agents available: no model can be reached. Add API keys (/providers key add <provider>) "
                "or start a local model server (e.g. Ollama) and check /models."
            )
        plan = PlanDocument(task=task)
        plan.warnings.extend(srt.agent_warnings)
        self._phase("Workspace-Analyse", str(srt.workspace))
        await srt.ensure_profile()
        team = prompts.team_overview(srt.manager.available())
        planner = self.lead_planner()
        triage_agent = self._triage_agent(planner)

        self._phase("Triage", f"by {triage_agent.name}")
        max_investigations = settings.orchestration.investigation_max_agents
        triage = await srt.runtime.run(
            triage_agent,
            AgentTask(
                title="Understand the task (triage)",
                instructions=prompts.triage_instructions(task, team, max_investigations),
                purpose="plan.triage",
                output="json",
                output_schema=prompts.TRIAGE_SCHEMA,
                readonly=True,
                max_steps=8,
                tool_groups=["filesystem", "git", "collaboration"],
                shared_context_query=task,
            ),
        )
        triage_data = triage.data if isinstance(triage.data, dict) else {}
        if triage.status == "failed":
            plan.warnings.append(f"Triage failed: {triage.error}")
        plan.understanding = str(triage_data.get("understanding") or "").strip()
        try:
            complexity = max(1, min(5, int(triage_data.get("complexity") or 3)))
        except (TypeError, ValueError):
            complexity = 3
        plan.open_questions = [str(q) for q in triage_data.get("clarifications") or [] if q]
        if plan.understanding:
            srt.board.add(triage_agent.name, "finding", "Task understanding", plan.understanding, importance=4)

        investigations = [i for i in triage_data.get("investigations") or [] if isinstance(i, dict)]
        if complexity > 1 and investigations and max_investigations > 0:
            await self._investigate(task, plan.understanding, investigations[:max_investigations])

        self._phase("Draft plan", f"by {planner.name}")
        draft = await srt.runtime.run(
            planner,
            AgentTask(
                title="Create the plan",
                instructions=prompts.plan_instructions(task, plan.understanding, team),
                purpose="plan.draft",
                output="json",
                output_schema=prompts.PLAN_SCHEMA,
                readonly=True,
                max_steps=12,
                shared_context_query=task,
            ),
        )
        if not (isinstance(draft.data, dict) and plan.apply_agent_data(draft.data)):
            raise MaoError(f"The planner could not produce a valid plan: {draft.error or one_line(draft.text, 300)}")
        srt.messages.post(planner.name, ["*"], "proposal", f"Draft plan: {plan.title} ({len(plan.graph.nodes)} steps)\n{plan.summary}", topic="plan")

        if settings.orchestration.debate_enabled and complexity >= 2:
            await self._critique_rounds(plan, planner)
        self.finalize(plan)
        return plan

    async def revise_with_feedback(self, plan: PlanDocument, feedback: str) -> PlanDocument:
        srt = self.srt
        planner = self.lead_planner()
        plan.feedback_history.append(feedback)
        self._phase("Plan revision", "user feedback")
        result = await srt.runtime.run(
            planner,
            AgentTask(
                title="Revise the plan after user feedback",
                instructions=prompts.revision_instructions(plan.task, plan.to_agent_json(), "", feedback),
                purpose="plan.feedback",
                output="json",
                output_schema=prompts.PLAN_SCHEMA,
                readonly=True,
                max_steps=8,
            ),
        )
        if not (isinstance(result.data, dict) and plan.apply_agent_data(result.data)):
            raise MaoError(f"Revision failed: {result.error or one_line(result.text, 300)}")
        plan.version += 1
        plan.approved = False
        plan.consensus = False
        if srt.settings.orchestration.debate_enabled:
            await self._critique_rounds(plan, planner, feedback)
        self.finalize(plan)
        return plan

    async def _investigate(self, task: str, understanding: str, investigations: list[dict[str, Any]]) -> None:
        srt = self.srt
        used: set[str] = set()
        jobs: list[tuple[Agent, str, AgentTask]] = []
        for item in investigations:
            role = str(item.get("role") or "architect")
            agent = self.pick_agent(role, used)
            if agent is None:
                continue
            used.add(agent.name)
            focus = str(item.get("focus") or "General analysis")
            questions = [str(q) for q in item.get("questions") or []]
            jobs.append(
                (
                    agent,
                    focus,
                    AgentTask(
                        title=f"Investigation: {one_line(focus, 60)}",
                        instructions=prompts.investigation_instructions(task, understanding, focus, questions),
                        purpose="plan.investigate",
                        output="json",
                        output_schema=prompts.INVESTIGATION_SCHEMA,
                        readonly=True,
                        max_steps=min(agent.max_steps, 15),
                        shared_context_query=focus,
                    ),
                )
            )
        if not jobs:
            return
        self._phase("Investigation", ", ".join(f"{a.name}: {one_line(f, 40)}" for a, f, _ in jobs))
        results = await asyncio.gather(*(srt.runtime.run(agent, job) for agent, _, job in jobs))
        for (agent, focus, _job), result in zip(jobs, results, strict=True):
            data = result.data if isinstance(result.data, dict) else {}
            lines = [str(data.get("summary") or result.text or result.error or "(no result)")]
            files: list[str] = []
            for finding in data.get("findings") or []:
                if isinstance(finding, dict):
                    lines.append(f"- [{finding.get('severity', 'info')}] {finding.get('title', '')}: {finding.get('detail', '')}")
                    files.extend(str(f) for f in finding.get("files") or [])
            for risk in data.get("risks") or []:
                lines.append(f"- Risk: {risk}")
            for recommendation in data.get("recommendations") or []:
                lines.append(f"- Recommendation: {recommendation}")
            srt.board.add(
                agent.name,
                "finding",
                f"Investigation ({agent.role.name}): {one_line(focus, 80)}",
                "\n".join(lines),
                files=files[:20],
                importance=4,
            )

    async def _critique_rounds(self, plan: PlanDocument, planner: Agent, feedback: str | None = None) -> None:
        srt = self.srt
        settings = srt.settings
        critics = srt.assigner.pick_reviewers(
            srt.manager.available(),
            capabilities=["critique", "review", "security", "architecture"],
            count=settings.orchestration.max_critics,
            exclude={planner.name},
            author_provider=self._provider(planner),
        )
        if not critics:
            plan.warnings.append("No critics available - the plan was not cross-checked")
            return
        max_rounds = settings.limits.max_rounds
        for round_number in range(1, max_rounds + 1):
            self._phase("Plan critique", f"round {round_number}/{max_rounds}: {', '.join(c.name for c in critics)}")
            plan_json = plan.to_agent_json()
            results = await asyncio.gather(
                *(
                    srt.runtime.run(
                        critic,
                        AgentTask(
                            title=f"Check the plan (round {round_number})",
                            instructions=prompts.critique_instructions(plan.task, plan_json, round_number),
                            purpose="plan.critique",
                            output="json",
                            output_schema=prompts.CRITIQUE_SCHEMA,
                            readonly=True,
                            max_steps=8,
                            shared_context_query=plan.task,
                        ),
                    )
                    for critic in critics
                )
            )
            records: list[CritiqueRecord] = []
            for critic, result in zip(critics, results, strict=True):
                data = result.data if isinstance(result.data, dict) else None
                if not data:
                    continue
                record = CritiqueRecord(
                    reviewer=critic.name,
                    round=round_number,
                    verdict=str(data.get("verdict") or "revise").lower(),
                    score=_as_float(data.get("score")),
                    strengths=[str(s) for s in data.get("strengths") or []],
                    issues=[i for i in data.get("issues") or [] if isinstance(i, dict)],
                    missing_steps=[str(m) for m in data.get("missing_steps") or []],
                )
                records.append(record)
                plan.critiques.append(record)
                srt.messages.post(critic.name, [planner.name], "critique", render_critiques([record]), topic="plan")
            plan.rounds = round_number
            if not records:
                plan.warnings.append(f"critique round {round_number} returned nothing usable")
                return
            total_weight = sum(srt.debate.weight(r.reviewer) for r in records)
            approve_weight = sum(srt.debate.weight(r.reviewer) for r in records if r.verdict == "approve")
            critical = any(str(i.get("severity", "")).lower() == "critical" for r in records for i in r.issues)
            ratio = approve_weight / total_weight if total_weight else 0.0
            if ratio >= settings.orchestration.consensus_threshold and not critical:
                plan.consensus = True
                self._phase("Plan critique", f"consensus reached ({ratio:.0%} approval)")
                srt.board.add("orchestrator", "decision", "Plan approved by team consensus", f"{plan.title}: {ratio:.0%} approval in round {round_number}", importance=4)
                return
            critiques_text = render_critiques(records)
            if round_number == max_rounds:
                await self._judge(plan, plan_json, critiques_text, planner)
                return
            self._phase("Plan revision", f"round {round_number} ({ratio:.0%} approval, critical={critical})")
            revision = await srt.runtime.run(
                planner,
                AgentTask(
                    title=f"Revise the plan (round {round_number})",
                    instructions=prompts.revision_instructions(plan.task, plan_json, critiques_text, feedback),
                    purpose="plan.revise",
                    output="json",
                    output_schema=prompts.PLAN_SCHEMA,
                    readonly=True,
                    max_steps=6,
                ),
            )
            if not (isinstance(revision.data, dict) and plan.apply_agent_data(revision.data)):
                plan.warnings.append(f"Revision in round {round_number} failed - the previous state is kept")
                return
            plan.version += 1
            changes = revision.data.get("changes") if isinstance(revision.data, dict) else None
            if changes:
                srt.messages.post(planner.name, [c.name for c in critics], "proposal", "Revised:\n" + "\n".join(f"- {c}" for c in changes), topic="plan")

    async def _judge(self, plan: PlanDocument, plan_json: str, critiques_text: str, planner: Agent) -> None:
        srt = self.srt
        orchestrator = srt.manager.orchestrator
        judge = orchestrator if orchestrator is not None and orchestrator.available else planner
        self._phase("Judge decision", f"no consensus after {plan.rounds} rounds - {judge.name} decides")
        result = await srt.runtime.run(
            judge,
            AgentTask(
                title="Decide the final plan",
                instructions=prompts.judge_plan_instructions(plan.task, plan_json, critiques_text),
                purpose="plan.judge",
                output="json",
                output_schema=prompts.PLAN_SCHEMA,
                readonly=True,
                max_steps=10,
            ),
        )
        if isinstance(result.data, dict) and plan.apply_agent_data(result.data):
            plan.version += 1
            rationale = str(result.data.get("decision_rationale") or "")
            srt.board.add(judge.name, "decision", "Plan settled by the judge's decision", rationale or plan.summary, importance=5)
        else:
            plan.warnings.append("The judge could not deliver a final plan - the last state is used (no consensus)")

    # ------------------------------------------------------------------ finalization

    def add_quality_gates(self, plan: PlanDocument) -> None:
        graph = plan.graph
        changers = [
            n for n in graph.nodes if n.kind is NodeKind.IMPLEMENTATION or (n.modifies_files and n.kind not in (NodeKind.TEST, NodeKind.DOCUMENTATION))
        ]
        if changers:
            change_ids = [n.id for n in changers]
            tests = [n for n in graph.nodes if n.kind is NodeKind.TEST]
            if not tests:
                tests = [
                    graph.add(
                        TaskNode(
                            id="qa_tests",
                            title="Run tests and fix failures",
                            description="Run the test suite; on failures, diagnose and fix in the test-fix cycle.",
                            kind=NodeKind.TEST,
                            depends_on=change_ids,
                            role="tester",
                            auto_added=True,
                        )
                    )
                ]
                plan.warnings.append("Quality gate added automatically: tests")
            if not any(n.kind in (NodeKind.REVIEW, NodeKind.SECURITY) for n in graph.nodes):
                graph.add(
                    TaskNode(
                        id="qa_review",
                        title="Code review of the changes",
                        description="Check all changes of this session (diff) for correctness, edge cases, security and maintainability.",
                        kind=NodeKind.REVIEW,
                        depends_on=[t.id for t in tests],
                        role="reviewer",
                        auto_added=True,
                    )
                )
                plan.warnings.append("Quality gate added automatically: review")
        synthesis = [n for n in graph.nodes if n.kind is NodeKind.SYNTHESIS]
        if synthesis:
            for node in synthesis:
                node.always_run = True
        else:
            sinks = [n.id for n in graph.nodes if not graph.dependents(n.id)]
            graph.add(
                TaskNode(
                    id="final_summary",
                    title="Write the final report",
                    description="Summarise the results of all steps honestly.",
                    kind=NodeKind.SYNTHESIS,
                    depends_on=sinks,
                    role="project_manager",
                    auto_added=True,
                    always_run=True,
                    complexity=1,
                )
            )

    def finalize(self, plan: PlanDocument) -> None:
        srt = self.srt
        plan.warnings.extend(plan.graph.validate_and_repair())
        if srt.settings.orchestration.auto_add_quality_gates:
            self.add_quality_gates(plan)
        available = srt.manager.available()
        plan.assignments = {}
        for node in plan.graph.topological_order():
            agent = srt.assigner.assign(node, available, busy=set())
            if agent is not None:
                plan.assignments[node.id] = agent.name
            else:
                reason = srt.assigner.explain_unassignable(node, available) or "no matching agent"
                plan.warnings.append(f"Step {node.id} ({node.title}): {reason}")
        plan.warnings = list(dict.fromkeys(plan.warnings))
        plan.estimate = srt.estimator.estimate(plan)
        srt.plan = plan
        srt.session.save_plan(plan)
        srt.board.add(
            "orchestrator",
            "plan",
            f"Plan v{plan.version}: {plan.title}",
            json.dumps(json.loads(plan.to_agent_json()), ensure_ascii=False, indent=1),
            importance=4,
        )
