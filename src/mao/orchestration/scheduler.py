"""RUN mode: executes the task graph with parallelism, assignment, retries and special node handlers."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from mao.agents.agent import Agent, AgentTask
from mao.core.errors import BudgetExceededError, OperationCancelled
from mao.core.events import PhaseEvent, TaskNodeEvent
from mao.core.text import one_line
from mao.orchestration import prompts
from mao.orchestration.plan import PlanDocument
from mao.orchestration.quality import NodeOutcome, outcome_from_result
from mao.orchestration.taskgraph import NodeKind, NodeStatus, TaskNode

if TYPE_CHECKING:
    from mao.orchestration.session import SessionRuntime

log = logging.getLogger("mao.scheduler")


class GraphScheduler:
    def __init__(self, srt: SessionRuntime) -> None:
        self.srt = srt

    def _node_event(self, node: TaskNode, detail: str = "") -> None:
        self.srt.bus.publish(
            TaskNodeEvent(node_id=node.id, title=node.title, status=node.status.value, agent=node.assigned_agent, outcome=node.outcome, detail=detail)
        )

    async def run(self, plan: PlanDocument) -> None:
        srt = self.srt
        graph = plan.graph
        graph.reset_interrupted()
        running: dict[asyncio.Task[NodeOutcome], TaskNode] = {}
        busy: set[str] = set()
        done, total = graph.progress()
        srt.bus.publish(PhaseEvent(phase="Execution", detail=f"{total} steps, {done} already done"))
        ram_warned = False
        try:
            while True:
                await srt.control.checkpoint()
                for node in graph.skip_blocked():
                    self._node_event(node, "skipped: dependency failed")
                running_ids = {n.id for n in running.values()}
                max_parallel = srt.settings.limits.max_parallel_agents
                for node in graph.ready():
                    if len(running) >= max_parallel:
                        break
                    if node.id in running_ids:
                        continue
                    # under memory pressure only one step at a time (never block when nothing runs)
                    if running and srt.app.resources.ram_pressure(srt.settings.limits.max_ram_percent):
                        if not ram_warned:
                            ram = srt.app.resources.sample().ram_percent or 0.0
                            srt.bus.log(f"RAM usage {ram:.0f} % - further parallel steps are held back", level="warning", source="resources")
                            ram_warned = True
                        break
                    agent = srt.assigner.assign(node, srt.manager.available(), busy)
                    if agent is None:
                        reason = srt.assigner.explain_unassignable(node, srt.manager.available())
                        if reason:
                            previous = f"{node.error} | " if node.error else ""
                            node.mark_finished(NodeStatus.FAILED, outcome="failed", error=previous + reason)
                            self._node_event(node, reason)
                            srt.save()
                        continue
                    node.mark_running(agent.name)
                    busy.add(agent.name)
                    self._node_event(node, f"attempt {node.attempts}")
                    task = asyncio.create_task(self._execute(node, agent, plan), name=f"node-{node.id}")
                    running[task] = node
                if not running:
                    break
                finished, _pending = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    node = running.pop(task)
                    if node.assigned_agent:
                        busy.discard(node.assigned_agent)
                    try:
                        outcome = task.result()
                    except (OperationCancelled, BudgetExceededError):
                        node.status = NodeStatus.PENDING
                        node.attempts = max(0, node.attempts - 1)
                        raise
                    except Exception as exc:  # noqa: BLE001 - one broken node must not stop the run
                        log.exception("node %s crashed", node.id)
                        outcome = NodeOutcome(status=NodeStatus.FAILED, outcome="failed", error=f"{type(exc).__name__}: {exc}", agent=node.assigned_agent)
                    self._apply(node, outcome)
        except (OperationCancelled, BudgetExceededError):
            for task in running:
                task.cancel()
            await asyncio.gather(*running.keys(), return_exceptions=True)
            for node in running.values():
                node.status = NodeStatus.PENDING
                node.attempts = max(0, node.attempts - 1)
                self._node_event(node, "interrupted - it is restarted when the session resumes")
            raise
        finally:
            srt.save()

    def _apply(self, node: TaskNode, outcome: NodeOutcome) -> None:
        srt = self.srt
        agent_name = outcome.agent or node.assigned_agent or "orchestrator"
        if outcome.status is NodeStatus.FAILED and node.attempts < srt.settings.limits.max_node_attempts and node.assigned_agent:
            node.excluded_agents.append(node.assigned_agent)
            node.status = NodeStatus.PENDING
            node.error = outcome.error
            self._node_event(node, f"failed ({one_line(outcome.error, 120)}) - retrying with a different agent")
            srt.save()
            return
        node.mark_finished(outcome.status, outcome=outcome.outcome, summary=outcome.summary, error=outcome.error)
        entry = srt.board.add(
            agent_name,
            "result",
            f"{node.id}: {node.title} ({outcome.status.value}/{outcome.outcome})",
            outcome.render(),
            files=outcome.files_changed,
            node_id=node.id,
            importance=4,
        )
        node.result_entry = entry.id
        srt.messages.post(agent_name, ["*"], "result", f"{node.title}: {outcome.summary or outcome.error or ''}", topic=node.id)
        self._node_event(node, one_line(outcome.summary or outcome.error, 160))
        srt.save()

    async def _execute(self, node: TaskNode, agent: Agent, plan: PlanDocument) -> NodeOutcome:
        srt = self.srt
        if node.kind is NodeKind.TEST:
            return await srt.quality.run_tests(node, agent)
        if node.kind in (NodeKind.REVIEW, NodeKind.SECURITY):
            return await srt.quality.run_review(node, agent)
        if node.kind is NodeKind.DECISION and srt.settings.orchestration.debate_enabled:
            decision = await srt.debate.debate(f"{node.title}\n{node.description}", context=f"Task: {plan.task}\nApproach: {plan.approach}", node_id=node.id)
            status = NodeStatus.DONE if decision.decision else NodeStatus.FAILED
            return NodeOutcome(
                status=status,
                outcome="success" if decision.decision else "failed",
                summary=f"Decision ({decision.method}): {one_line(decision.decision, 400)}",
                details=decision.rationale,
                issues=decision.open_risks,
                agent=agent.name,
                error=None if decision.decision else decision.rationale,
            )
        if node.kind is NodeKind.SYNTHESIS:
            return await self._synthesis(node, agent, plan)
        result = await srt.runtime.run(
            agent,
            AgentTask(
                title=node.title,
                instructions=prompts.node_instructions(
                    plan.task, plan.title, plan.approach, node.title, node.id, node.description, node.acceptance_criteria, node.modifies_files
                ),
                purpose=f"execute.{node.kind.value}",
                output="json",
                output_schema=prompts.NODE_RESULT_SCHEMA,
                node_id=node.id,
                depends_on=node.depends_on,
                shared_context_query=f"{node.title} {node.description}",
            ),
        )
        return outcome_from_result(result)

    async def _synthesis(self, node: TaskNode, agent: Agent, plan: PlanDocument) -> NodeOutcome:
        srt = self.srt
        results = "\n".join(
            f"- {n.id} {n.title}: {n.status.value}/{n.outcome or '-'} – {one_line(n.result_summary or n.error or '', 500)}"
            for n in plan.graph.nodes
            if n.id != node.id
        )
        changes = srt.changes.summary()
        change_text = "\n".join(f"{kind}: {', '.join(paths) or '-'}" for kind, paths in changes.items())
        tests = srt.test_summaries[-1].headline() if srt.test_summaries else "No automatic tests were run"
        decisions = "\n".join(f"- {e.title}: {e.summary}" for e in srt.board.entries("decision")) or "none"
        result = await srt.runtime.run(
            agent,
            AgentTask(
                title=node.title,
                instructions=prompts.synthesis_instructions(plan.task, plan.title, results, change_text, tests, decisions),
                purpose="execute.synthesis",
                output="json",
                output_schema=prompts.SYNTHESIS_SCHEMA,
                readonly=True,
                node_id=node.id,
                max_steps=6,
                include_shared_context=False,
            ),
        )
        if isinstance(result.data, dict):
            srt.final_synthesis = result.data
        outcome = outcome_from_result(result)
        if isinstance(result.data, dict) and result.data.get("summary"):
            outcome.summary = str(result.data["summary"])
            outcome.status = NodeStatus.DONE
            outcome.outcome = "success"
            outcome.error = None
        return outcome
