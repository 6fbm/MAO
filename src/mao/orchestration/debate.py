"""Debate and consensus: proposals, critiques, revisions, weighted votes and an evidence-checking judge."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mao.agents.agent import Agent, AgentTask, AgentTaskResult
from mao.agents.manager import AgentManager
from mao.config.schema import Settings
from mao.core.events import DebateEvent, DecisionEvent, EventBus
from mao.core.text import one_line, truncate_end
from mao.core.types import new_id
from mao.messaging.blackboard import Blackboard
from mao.messaging.bus import MessageBus
from mao.orchestration import prompts
from mao.orchestration.assignment import AgentAssigner

DEBATE_TOOLS = ["filesystem", "git", "web", "collaboration"]


class Proposal(BaseModel):
    id: str
    author: str
    title: str = ""
    approach: str = ""
    rationale: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    withdrawn: bool = False
    endorse: str | None = None

    def render(self) -> str:
        parts = [f"[{self.id}] {self.title} (by {self.author}, confidence {self.confidence:.2f})", f"Approach: {self.approach}"]
        if self.rationale:
            parts.append(f"Rationale: {self.rationale}")
        if self.pros:
            parts.append("Pro: " + "; ".join(self.pros))
        if self.cons:
            parts.append("Contra: " + "; ".join(self.cons))
        if self.risks:
            parts.append("Risks: " + "; ".join(self.risks))
        if self.evidence:
            parts.append("Evidence: " + "; ".join(self.evidence))
        return "\n".join(parts)


class Assessment(BaseModel):
    critic: str
    proposal_id: str
    stance: str = "amend"
    severity: str = "none"
    arguments: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    suggested_changes: list[str] = Field(default_factory=list)


class Vote(BaseModel):
    voter: str
    proposal_id: str
    score: float
    rationale: str = ""


class Decision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dec_"))
    ts: float = Field(default_factory=time.time)
    question: str
    chosen_proposal: str | None = None
    decision: str = ""
    rationale: str = ""
    method: str = "judge"  # consensus | vote | judge | single | fallback
    consensus: bool = False
    confidence: float = 0.5
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    open_risks: list[str] = Field(default_factory=list)
    evidence_checked: list[str] = Field(default_factory=list)
    rounds: int = 0
    participants: list[str] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    vote_totals: dict[str, float] = Field(default_factory=dict)
    node_id: str | None = None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if value:
        return [str(value)]
    return []


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DebateEngine:
    def __init__(
        self,
        *,
        runtime: Any,  # AgentRuntime (typed loosely to avoid an import cycle)
        manager: AgentManager,
        assigner: AgentAssigner,
        board: Blackboard,
        messages: MessageBus,
        bus: EventBus,
        settings: Settings,
        decisions_path: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.manager = manager
        self.assigner = assigner
        self.board = board
        self.messages = messages
        self.bus = bus
        self.settings = settings
        self.decisions_path = decisions_path

    # ------------------------------------------------------------------ participants

    def weight(self, agent_name: str) -> float:
        agent = self.manager.get(agent_name)
        if agent is None:
            return 1.0
        weights = self.settings.orchestration.vote_weights
        return float(weights.get(agent.role.name, weights.get(agent.config_name, 1.0)))

    def pick_participants(self, exclude: set[str] | None = None) -> tuple[list[Agent], list[Agent]]:
        exclude = exclude or set()
        available = [a for a in self.manager.available() if a.name not in exclude]
        wanted = ["architecture", "planning", "coding", "research", "design"]
        ranked = sorted(available, key=lambda a: (-sum(c in a.capabilities for c in wanted), a.stats.tasks, a.name))
        proposers: list[Agent] = []
        providers: set[str] = set()
        for agent in ranked:
            provider = (agent.model_ref or "").split("/", 1)[0]
            if provider not in providers:
                proposers.append(agent)
                providers.add(provider)
            if len(proposers) == 2:
                break
        for agent in ranked:
            if len(proposers) >= 2:
                break
            if agent not in proposers:
                proposers.append(agent)
        critics = self.assigner.pick_reviewers(
            available,
            capabilities=["critique", "review", "security"],
            count=self.settings.orchestration.max_critics,
            exclude={p.name for p in proposers},
        )
        return proposers, critics

    # ------------------------------------------------------------------ helpers

    def _event(self, debate_id: str, stage: str, round_number: int, detail: str) -> None:
        self.bus.publish(DebateEvent(debate_id=debate_id, stage=stage, round=round_number, detail=detail))

    def _task(self, title: str, instructions: str, schema: str, purpose: str, question: str, node_id: str | None) -> AgentTask:
        return AgentTask(
            title=title,
            instructions=instructions,
            purpose=purpose,
            output="json",
            output_schema=schema,
            readonly=True,
            tool_groups=DEBATE_TOOLS,
            max_steps=8,
            node_id=node_id,
            shared_context_query=question,
        )

    async def _run_all(self, jobs: list[tuple[Agent, AgentTask]]) -> list[tuple[Agent, AgentTaskResult]]:
        results = await asyncio.gather(*(self.runtime.run(agent, task) for agent, task in jobs))
        return list(zip([a for a, _ in jobs], results, strict=True))

    @staticmethod
    def _parse_proposal(pid: str, author: str, data: Any, previous: Proposal | None = None) -> Proposal | None:
        if not isinstance(data, dict):
            return previous
        approach = str(data.get("approach") or (previous.approach if previous else "")).strip()
        if not approach:
            return previous
        return Proposal(
            id=pid,
            author=author,
            title=str(data.get("title") or (previous.title if previous else pid)),
            approach=approach,
            rationale=str(data.get("rationale") or ""),
            pros=_as_list(data.get("pros")),
            cons=_as_list(data.get("cons")),
            risks=_as_list(data.get("risks")),
            evidence=_as_list(data.get("evidence")),
            confidence=max(0.0, min(1.0, _as_float(data.get("confidence"), 0.5))),
            withdrawn=bool(data.get("withdrawn")),
            endorse=str(data["endorse"]) if data.get("endorse") else None,
        )

    def _support(self, proposals: list[Proposal], assessments: list[Assessment]) -> dict[str, tuple[float, bool]]:
        result: dict[str, tuple[float, bool]] = {}
        for proposal in proposals:
            support = oppose = 0.0
            blocking = False
            for item in assessments:
                if item.proposal_id != proposal.id:
                    continue
                weight = self.weight(item.critic)
                if item.stance == "support":
                    support += weight
                elif item.stance == "amend":
                    support += 0.5 * weight
                    oppose += 0.5 * weight if item.severity in ("major", "critical") else 0.0
                else:
                    oppose += weight
                if item.stance == "oppose" and item.severity == "critical":
                    blocking = True
            ratio = support / (support + oppose) if support + oppose > 0 else 0.0
            result[proposal.id] = (ratio, blocking)
        return result

    @staticmethod
    def _render_assessments(assessments: list[Assessment]) -> str:
        lines = []
        for item in assessments:
            text = f"- {item.critic} zu {item.proposal_id}: {item.stance} (severity: {item.severity}) - " + "; ".join(item.arguments)
            if item.suggested_changes:
                text += " | proposals: " + "; ".join(item.suggested_changes)
            lines.append(truncate_end(text, 1_200))
        return "\n".join(lines) or "(none)"

    def _persist(self, decision: Decision) -> None:
        if self.decisions_path is None:
            return
        existing: list[dict[str, Any]] = []
        if self.decisions_path.exists():
            try:
                existing = json.loads(self.decisions_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(decision.model_dump())
        tmp = self.decisions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.decisions_path)

    # ------------------------------------------------------------------ main flow

    async def debate(self, question: str, context: str = "", *, node_id: str | None = None, max_rounds: int | None = None) -> Decision:
        rounds = max_rounds or self.settings.limits.max_rounds
        debate_id = new_id("deb_")
        proposers, critics = self.pick_participants()
        judge = self.manager.orchestrator if self.manager.orchestrator and self.manager.orchestrator.available else (proposers[0] if proposers else None)
        decision = Decision(question=question, node_id=node_id, participants=[a.name for a in proposers + critics])
        if not proposers:
            decision.method = "fallback"
            decision.rationale = "No agents available for a debate."
            return decision
        self._event(debate_id, "start", 0, f"{len(proposers)} proposals, {len(critics)} critics")
        self.messages.post("orchestrator", [a.name for a in proposers + critics], "info", f"Debate started: {question}", topic=debate_id)

        # 1) proposals
        jobs = [
            (agent, self._task(f"Proposal: {one_line(question, 60)}", prompts.proposal_instructions(question, context), prompts.PROPOSAL_SCHEMA, "debate.propose", question, node_id))
            for agent in proposers
        ]
        proposals: list[Proposal] = []
        for index, (agent, result) in enumerate(await self._run_all(jobs), 1):
            proposal = self._parse_proposal(f"P{index}", agent.name, result.data)
            if proposal is not None:
                proposals.append(proposal)
                self.messages.post(agent.name, ["*"], "proposal", proposal.render(), topic=debate_id, summary=f"{proposal.id}: {proposal.title}")
        decision.proposals = proposals
        if not proposals:
            decision.method = "fallback"
            decision.rationale = "No agent could deliver a usable proposal."
            self._finalize(decision, judge, debate_id)
            return decision
        self._event(debate_id, "proposals", 0, ", ".join(f"{p.id}: {p.title}" for p in proposals))

        # 2) critique / revision rounds
        assessments: list[Assessment] = []
        best_id: str | None = None
        for round_number in range(1, rounds + 1):
            decision.rounds = round_number
            active = [p for p in proposals if not p.withdrawn]
            if not critics:
                break
            rendered = "\n\n".join(p.render() for p in active)
            jobs = [
                (critic, self._task(f"Score the proposals (round {round_number})", prompts.assessment_instructions(question, rendered), prompts.ASSESSMENT_SCHEMA, "debate.critique", question, node_id))
                for critic in critics
            ]
            assessments = []
            for critic, result in await self._run_all(jobs):
                items = result.data.get("assessments") if isinstance(result.data, dict) else None
                for item in items or []:
                    if not isinstance(item, dict) or not item.get("proposal_id"):
                        continue
                    assessment = Assessment(
                        critic=critic.name,
                        proposal_id=str(item["proposal_id"]),
                        stance=str(item.get("stance") or "amend").lower(),
                        severity=str(item.get("severity") or "none").lower(),
                        arguments=_as_list(item.get("arguments")),
                        evidence=_as_list(item.get("evidence")),
                        suggested_changes=_as_list(item.get("suggested_changes")),
                    )
                    assessments.append(assessment)
                    self.messages.post(critic.name, [p.author for p in active if p.id == assessment.proposal_id] or ["*"], "critique", f"{assessment.proposal_id}: {assessment.stance} – " + "; ".join(assessment.arguments), topic=debate_id)
            decision.assessments = assessments
            support = self._support(active, assessments)
            if support:
                best_id, (ratio, blocking) = max(support.items(), key=lambda kv: kv[1][0])
                self._event(debate_id, "consensus_check", round_number, f"best proposal {best_id}: approval {ratio:.0%}, blocking={blocking}")
                if ratio >= self.settings.orchestration.consensus_threshold and not blocking:
                    decision.consensus = True
                    break
            if round_number == rounds:
                break
            # revisions
            others_by_id = {p.id: p.render() for p in active}
            jobs = []
            for proposal in active:
                author = self.manager.get(proposal.author)
                if author is None:
                    continue
                own_assessments = [a for a in assessments if a.proposal_id == proposal.id]
                others = "\n\n".join(text for pid, text in others_by_id.items() if pid != proposal.id) or "(none)"
                jobs.append(
                    (author, self._task(f"Revise the proposal (round {round_number})", prompts.proposal_revision_instructions(question, proposal.render(), self._render_assessments(own_assessments), others), prompts.PROPOSAL_REVISION_SCHEMA, "debate.revise", question, node_id))
                )
            revised: dict[str, Proposal] = {}
            for (author, result), proposal in zip(await self._run_all(jobs), [p for p in active if self.manager.get(p.author)], strict=True):
                updated = self._parse_proposal(proposal.id, author.name, result.data, proposal)
                if updated is not None:
                    revised[proposal.id] = updated
            proposals = [revised.get(p.id, p) for p in proposals]
            for proposal in proposals:
                if proposal.withdrawn:
                    self._event(debate_id, "withdrawn", round_number, f"{proposal.id} withdrawn (supports {proposal.endorse or '-'})")
            decision.proposals = proposals

        active = [p for p in proposals if not p.withdrawn] or proposals

        # 3) votes (skipped on clear consensus with a single remaining proposal)
        totals: dict[str, float] = {p.id: 0.0 for p in active}
        if not (decision.consensus and len(active) == 1):
            rendered = "\n\n".join(p.render() for p in active)
            voters = list({a.name: a for a in proposers + critics}.values())
            jobs = [
                (voter, self._task("Vote", prompts.vote_instructions(question, rendered, self._render_assessments(assessments)), prompts.VOTE_SCHEMA, "debate.vote", question, node_id))
                for voter in voters
            ]
            for voter, result in await self._run_all(jobs):
                for item in (result.data or {}).get("votes", []) if isinstance(result.data, dict) else []:
                    if isinstance(item, dict) and str(item.get("proposal_id")) in totals:
                        score = max(1.0, min(10.0, _as_float(item.get("score"), 5.0)))
                        totals[str(item["proposal_id"])] += score * self.weight(voter.name)
            self._event(debate_id, "votes", decision.rounds, ", ".join(f"{k}: {v:.1f}" for k, v in totals.items()))
        decision.vote_totals = totals
        vote_leader = max(totals, key=lambda k: totals[k]) if any(totals.values()) else (best_id or active[0].id)

        # 4) decision
        chosen = next((p for p in active if p.id == (best_id if decision.consensus else vote_leader)), active[0])
        if decision.consensus and (vote_leader == chosen.id or not any(totals.values())):
            decision.method = "consensus" if critics else "single"
            decision.chosen_proposal = chosen.id
            decision.decision = chosen.approach
            decision.rationale = f"Consensus of the critics for {chosen.id}. {chosen.rationale}"
            decision.confidence = max(chosen.confidence, 0.7)
            decision.open_risks = chosen.risks
        elif judge is not None:
            await self._judge(decision, judge, question, context, active, assessments, totals, node_id, debate_id)
            if not decision.decision:
                decision.method = "vote"
                decision.chosen_proposal = vote_leader
                leader = next(p for p in active if p.id == vote_leader)
                decision.decision = leader.approach
                decision.rationale = "The judge's decision failed - the highest weighted vote was taken."
        self._finalize(decision, judge, debate_id)
        return decision

    async def _judge(
        self,
        decision: Decision,
        judge: Agent,
        question: str,
        context: str,
        proposals: list[Proposal],
        assessments: list[Assessment],
        totals: dict[str, float],
        node_id: str | None,
        debate_id: str,
    ) -> None:
        self._event(debate_id, "judge", decision.rounds, f"Richter: {judge.name}")
        task = self._task(
            "Take a decision",
            prompts.judge_instructions(
                question,
                context,
                "\n\n".join(p.render() for p in proposals),
                self._render_assessments(assessments),
                ", ".join(f"{k}: {v:.1f}" for k, v in totals.items()) or "(no votes)",
            ),
            prompts.JUDGE_SCHEMA,
            "debate.judge",
            question,
            node_id,
        )
        task.max_steps = 10
        result = await self.runtime.run(judge, task)
        data = result.data if isinstance(result.data, dict) else {}
        if not data.get("decision"):
            return
        decision.method = "judge"
        chosen = str(data.get("chosen_proposal") or "")
        decision.chosen_proposal = chosen if chosen in {p.id for p in proposals} else ("merged" if chosen else None)
        decision.decision = str(data["decision"])
        decision.rationale = str(data.get("rationale") or "")
        decision.evidence_checked = _as_list(data.get("evidence_checked"))
        decision.rejected = [r for r in data.get("rejected") or [] if isinstance(r, dict)]
        decision.confidence = max(0.0, min(1.0, _as_float(data.get("confidence"), 0.6)))
        decision.open_risks = _as_list(data.get("open_risks"))

    def _finalize(self, decision: Decision, judge: Agent | None, debate_id: str) -> None:
        author = judge.name if judge else "orchestrator"
        content = f"Decision: {decision.decision}\n\nRationale: {decision.rationale}\nMethod: {decision.method}, consensus: {decision.consensus}"
        if decision.open_risks:
            content += "\nOpen risks: " + "; ".join(decision.open_risks)
        self.board.add(author, "decision", one_line(decision.question, 120), content, node_id=decision.node_id, importance=5)
        self.messages.post(author, ["*"], "decision", content, topic=debate_id, summary=f"Decision: {one_line(decision.decision, 140)}")
        self._persist(decision)
        self.bus.publish(
            DecisionEvent(
                decision_id=decision.id,
                question=one_line(decision.question, 160),
                summary=one_line(decision.decision, 200),
                method=decision.method,
                consensus=decision.consensus,
            )
        )
