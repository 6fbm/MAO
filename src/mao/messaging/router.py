"""Decides which shared information reaches which agent, within a token budget.

Push compact context (decisions, direct messages, dependency results, the most
relevant findings) and let the agent pull details with ``read_board``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mao.core.text import truncate_end
from mao.messaging.blackboard import Blackboard, BoardEntry
from mao.messaging.bus import MessageBus
from mao.tokens.estimator import ESTIMATOR, TokenEstimator

FINDING_KINDS = ["finding", "research", "risk", "recommendation", "proposal"]


@dataclass
class RoutingRequest:
    agent_name: str
    role: str
    capabilities: list[str] = field(default_factory=list)
    query_text: str = ""
    node_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    budget_tokens: int = 8_000
    include_workspace: bool = True


class ContextRouter:
    def __init__(self, board: Blackboard, bus: MessageBus, estimator: TokenEstimator = ESTIMATOR) -> None:
        self.board = board
        self.bus = bus
        self.estimator = estimator

    def build(self, request: RoutingRequest) -> str:
        remaining = request.budget_tokens
        sections: list[str] = []
        omitted = 0
        used_ids: set[str] = set()

        def add_section(title: str, items: list[str]) -> None:
            nonlocal remaining, omitted
            accepted: list[str] = []
            header_cost = self.estimator.count_text(title) + 4
            for item in items:
                cost = self.estimator.count_text(item) + 2
                if cost + header_cost > remaining:
                    omitted += 1
                    continue
                accepted.append(item)
                remaining -= cost
            if accepted:
                remaining -= header_cost
                sections.append(f"### {title}\n" + "\n\n".join(accepted))

        decisions = self.board.entries("decision")
        add_section("Decisions taken", [e.render(full=False) for e in decisions[-10:]])
        used_ids.update(e.id for e in decisions)

        inbox = self.bus.inbox(request.agent_name, request.role, unread_only=True)
        direct = [m for m in inbox if not m.is_broadcast]
        broadcast = [m for m in inbox if m.is_broadcast][-8:]
        add_section(
            "Messages addressed to you",
            [f"[{m.id}] of {m.sender} ({m.kind}): {truncate_end(m.content, 1_500)}" for m in direct],
        )
        self.bus.mark_read(request.agent_name, direct)

        dependency_results: list[BoardEntry] = []
        for dep in request.depends_on:
            dependency_results.extend(self.board.by_node(dep, "result"))
        add_section(
            "Results of the preceding steps",
            [e.render(full=True, max_chars=2_500) for e in dependency_results],
        )
        used_ids.update(e.id for e in dependency_results)

        query = " ".join([request.query_text, *request.capabilities])
        findings = self.board.search(query, kinds=FINDING_KINDS, limit=8, exclude_ids=used_ids)
        if len(findings) < 3:
            findings += [e for e in self.board.search(None, kinds=FINDING_KINDS, limit=5, exclude_ids=used_ids | {f.id for f in findings})]
        add_section("Relevant findings from other agents", [e.render(full=False) for e in findings])
        used_ids.update(e.id for e in findings)

        if request.include_workspace:
            workspace = self.board.entries("workspace")
            if workspace:
                add_section("Workspace overview", [truncate_end(workspace[-1].content, 6_000)])
                used_ids.add(workspace[-1].id)

        add_section("Broadcast messages", [f"{m.sender} ({m.kind}): {m.summary}" for m in broadcast])
        self.bus.mark_read(request.agent_name, broadcast)

        total_entries = len(self.board.entries())
        not_shown = max(0, total_entries - len(used_ids)) + omitted
        if not sections:
            return ""
        footer = f"\n({not_shown} further entries on the blackboard - fetch them with read_board when you need them.)" if not_shown else ""
        return "## Shared context\n" + "\n\n".join(sections) + footer
