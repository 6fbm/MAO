"""Chooses the most suitable agent for a task node."""

from __future__ import annotations

from mao.agents.agent import Agent
from mao.orchestration.taskgraph import NodeKind, TaskNode
from mao.security.permissions import CAPABILITY_LABELS, Capability, missing_capabilities

KIND_CAPABILITIES: dict[NodeKind, list[str]] = {
    NodeKind.ANALYSIS: ["analysis", "architecture"],
    NodeKind.RESEARCH: ["research"],
    NodeKind.DESIGN: ["architecture", "design"],
    NodeKind.DECISION: ["architecture", "planning", "judging"],
    NodeKind.IMPLEMENTATION: ["coding", "implementation"],
    NodeKind.TEST: ["testing", "qa"],
    NodeKind.REVIEW: ["review", "quality"],
    NodeKind.SECURITY: ["security"],
    NodeKind.DOCUMENTATION: ["documentation", "writing"],
    NodeKind.SYNTHESIS: ["management", "synthesis", "planning"],
}

KIND_REQUIRED: dict[NodeKind, set[Capability]] = {
    NodeKind.IMPLEMENTATION: {Capability.WRITE},
    NodeKind.TEST: {Capability.EXECUTE},
}


def required_capabilities(node: TaskNode) -> set[Capability]:
    required = set(KIND_REQUIRED.get(node.kind, set()))
    if node.modifies_files and node.kind in (NodeKind.IMPLEMENTATION, NodeKind.DOCUMENTATION):
        required.add(Capability.WRITE)
    return required


class AgentAssigner:
    def score(self, node: TaskNode, agent: Agent) -> float | None:
        if not agent.available or agent.internal or agent.name in node.excluded_agents:
            return None
        if missing_capabilities(agent.permissions, required_capabilities(node)):
            return None
        score = 0.0
        if node.agent and node.agent.lower() == agent.name.lower():
            score += 25
        if node.role and node.role.lower() in (agent.role.name.lower(), agent.config_name.lower()):
            score += 8
        wanted = KIND_CAPABILITIES.get(node.kind, [])
        score += 3 * sum(1 for cap in wanted if cap in agent.capabilities)
        if node.kind is NodeKind.RESEARCH and agent.permissions.internet:
            score += 3
        score -= 1.5 * agent.stats.failures
        score -= 0.1 * agent.stats.tasks
        return score

    def rank(self, node: TaskNode, agents: list[Agent]) -> list[tuple[float, Agent]]:
        ranked = [(s, a) for a in agents if (s := self.score(node, a)) is not None]
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        return ranked

    def assign(self, node: TaskNode, agents: list[Agent], busy: set[str]) -> Agent | None:
        for _score, agent in self.rank(node, agents):
            if agent.name not in busy:
                return agent
        return None

    def explain_unassignable(self, node: TaskNode, agents: list[Agent]) -> str | None:
        """Reason why no agent can ever take this node (None if some agent could)."""
        if self.rank(node, agents):
            return None
        required = required_capabilities(node)
        if required:
            labels = ", ".join(CAPABILITY_LABELS[c] for c in required)
            return f"No available agent with these permissions: {labels}"
        return "No available agent (all offline or excluded)"

    def pick_reviewers(self, agents: list[Agent], *, capabilities: list[str], count: int, exclude: set[str], author_provider: str | None = None) -> list[Agent]:
        candidates = [a for a in agents if a.available and not a.internal and a.name not in exclude]

        def key(agent: Agent) -> tuple[int, int, str]:
            overlap = sum(1 for cap in capabilities if cap in agent.capabilities)
            same_provider = 1 if author_provider and agent.model_ref and agent.model_ref.startswith(author_provider + "/") else 0
            return (-overlap, same_provider, agent.name)

        chosen = [a for a in sorted(candidates, key=key) if any(cap in a.capabilities for cap in capabilities)]
        return chosen[:count]
