"""Task graph (DAG) with validation, repair and scheduling queries."""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    ANALYSIS = "analysis"
    RESEARCH = "research"
    DESIGN = "design"
    DECISION = "decision"
    IMPLEMENTATION = "implementation"
    TEST = "test"
    REVIEW = "review"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    SYNTHESIS = "synthesis"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


_KIND_ALIASES = {
    "analyze": NodeKind.ANALYSIS, "analyse": NodeKind.ANALYSIS, "investigation": NodeKind.ANALYSIS,
    "research": NodeKind.RESEARCH, "search": NodeKind.RESEARCH,
    "design": NodeKind.DESIGN, "architecture": NodeKind.DESIGN,
    "decision": NodeKind.DECISION, "decide": NodeKind.DECISION, "debate": NodeKind.DECISION,
    "implementation": NodeKind.IMPLEMENTATION, "implement": NodeKind.IMPLEMENTATION, "coding": NodeKind.IMPLEMENTATION,
    "code": NodeKind.IMPLEMENTATION, "fix": NodeKind.IMPLEMENTATION, "refactor": NodeKind.IMPLEMENTATION,
    "test": NodeKind.TEST, "tests": NodeKind.TEST, "testing": NodeKind.TEST, "verification": NodeKind.TEST,
    "review": NodeKind.REVIEW, "code_review": NodeKind.REVIEW,
    "security": NodeKind.SECURITY, "security_review": NodeKind.SECURITY,
    "documentation": NodeKind.DOCUMENTATION, "docs": NodeKind.DOCUMENTATION,
    "synthesis": NodeKind.SYNTHESIS, "summary": NodeKind.SYNTHESIS, "report": NodeKind.SYNTHESIS,
}


def parse_kind(value: object) -> NodeKind:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return NodeKind(text)
    except ValueError:
        return _KIND_ALIASES.get(text, NodeKind.ANALYSIS)


class TaskNode(BaseModel):
    id: str
    title: str
    description: str = ""
    kind: NodeKind = NodeKind.ANALYSIS
    depends_on: list[str] = Field(default_factory=list)
    role: str | None = None
    agent: str | None = None
    modifies_files: bool = False
    acceptance_criteria: list[str] = Field(default_factory=list)
    complexity: int = Field(2, ge=1, le=5)
    auto_added: bool = False
    # runs once all dependencies are finished, even if some failed (e.g. the final report)
    always_run: bool = False
    status: NodeStatus = NodeStatus.PENDING
    outcome: str | None = None  # success | issues | failed
    attempts: int = 0
    assigned_agent: str | None = None
    excluded_agents: list[str] = Field(default_factory=list)
    result_summary: str = ""
    result_entry: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def terminal(self) -> bool:
        return self.status in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.SKIPPED)

    def mark_running(self, agent: str) -> None:
        self.status = NodeStatus.RUNNING
        self.assigned_agent = agent
        self.attempts += 1
        self.started_at = time.time()
        self.error = None

    def mark_finished(self, status: NodeStatus, *, outcome: str | None = None, summary: str = "", error: str | None = None) -> None:
        self.status = status
        self.outcome = outcome
        self.result_summary = summary
        self.error = error
        self.finished_at = time.time()


class TaskGraph(BaseModel):
    nodes: list[TaskNode] = Field(default_factory=list)

    def get(self, node_id: str) -> TaskNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def add(self, node: TaskNode) -> TaskNode:
        existing = set(self.ids())
        if node.id in existing:
            base, suffix = node.id, 2
            while f"{base}_{suffix}" in existing:
                suffix += 1
            node.id = f"{base}_{suffix}"
        self.nodes.append(node)
        return node

    def validate_and_repair(self) -> list[str]:
        warnings: list[str] = []
        seen: set[str] = set()
        for node in self.nodes:
            if not node.id or node.id in seen:
                new_id = f"s{len(seen) + 1}"
                while new_id in seen:
                    new_id += "_x"
                warnings.append(f"Duplicate or empty step ID '{node.id}' renamed to '{new_id}'")
                node.id = new_id
            seen.add(node.id)
        for node in self.nodes:
            cleaned = []
            for dep in node.depends_on:
                if dep == node.id:
                    warnings.append(f"{node.id}: removed self-dependency")
                elif dep not in seen:
                    warnings.append(f"{node.id}: unknown dependency '{dep}' removed")
                elif dep not in cleaned:
                    cleaned.append(dep)
            node.depends_on = cleaned
        # break cycles: drop edges to nodes that appear later in the list
        while True:
            cycle_nodes = self._cycle_members()
            if not cycle_nodes:
                break
            order = {n.id: i for i, n in enumerate(self.nodes)}
            victim = max(cycle_nodes, key=lambda nid: order[nid])
            node = self.get(victim)
            assert node is not None
            removed = [d for d in node.depends_on if d in cycle_nodes and order[d] > order[victim]] or [d for d in node.depends_on if d in cycle_nodes]
            node.depends_on = [d for d in node.depends_on if d not in removed[:1]]
            warnings.append(f"Cyclic dependency resolved: {victim} no longer depends on {removed[0]} ab")
        return warnings

    def _cycle_members(self) -> set[str]:
        indegree = {n.id: 0 for n in self.nodes}
        children: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for node in self.nodes:
            for dep in node.depends_on:
                if dep in indegree:
                    indegree[node.id] += 1
                    children[dep].append(node.id)
        queue = [nid for nid, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        return {nid for nid, deg in indegree.items() if deg > 0}

    def ready(self) -> list[TaskNode]:
        done = {n.id for n in self.nodes if n.status is NodeStatus.DONE}
        finished = {n.id for n in self.nodes if n.terminal}
        return [
            n
            for n in self.nodes
            if n.status is NodeStatus.PENDING
            and all(d in (finished if n.always_run else done) for d in n.depends_on)
        ]

    def skip_blocked(self) -> list[TaskNode]:
        """Mark pending nodes whose dependencies failed or were skipped."""
        skipped: list[TaskNode] = []
        changed = True
        while changed:
            changed = False
            dead = {n.id for n in self.nodes if n.status in (NodeStatus.FAILED, NodeStatus.SKIPPED)}
            for node in self.nodes:
                if node.always_run:
                    continue
                if node.status is NodeStatus.PENDING and any(d in dead for d in node.depends_on):
                    node.mark_finished(NodeStatus.SKIPPED, outcome="skipped", error="dependency failed")
                    skipped.append(node)
                    changed = True
        return skipped

    def dependents(self, node_id: str) -> list[TaskNode]:
        return [n for n in self.nodes if node_id in n.depends_on]

    def topological_order(self) -> list[TaskNode]:
        order: list[TaskNode] = []
        placed: set[str] = set()
        remaining = list(self.nodes)
        while remaining:
            progress = False
            for node in list(remaining):
                if all(d in placed for d in node.depends_on):
                    order.append(node)
                    placed.add(node.id)
                    remaining.remove(node)
                    progress = True
            if not progress:  # cycle – should have been repaired
                order.extend(remaining)
                break
        return order

    def progress(self) -> tuple[int, int]:
        return sum(1 for n in self.nodes if n.terminal), len(self.nodes)

    def is_finished(self) -> bool:
        return all(n.terminal for n in self.nodes)

    def reset_interrupted(self) -> None:
        for node in self.nodes:
            if node.status is NodeStatus.RUNNING:
                node.status = NodeStatus.PENDING
                node.assigned_agent = None

    def counts(self) -> dict[str, int]:
        result = {status.value: 0 for status in NodeStatus}
        for node in self.nodes:
            result[node.status.value] += 1
        return result
