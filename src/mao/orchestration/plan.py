"""The plan document produced in PLAN mode and executed in RUN mode."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from mao.orchestration.taskgraph import NodeKind, TaskGraph, TaskNode, parse_kind


class PlanRisk(BaseModel):
    risk: str
    severity: str = "medium"
    mitigation: str = ""


class CritiqueRecord(BaseModel):
    reviewer: str
    round: int
    verdict: str = "revise"
    score: float | None = None
    strengths: list[str] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)


class PlanEstimate(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    low_tokens: int = 0
    high_tokens: int = 0
    cost_usd: float = 0.0
    low_cost_usd: float = 0.0
    high_cost_usd: float = 0.0
    planning_tokens_used: int = 0
    planning_cost_used: float = 0.0
    unpriced_models: list[str] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PlanDocument(BaseModel):
    task: str
    title: str = ""
    summary: str = ""
    understanding: str = ""
    approach: str = ""
    graph: TaskGraph = Field(default_factory=TaskGraph)
    risks: list[PlanRisk] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    critiques: list[CritiqueRecord] = Field(default_factory=list)
    consensus: bool = False
    rounds: int = 0
    warnings: list[str] = Field(default_factory=list)
    assignments: dict[str, str] = Field(default_factory=dict)
    estimate: PlanEstimate | None = None
    feedback_history: list[str] = Field(default_factory=list)
    version: int = 1
    approved: bool = False
    created_at: float = Field(default_factory=time.time)

    def to_agent_json(self) -> str:
        """Compact JSON of the plan content as agents should see it."""
        return json.dumps(
            {
                "title": self.title,
                "summary": self.summary,
                "approach": self.approach,
                "steps": [
                    {
                        "id": n.id,
                        "title": n.title,
                        "description": n.description,
                        "kind": n.kind.value,
                        "role": n.role,
                        "depends_on": n.depends_on,
                        "modifies_files": n.modifies_files,
                        "acceptance_criteria": n.acceptance_criteria,
                        "complexity": n.complexity,
                    }
                    for n in self.graph.nodes
                ],
                "risks": [r.model_dump() for r in self.risks],
                "assumptions": self.assumptions,
                "open_questions": self.open_questions,
                "success_criteria": self.success_criteria,
            },
            ensure_ascii=False,
            indent=1,
        )

    def apply_agent_data(self, data: dict[str, Any]) -> bool:
        """Update content from a planner JSON answer. Returns False if it has no usable steps."""
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            return False
        nodes: list[TaskNode] = []
        for index, raw in enumerate(steps, 1):
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or raw.get("name") or f"Step {index}").strip()
            depends = raw.get("depends_on") or raw.get("dependencies") or []
            if isinstance(depends, str):
                depends = [depends]
            criteria = raw.get("acceptance_criteria") or []
            if isinstance(criteria, str):
                criteria = [criteria]
            try:
                complexity = max(1, min(5, int(raw.get("complexity") or 2)))
            except (TypeError, ValueError):
                complexity = 2
            kind = parse_kind(raw.get("kind"))
            nodes.append(
                TaskNode(
                    id=str(raw.get("id") or f"s{index}").strip(),
                    title=title,
                    description=str(raw.get("description") or "").strip(),
                    kind=kind,
                    always_run=kind is NodeKind.SYNTHESIS,
                    depends_on=[str(d).strip() for d in depends],
                    role=(str(raw["role"]).strip() if raw.get("role") else None),
                    agent=(str(raw["agent"]).strip() if raw.get("agent") else None),
                    modifies_files=bool(raw.get("modifies_files")),
                    acceptance_criteria=[str(c) for c in criteria],
                    complexity=complexity,
                )
            )
        if not nodes:
            return False
        self.graph = TaskGraph(nodes=nodes)
        self.title = str(data.get("title") or self.title or "Plan").strip()
        self.summary = str(data.get("summary") or self.summary).strip()
        self.approach = str(data.get("approach") or self.approach).strip()
        risks = []
        for item in data.get("risks") or []:
            if isinstance(item, dict) and item.get("risk"):
                risks.append(PlanRisk(risk=str(item["risk"]), severity=str(item.get("severity") or "medium"), mitigation=str(item.get("mitigation") or "")))
            elif isinstance(item, str):
                risks.append(PlanRisk(risk=item))
        self.risks = risks
        for field_name in ("assumptions", "open_questions", "success_criteria"):
            value = data.get(field_name)
            if isinstance(value, list):
                setattr(self, field_name, [str(v) for v in value])
        return True
