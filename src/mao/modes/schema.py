"""Data model of the mode system: definitions, effects and the active state."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MODE_NAME_RE = re.compile(r"^[a-z][a-z0-9_\-]{1,31}$")

ConfirmAction = Literal[
    "create", "overwrite", "edit", "mkdir", "move", "delete", "execute",
    "internet", "git_write", "git_destructive", "sensitive_read",
]

NUMERIC_EFFECTS = (
    "critique_rounds",
    "critics",
    "investigations",
    "proposals",
    "reviewers",
    "review_iterations",
    "fix_iterations",
    "node_attempts",
    "consultations",
    "parallel_agents",
    "consensus_threshold",
    "max_plan_steps",
)

FLAG_EFFECTS = (
    "idea_generation",
    "debate_on_approach",
    "research_first",
    "parallel_implementation",
    "security_gate",
    "documentation_gate",
    "minor_issues_blocking",
    "skip_critique_when_simple",
    "spawn_agents",
    "auto_approve_plan",
    "skip_votes_on_consensus",
)


class ModeEffects(BaseModel):
    """Effects on the orchestration. Numeric values are deltas at 100 % intensity.

    Deliberately there is no field that could loosen permissions, approvals, sandbox or API
    protections. ``require_confirmation`` can only make approvals stricter and applies fully
    whenever the mode is active – intensity never weakens a security effect.
    """

    model_config = ConfigDict(extra="forbid")

    critique_rounds: float = Field(0, ge=-5, le=5)
    critics: float = Field(0, ge=-5, le=5)
    investigations: float = Field(0, ge=-5, le=5)
    proposals: float = Field(0, ge=-3, le=4)
    reviewers: float = Field(0, ge=-2, le=3)
    review_iterations: float = Field(0, ge=-3, le=5)
    fix_iterations: float = Field(0, ge=-3, le=5)
    node_attempts: float = Field(0, ge=-2, le=3)
    consultations: float = Field(0, ge=-5, le=5)
    parallel_agents: float = Field(0, ge=-8, le=8)
    consensus_threshold: float = Field(0, ge=-0.3, le=0.3)
    max_plan_steps: float = Field(0, ge=-8, le=8)
    context_factor: float = Field(1.0, ge=0.3, le=2.0)

    idea_generation: bool = False
    debate_on_approach: bool = False
    research_first: bool = False
    parallel_implementation: bool = False
    security_gate: bool = False
    documentation_gate: bool = False
    minor_issues_blocking: bool = False
    skip_critique_when_simple: bool = False
    spawn_agents: bool = False
    auto_approve_plan: bool = False
    skip_votes_on_consensus: bool = False
    # flags become active from this intensity on
    flag_threshold: int = Field(50, ge=0, le=100)

    require_confirmation: list[ConfirmAction] = Field(default_factory=list)


class ModeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str = ""
    description: str = ""
    kind: Literal["behavior", "personality", "strategy"] = "behavior"
    default_intensity: int = Field(80, ge=0, le=100)
    priorities: list[str] = Field(default_factory=list)
    directives: list[str] = Field(default_factory=list)
    communication: list[str] = Field(default_factory=list)
    effects: ModeEffects = Field(default_factory=ModeEffects)
    conflicts_with: list[str] = Field(default_factory=list)
    # set by the registry, never read from YAML semantics
    builtin: bool = False
    source: str = ""

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        value = value.strip().lower()
        if not MODE_NAME_RE.match(value):
            raise ValueError("Mode name: 2-32 characters, lowercase letters, digits, '-' or '_', starting with a letter")
        return value

    @property
    def label(self) -> str:
        return (self.title or self.name).upper()


class ActiveMode(BaseModel):
    name: str
    intensity: int = Field(80, ge=0, le=100)


class ModeState(BaseModel):
    global_modes: list[ActiveMode] = Field(default_factory=list)
    orchestrator_modes: list[ActiveMode] = Field(default_factory=list)
    agent_modes: dict[str, list[ActiveMode]] = Field(default_factory=dict)
    temporary_pending: list[ActiveMode] | None = None
    temporary_active: list[ActiveMode] | None = None
