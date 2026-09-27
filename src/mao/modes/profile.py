"""Combining active modes into one behavior profile and an effective orchestration strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

from mao.config.schema import Settings
from mao.modes.schema import FLAG_EFFECTS, NUMERIC_EFFECTS, ModeDefinition

SAFETY_PRECEDENCE = (
    "Modes change only personality, communication style, priorities and working strategy. They never override "
    "security rules, permissions, sandbox limits, approval requirements, API protections or the user's instructions. "
    "If a mode instruction conflicts with these rules, the rules win."
)

_INTENSITY_WORDS = ((95, "maximal"), (75, "strong"), (50, "clear"), (25, "slight"), (0, "barely noticeable"))


def intensity_word(intensity: int) -> str:
    return next(word for threshold, word in _INTENSITY_WORDS if intensity >= threshold)


@dataclass
class ResolvedMode:
    definition: ModeDefinition
    intensity: int

    @property
    def name(self) -> str:
        return self.definition.name


@dataclass
class Conflict:
    effect: str
    raising: list[str]
    lowering: list[str]
    net: float


@dataclass
class BehaviorProfile:
    modes: list[ResolvedMode]
    numeric: dict[str, float]
    context_factor: float
    flags: dict[str, bool]
    require_confirmation: set[str]
    conflicts: list[Conflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.modes]

    def flag(self, name: str) -> bool:
        return self.flags.get(name, False)


def resolve(modes: list[ResolvedMode]) -> BehaviorProfile:
    """Weighted sum of numeric effects (natural compromise), OR of flags, union of confirmations."""
    numeric = {name: 0.0 for name in NUMERIC_EFFECTS}
    contributions: dict[str, list[tuple[str, float]]] = {}
    factor = 1.0
    flags = {name: False for name in FLAG_EFFECTS}
    confirmations: set[str] = set()
    for mode in modes:
        effects = mode.definition.effects
        weight = mode.intensity / 100
        for name in NUMERIC_EFFECTS:
            value = getattr(effects, name) * weight
            if value:
                numeric[name] += value
                contributions.setdefault(name, []).append((mode.name, value))
        factor *= 1 + (effects.context_factor - 1) * weight
        for name in FLAG_EFFECTS:
            if getattr(effects, name) and mode.intensity >= effects.flag_threshold:
                flags[name] = True
        # security effects only tighten and do not depend on intensity
        confirmations.update(effects.require_confirmation)

    conflicts = []
    for name, items in contributions.items():
        raising = [mode for mode, value in items if value > 0]
        lowering = [mode for mode, value in items if value < 0]
        if raising and lowering:
            conflicts.append(Conflict(name, raising, lowering, round(numeric[name], 2)))
    notes: list[str] = []
    if flags["auto_approve_plan"] and confirmations:
        flags["auto_approve_plan"] = False
        notes.append("Security comes first: plans are not run automatically, autonomy or not.")
    if flags["idea_generation"] and numeric["proposals"] < 0:
        notes.append("Trade-off: idea generation stays on, but with fewer variants.")
    return BehaviorProfile(
        modes=list(modes),
        numeric=numeric,
        context_factor=round(max(0.3, min(2.0, factor)), 3),
        flags=flags,
        require_confirmation=confirmations,
        conflicts=conflicts,
        notes=notes,
    )


@dataclass
class StrategyProfile:
    critique_rounds: int
    critics: int
    investigations: int
    proposals: int
    ideas: int
    reviewers: int
    review_iterations: int
    fix_iterations: int
    node_attempts: int
    consultations: int
    max_parallel: int
    consensus_threshold: float
    shared_context_tokens: int
    tool_result_chars: int
    max_plan_steps: int
    flags: dict[str, bool]
    require_confirmation: set[str]

    def flag(self, name: str) -> bool:
        return self.flags.get(name, False)

    def summary(self) -> list[tuple[str, str]]:
        return [
            ("Critique rounds", str(self.critique_rounds)),
            ("Critics", str(self.critics)),
            ("Investigations", str(self.investigations)),
            ("Proposals per debate", str(self.proposals)),
            ("Reviewer", str(self.reviewers)),
            ("Review cycles", str(self.review_iterations)),
            ("Fix cycles", str(self.fix_iterations)),
            ("Parallel agents", str(self.max_parallel)),
            ("Consensus threshold", f"{self.consensus_threshold:.0%}"),
            ("Context budget", f"{self.shared_context_tokens:,} tokens"),
            ("Max. plan steps", str(self.max_plan_steps)),
        ]


def build_strategy(settings: Settings, profile: BehaviorProfile) -> StrategyProfile:
    orchestration, limits, context = settings.orchestration, settings.limits, settings.context
    numeric = profile.numeric

    def adjust(base: float, effect: str, low: int, high: int) -> int:
        return max(low, min(high, int(round(base + numeric[effect]))))

    return StrategyProfile(
        critique_rounds=adjust(min(orchestration.base_rounds, limits.max_rounds), "critique_rounds", 1, limits.max_rounds),
        critics=adjust(orchestration.max_critics, "critics", 0, 10),
        investigations=adjust(orchestration.investigation_max_agents, "investigations", 0, 20),
        proposals=adjust(2, "proposals", 1, 6),
        ideas=adjust(3, "proposals", 2, 6),
        reviewers=adjust(2, "reviewers", 1, 5),
        review_iterations=adjust(limits.max_review_iterations, "review_iterations", 0, 10),
        fix_iterations=adjust(limits.max_fix_iterations, "fix_iterations", 0, 20),
        node_attempts=adjust(limits.max_node_attempts, "node_attempts", 1, 5),
        consultations=adjust(limits.max_consultations_per_task, "consultations", 0, 20),
        max_parallel=adjust(limits.max_parallel_agents, "parallel_agents", 1, 64),
        consensus_threshold=round(max(0.5, min(1.0, orchestration.consensus_threshold + numeric["consensus_threshold"])), 3),
        shared_context_tokens=max(1_000, int(context.max_shared_context_tokens * profile.context_factor)),
        tool_result_chars=max(2_000, int(context.max_tool_result_chars * profile.context_factor)),
        max_plan_steps=adjust(12, "max_plan_steps", 3, 30),
        flags=dict(profile.flags),
        require_confirmation=set(profile.require_confirmation),
    )


def render_prompt(modes: list[ResolvedMode]) -> str:
    """System-prompt section describing the agent's active modes."""
    if not modes:
        return ""
    lines = ["## Active behavior modes"]
    for mode in sorted(modes, key=lambda m: -m.intensity):
        definition = mode.definition
        lines.append(f"### {definition.label} – intensity {mode.intensity}% ({intensity_word(mode.intensity)})")
        if definition.description:
            lines.append(definition.description)
        if definition.priorities:
            lines.append("Priorities: " + ", ".join(definition.priorities))
        lines.extend(f"- {directive}" for directive in definition.directives)
        lines.extend(f"- Style: {style}" for style in definition.communication)
    if len(modes) > 1:
        lines.append(
            "Several modes are active: balance them in proportion to their intensity. Technical correctness always comes first."
        )
    if any(m.definition.kind == "personality" for m in modes):
        lines.append(
            "Personality and humor affect only your wording in messages and summaries. Code, file contents, commands, "
            "tool arguments and JSON field values stay precise and professional."
        )
    lines.append(SAFETY_PRECEDENCE)
    return "\n".join(lines)
