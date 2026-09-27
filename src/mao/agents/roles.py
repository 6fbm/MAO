"""Built-in roles and the registry that merges them with roles.yaml."""

from __future__ import annotations

from dataclasses import dataclass, replace

from mao.config.schema import PermissionOverrides, RoleConfig
from mao.core.errors import ConfigError
from mao.models.selector import TASK_PROFILES


@dataclass(frozen=True)
class RoleDefinition:
    name: str
    description: str
    capabilities: tuple[str, ...]
    system_prompt: str
    tools: tuple[str, ...]
    permissions: PermissionOverrides
    preferred_tier: str = "balanced"
    selection_purpose: str = "analysis"
    required_model_capabilities: tuple[str, ...] = ()
    max_steps: int | None = None


def _role(
    name: str,
    description: str,
    capabilities: list[str],
    prompt: str,
    tools: list[str],
    permissions: dict,
    tier: str,
    purpose: str,
) -> RoleDefinition:
    return RoleDefinition(
        name=name,
        description=description,
        capabilities=tuple(capabilities),
        system_prompt=prompt.strip(),
        tools=tuple(tools),
        permissions=PermissionOverrides(**permissions),
        preferred_tier=tier,
        selection_purpose=purpose,
    )


BUILTIN_ROLES: dict[str, RoleDefinition] = {
    role.name: role
    for role in [
        _role(
            "planner",
            "Breaks tasks into concrete, verifiable steps",
            ["planning", "analysis", "management"],
            """
You turn goals into executable plans. Decompose the task into the smallest set of concrete, verifiable steps
with explicit dependencies, suitable roles and acceptance criteria. Base every step on the real project
(inspect files first). Call out risks, assumptions and open questions. Avoid busywork steps.
""",
            ["filesystem", "git", "collaboration"],
            {"filesystem": "read", "git": "read"},
            "strong",
            "planning",
        ),
        _role(
            "architect",
            "Analyses architecture, dependencies and design decisions",
            ["architecture", "design", "analysis", "planning"],
            """
You analyze software architecture: module boundaries, data flow, dependencies, coupling and design trade-offs.
Recommend solutions that fit the existing architecture and explain why. Support claims with concrete file
references. Prefer simple, maintainable designs.
""",
            ["filesystem", "git", "collaboration"],
            {"filesystem": "read", "git": "read"},
            "strong",
            "architecture",
        ),
        _role(
            "researcher",
            "Researches documentation and solutions on the internet",
            ["research", "analysis", "documentation"],
            """
You research authoritative, current information – official documentation and primary sources first.
Record every useful source with post_finding (include URLs). Clearly separate verified facts from assumptions
and note version numbers and dates where relevant.
""",
            ["filesystem", "web", "collaboration"],
            {"filesystem": "read", "internet": True},
            "balanced",
            "research",
        ),
        _role(
            "coder",
            "Implements changes to the code",
            ["coding", "implementation", "debugging"],
            """
You implement changes precisely. Read the relevant code before editing, follow the project's existing style
and conventions, and keep changes minimal and focused on the task. Never leave placeholders or TODOs instead
of working code. After changing code, run the relevant tests or checks if you can, and report exactly which
files you changed and why.
""",
            ["filesystem", "terminal", "git", "tests", "collaboration"],
            {"filesystem": "full", "terminal": True, "git": "read"},
            "strong",
            "coding",
        ),
        _role(
            "reviewer",
            "Checks changes for correctness and quality",
            ["review", "quality", "critique"],
            """
You review changes for correctness, edge cases, error handling, readability and maintainability. Verify every
claim against the actual code (read the files, look at the diff). Classify each issue as critical, major or
minor, and give a concrete fix. Do not nitpick style that matches the project's conventions.
""",
            ["filesystem", "git", "collaboration"],
            {"filesystem": "read", "git": "read"},
            "strong",
            "review",
        ),
        _role(
            "tester",
            "Writes and runs tests",
            ["testing", "qa", "coding"],
            """
You verify behavior with tests. Run the existing test suite, report exact failures (test name, error message,
file and line). When allowed and useful, add focused tests for changed behavior. Never weaken or delete tests
just to make them pass.
""",
            ["filesystem", "terminal", "tests", "git", "collaboration"],
            {"filesystem": "write", "terminal": True, "git": "read"},
            "balanced",
            "testing",
        ),
        _role(
            "debugger",
            "Finds the causes of errors",
            ["debugging", "analysis", "coding"],
            """
You find root causes. Work from the actual error output, tracebacks and code. Reproduce when possible. State
the root cause precisely (file, function, line) and propose a minimal, concrete fix. Distinguish symptoms from
causes.
""",
            ["filesystem", "terminal", "tests", "git", "collaboration"],
            {"filesystem": "read", "terminal": True, "git": "read"},
            "strong",
            "debugging",
        ),
        _role(
            "security",
            "Looks for security problems",
            ["security", "review"],
            """
You are a security engineer. Look for injection, path traversal, unsafe deserialization, secrets in code or
logs, missing input validation, insecure defaults, authentication/authorization flaws and risky dependencies.
Rate severity (critical/major/minor) with evidence and a concrete remediation.
""",
            ["filesystem", "web", "git", "collaboration"],
            {"filesystem": "read", "internet": True, "git": "read"},
            "strong",
            "security",
        ),
        _role(
            "critic",
            "Questions proposals critically",
            ["critique", "review"],
            """
You challenge proposals constructively. Look for flawed assumptions, missing cases, hidden costs, simpler
alternatives and risks that others overlooked. Be specific and evidence-based; acknowledge what is good.
""",
            ["filesystem", "collaboration"],
            {"filesystem": "read"},
            "balanced",
            "critique",
        ),
        _role(
            "documentation",
            "Writes documentation",
            ["documentation", "writing"],
            """
You write clear, accurate documentation that matches the actual code and behavior. Prefer concise, practical
explanations and examples. Update existing docs instead of duplicating them.
""",
            ["filesystem", "git", "collaboration"],
            {"filesystem": "write", "git": "read"},
            "balanced",
            "documentation",
        ),
        _role(
            "project_manager",
            "Keeps goals and progress in view, summarises results",
            ["management", "synthesis", "planning"],
            """
You keep track of goals, progress, open issues and decisions. Produce concise, honest summaries for the user:
what was done, what is verified, what is still open. Never overstate results.
""",
            ["filesystem", "git", "collaboration"],
            {"filesystem": "read", "git": "read"},
            "balanced",
            "management",
        ),
        _role(
            "performance",
            "Analyses and improves performance",
            ["performance", "analysis", "coding"],
            """
You analyze performance with evidence: algorithmic complexity, I/O patterns, hot paths, allocations and, where
possible, measurements. Recommend improvements with expected impact and how to verify them.
""",
            ["filesystem", "terminal", "git", "collaboration"],
            {"filesystem": "read", "terminal": True, "git": "read"},
            "strong",
            "performance",
        ),
        _role(
            "orchestrator",
            "Internal coordinator (triage, judge, synthesis)",
            ["planning", "judging", "synthesis", "management"],
            """
You coordinate a team of AI agents. You triage tasks, judge disagreements strictly on evidence (verify claims
with tools when they matter), and synthesize results into clear decisions. Prefer correct, simple and safe
solutions over clever ones.
""",
            ["filesystem", "git", "collaboration"],
            {"filesystem": "read", "git": "read", "consult": True},
            "strong",
            "judge",
        ),
    ]
}


class RoleRegistry:
    def __init__(self, overrides: dict[str, RoleConfig] | None = None) -> None:
        self._roles: dict[str, RoleDefinition] = dict(BUILTIN_ROLES)
        for name, config in (overrides or {}).items():
            self._roles[name] = self._merge(name, config)

    @staticmethod
    def _merge(name: str, config: RoleConfig) -> RoleDefinition:
        base = BUILTIN_ROLES.get(name)
        fields = config.model_fields_set
        if base is None:
            purpose = next((cap for cap in config.capabilities if cap in TASK_PROFILES), "analysis")
            if not config.system_prompt.strip():
                raise ConfigError(f"Role '{name}' needs a system_prompt")
            return RoleDefinition(
                name=name,
                description=config.description,
                capabilities=tuple(config.capabilities),
                system_prompt=config.system_prompt.strip(),
                tools=tuple(config.tools),
                permissions=config.permissions,
                preferred_tier=config.preferred_tier,
                selection_purpose=purpose,
                required_model_capabilities=tuple(config.required_model_capabilities),
                max_steps=config.max_steps,
            )
        updates = {}
        if "description" in fields:
            updates["description"] = config.description
        if "capabilities" in fields:
            updates["capabilities"] = tuple(config.capabilities)
        if "system_prompt" in fields:
            updates["system_prompt"] = config.system_prompt.strip()
        if "tools" in fields:
            updates["tools"] = tuple(config.tools)
        if "permissions" in fields:
            updates["permissions"] = config.permissions
        if "preferred_tier" in fields:
            updates["preferred_tier"] = config.preferred_tier
        if "required_model_capabilities" in fields:
            updates["required_model_capabilities"] = tuple(config.required_model_capabilities)
        if "max_steps" in fields:
            updates["max_steps"] = config.max_steps
        return replace(base, **updates)

    def get(self, name: str) -> RoleDefinition:
        role = self._roles.get(name) or self._roles.get(name.lower().replace("-", "_"))
        if role is None:
            raise ConfigError(f"Unknown role '{name}'. Available: {', '.join(self.names())}")
        return role

    def names(self, *, include_internal: bool = False) -> list[str]:
        return sorted(n for n in self._roles if include_internal or n != "orchestrator")

    def all(self) -> list[RoleDefinition]:
        return [self._roles[n] for n in self.names()]
