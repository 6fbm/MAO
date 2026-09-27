"""Creates agent instances from configuration and resolves their models."""

from __future__ import annotations

from mao.agents.agent import Agent, AgentState
from mao.agents.roles import RoleDefinition, RoleRegistry
from mao.config.schema import AgentConfig, AgentsFile, PermissionsConfig, Settings
from mao.core.errors import ConfigError
from mao.core.events import AgentStatusEvent, EventBus
from mao.messaging.bus import base_name
from mao.models.catalog import ModelCatalog
from mao.models.selector import ModelSelector, SelectionRequest
from mao.security.permissions import resolve_permissions

ORCHESTRATOR_NAME = "orchestrator"


class AgentManager:
    def __init__(
        self,
        *,
        agents_config: AgentsFile,
        roles: RoleRegistry,
        catalog: ModelCatalog,
        selector: ModelSelector,
        settings: Settings,
        permissions: PermissionsConfig,
        bus: EventBus,
    ) -> None:
        self.agents_config = agents_config
        self.roles = roles
        self.catalog = catalog
        self.selector = selector
        self.settings = settings
        self.permissions = permissions
        self.bus = bus
        self._agents: dict[str, Agent] = {}
        self.orchestrator: Agent | None = None
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ building

    def _selection_request(self, role: RoleDefinition, usage: dict[str, int]) -> SelectionRequest:
        return SelectionRequest(
            purpose=role.selection_purpose,
            preferred_tier=role.preferred_tier,
            required_capabilities=[*role.required_model_capabilities, "tools"],
            local_only=self.settings.privacy.local_only,
            provider_usage=dict(usage),
            policy=self.settings.orchestration.model_policy,
        )

    def resolve_model(self, model: str, fallbacks: list[str], role: RoleDefinition, usage: dict[str, int]) -> tuple[str | None, list[str], str]:
        request = self._selection_request(role, usage)
        auto = self.settings.orchestration.auto_model_selection
        if model.strip().lower() == "auto":
            if not auto:
                return None, [], "model: auto, but auto_model_selection is disabled"
            entry = self.selector.select(request)
            if entry is None:
                return None, [], "no model available (API key missing or local server offline)"
        else:
            try:
                entry = self.catalog.resolve(model)
            except ConfigError as exc:
                return None, [], str(exc)
            if not self.catalog.is_available(entry):
                return None, [], f"{entry.ref}: {self.catalog.unavailable_reason(entry)}"
        if fallbacks:
            fallback_refs = list(fallbacks)
        elif auto:
            fallback_refs = [e.ref for e in self.selector.fallbacks(entry, request, 2)]
        else:
            fallback_refs = []
        return entry.ref, fallback_refs, ""

    def _make_agent(self, name: str, config: AgentConfig, role: RoleDefinition, usage: dict[str, int]) -> Agent:
        permissions = resolve_permissions(self.permissions.defaults, role.permissions, config.permissions, self.permissions.ceiling)
        model_ref, fallbacks, reason = self.resolve_model(config.model, config.fallback_models, role, usage)
        if model_ref:
            provider = model_ref.split("/", 1)[0]
            usage[provider] = usage.get(provider, 0) + 1
        agent = Agent(
            name=name,
            config_name=config.name,
            role=role,
            model_ref=model_ref,
            fallback_refs=fallbacks,
            permissions=permissions,
            tool_groups=list(config.tools) if config.tools is not None else list(role.tools),
            capabilities=list(config.capabilities) if config.capabilities is not None else list(role.capabilities),
            system_prompt=(config.system_prompt or role.system_prompt).strip(),
            extra_instructions=config.extra_instructions,
            temperature=config.temperature,
            max_steps=config.max_steps or role.max_steps or self.settings.limits.max_steps_per_task,
            max_output_tokens=config.max_output_tokens,
            offline_reason=reason,
        )
        if not agent.available:
            agent.state = AgentState.OFFLINE
        return agent

    def build(self) -> list[str]:
        self.warnings = []
        self._agents = {}
        instances: list[tuple[str, AgentConfig, RoleDefinition]] = []
        for config in self.agents_config.agents:
            if not config.enabled:
                continue
            try:
                role = self.roles.get(config.role)
            except ConfigError as exc:
                self.warnings.append(f"Agent '{config.name}': {exc}")
                continue
            for index in range(config.count):
                name = config.name if config.count == 1 else f"{config.name}-{index + 1}"
                instances.append((name, config, role))
        limit = self.settings.limits.max_agents
        if len(instances) > limit:
            self.warnings.append(f"{len(instances)} agents configured, limit max_agents={limit} - the extra ones are ignored")
            instances = instances[:limit]
        usage: dict[str, int] = {}
        for name, config, role in instances:
            agent = self._make_agent(name, config, role, usage)
            self._agents[name] = agent
            if not agent.available:
                self.warnings.append(f"Agent '{name}' offline: {agent.offline_reason}")
        self.orchestrator = self._build_orchestrator()
        return self.warnings

    def _build_orchestrator(self) -> Agent:
        role = self.roles.get("orchestrator")
        config = AgentConfig(name=ORCHESTRATOR_NAME, role="orchestrator", model=self.settings.orchestration.orchestrator_model)
        agent = self._make_agent(ORCHESTRATOR_NAME, config, role, {})
        agent.internal = True
        return agent

    # ------------------------------------------------------------------ queries

    def all(self, *, include_orchestrator: bool = False) -> list[Agent]:
        agents = list(self._agents.values())
        if include_orchestrator and self.orchestrator is not None:
            agents.append(self.orchestrator)
        return agents

    def available(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.available]

    def get(self, name: str) -> Agent | None:
        if self.orchestrator is not None and name.lower() == ORCHESTRATOR_NAME:
            return self.orchestrator
        for agent_name, agent in self._agents.items():
            if agent_name.lower() == name.lower():
                return agent
        return None

    def find(self, target: str, *, exclude: str | None = None) -> Agent | None:
        """Resolve a name, a config name ('coder') or a role name, preferring idle agents."""
        exact = self.get(target)
        if exact is not None and exact.name != exclude and exact.available:
            return exact
        wanted = target.lower()
        candidates = [
            a
            for a in self._agents.values()
            if a.available
            and a.name != exclude
            and wanted in (a.config_name.lower(), a.role.name.lower(), base_name(a.name).lower())
        ]
        candidates.sort(key=lambda a: (a.busy, a.stats.tasks))
        return candidates[0] if candidates else None

    def set_state(self, agent: Agent, state: AgentState, activity: str = "") -> None:
        agent.state = state
        agent.activity = activity
        self.bus.publish(AgentStatusEvent(agent=agent.name, status=state.value, activity=activity, model=agent.model_ref))
