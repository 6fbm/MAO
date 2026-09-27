"""Pydantic models for every configuration file.

``extra="forbid"`` everywhere: a typo in YAML must produce a clear error
instead of silently being ignored.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# =========================================================================== permissions

FilesystemLevel = Literal["none", "read", "write", "full"]
GitLevel = Literal["none", "read", "write"]
ApprovalRule = Literal["allow", "ask", "deny"]

_FS_ORDER = {"none": 0, "read": 1, "write": 2, "full": 3}
_GIT_ORDER = {"none": 0, "read": 1, "write": 2}


class PermissionSet(StrictModel):
    """Fully resolved permissions of one agent."""

    read: bool = True
    write: bool = False
    delete: bool = False
    execute: bool = False
    internet: bool = False
    git: GitLevel = "read"
    consult: bool = True

    def describe(self) -> str:
        parts = [name for name in ("read", "write", "delete", "execute", "internet") if getattr(self, name)]
        parts.append(f"git:{self.git}")
        if self.consult:
            parts.append("consult")
        return ", ".join(parts)


class PermissionOverrides(StrictModel):
    """Partial permissions as written in YAML.

    ``filesystem`` is a shorthand (none/read/write/full); explicit booleans win.
    ``terminal`` is an alias for ``execute``.
    """

    filesystem: FilesystemLevel | None = None
    read: bool | None = None
    write: bool | None = None
    delete: bool | None = None
    execute: bool | None = None
    terminal: bool | None = None
    internet: bool | None = None
    git: GitLevel | None = None
    consult: bool | None = None

    def apply(self, base: PermissionSet) -> PermissionSet:
        data = base.model_dump()
        if self.filesystem is not None:
            data["read"] = self.filesystem != "none"
            data["write"] = self.filesystem in ("write", "full")
            data["delete"] = self.filesystem == "full"
        for name in ("read", "write", "delete", "execute", "internet", "git", "consult"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.terminal is not None:
            data["execute"] = self.terminal
        if data["write"] or data["delete"]:
            data["read"] = True
        return PermissionSet(**data)

    def restrict(self, perms: PermissionSet) -> PermissionSet:
        """Use these overrides as a ceiling: they may only take rights away."""
        data = perms.model_dump()
        if self.filesystem is not None:
            level = self.filesystem
            if _FS_ORDER[level] < 1:
                data["read"] = False
            if _FS_ORDER[level] < 2:
                data["write"] = False
            if _FS_ORDER[level] < 3:
                data["delete"] = False
        for name in ("read", "write", "delete", "execute", "internet", "consult"):
            value = getattr(self, name)
            if value is False:
                data[name] = False
        if self.terminal is False:
            data["execute"] = False
        if self.git is not None and _GIT_ORDER[self.git] < _GIT_ORDER[data["git"]]:
            data["git"] = self.git
        return PermissionSet(**data)


class ApprovalRules(StrictModel):
    create: ApprovalRule = "allow"
    overwrite: ApprovalRule = "allow"
    edit: ApprovalRule = "allow"
    mkdir: ApprovalRule = "allow"
    move: ApprovalRule = "ask"
    delete: ApprovalRule = "ask"
    execute: Literal["allow", "ask", "ask_risky", "deny"] = "ask_risky"
    internet: ApprovalRule = "allow"
    git_write: ApprovalRule = "allow"
    git_destructive: ApprovalRule = "ask"
    sensitive_read: ApprovalRule = "ask"
    outside_workspace: ApprovalRule = "deny"


class PermissionsConfig(StrictModel):
    defaults: PermissionOverrides = Field(
        default_factory=lambda: PermissionOverrides(filesystem="read", execute=False, internet=False, git="read")
    )
    ceiling: PermissionOverrides | None = None
    approval: ApprovalRules = Field(default_factory=ApprovalRules)
    auto_approve: bool = False
    sensitive_patterns: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "*.pfx",
            "*.p12",
            "id_rsa*",
            "id_ed25519*",
            "*secret*",
            "*credential*",
            ".npmrc",
            ".pypirc",
        ]
    )
    protected_patterns: list[str] = Field(default_factory=lambda: [".git", ".git/**"])
    command_allowlist: list[str] = Field(
        default_factory=lambda: [
            "python -m pytest*",
            "pytest*",
            "py -m pytest*",
            "npm test*",
            "npm run test*",
            "npm run lint*",
            "npx tsc*",
            "cargo test*",
            "cargo check*",
            "go test*",
            "go vet*",
            "dotnet test*",
            "dotnet build*",
            "git status*",
            "git diff*",
            "git log*",
            "python -m py_compile*",
            "ruff*",
            "mypy*",
        ]
    )
    command_denylist: list[str] = Field(
        default_factory=lambda: [
            "format *",
            "diskpart*",
            "shutdown*",
            "restart-computer*",
            "stop-computer*",
            "bcdedit*",
            "reg delete*",
            "vssadmin*",
            "cipher /w*",
            "rm -rf /*",
            "rm -rf ~*",
            "mkfs*",
            "dd if=*",
            "git push*--force*",
            "git push -f*",
        ]
    )
    plan_mode_commands: list[str] = Field(
        default_factory=lambda: [
            "git status*",
            "git log*",
            "git diff*",
            "git branch*",
            "git show*",
            "python --version",
            "py --version",
            "node --version",
            "npm --version",
            "pip list*",
            "pip show*",
            "python -m pip list*",
            "dotnet --version",
            "go version",
            "cargo --version",
        ]
    )
    scrub_env_patterns: list[str] = Field(
        default_factory=lambda: ["*API_KEY*", "*APIKEY*", "*_TOKEN", "*TOKEN_*", "*SECRET*", "*PASSWORD*", "*PASSWD*"]
    )


# =========================================================================== providers

ModelTier = Literal["frontier", "strong", "balanced", "fast"]
ToolCallingMode = Literal["native", "prompt", "none"]

TIER_RANK: dict[str, int] = {"fast": 1, "balanced": 2, "strong": 3, "frontier": 4}

KNOWN_MODEL_CAPABILITIES = frozenset(
    {"tools", "coding", "reasoning", "vision", "web_search", "long_context", "json", "research", "local"}
)


class PricingConfig(StrictModel):
    """USD per 1M tokens. Values are estimates – verify with the provider."""

    input_per_mtok: float = Field(0.0, ge=0)
    output_per_mtok: float = Field(0.0, ge=0)
    cached_input_per_mtok: float | None = Field(None, ge=0)
    long_context_threshold: int | None = Field(None, ge=1)
    long_input_per_mtok: float | None = Field(None, ge=0)
    long_output_per_mtok: float | None = Field(None, ge=0)

    @property
    def is_free(self) -> bool:
        return self.input_per_mtok == 0 and self.output_per_mtok == 0


class ModelConfig(StrictModel):
    id: str | None = None  # API model id; defaults to the YAML key
    display_name: str | None = None
    enabled: bool = True
    context_window: int = Field(128_000, ge=1_024)
    max_output_tokens: int = Field(8_192, ge=16)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    tier: ModelTier = "balanced"
    capabilities: list[str] = Field(default_factory=lambda: ["tools"])
    tool_calling: ToolCallingMode = "native"
    supports_temperature: bool = True
    reasoning: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ApiKeysConfig(StrictModel):
    env: list[str] = Field(default_factory=list)
    use_secret_store: bool = True


class RateLimitConfig(StrictModel):
    max_concurrency: int = Field(4, ge=1, le=256)
    requests_per_minute: int | None = Field(None, ge=1)
    tokens_per_minute: int | None = Field(None, ge=1)


class ProviderConfig(StrictModel):
    type: str
    display_name: str | None = None
    enabled: bool = True
    base_url: str | None = None
    requires_api_key: bool = True
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    local: bool = False
    selectable: bool = True  # may the auto selector pick this provider?
    timeout_s: float = Field(180.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    headers: dict[str, str] = Field(default_factory=dict)
    auto_discover: bool = False
    default_model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)


class ProvidersFile(StrictModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


# =========================================================================== roles & agents

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,47}$")


class RoleConfig(StrictModel):
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=lambda: ["filesystem", "collaboration"])
    permissions: PermissionOverrides = Field(default_factory=PermissionOverrides)
    preferred_tier: ModelTier = "balanced"
    required_model_capabilities: list[str] = Field(default_factory=list)
    max_steps: int | None = Field(None, ge=1)


class RolesFile(StrictModel):
    roles: dict[str, RoleConfig] = Field(default_factory=dict)


class AgentConfig(StrictModel):
    name: str
    role: str
    model: str = "auto"
    fallback_models: list[str] = Field(default_factory=list)
    count: int = Field(1, ge=1, le=100)
    enabled: bool = True
    permissions: PermissionOverrides | None = None
    tools: list[str] | None = None
    capabilities: list[str] | None = None
    system_prompt: str | None = None
    extra_instructions: str | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    max_steps: int | None = Field(None, ge=1, le=500)
    max_output_tokens: int | None = Field(None, ge=64)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError("The name may only contain letters, digits, '_', '-' and '.' (max. 48 characters)")
        return value


class AgentsFile(StrictModel):
    agents: list[AgentConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> AgentsFile:
        seen: set[str] = set()
        for agent in self.agents:
            key = agent.name.lower()
            if key in seen:
                raise ValueError(f"Duplicate agent name: {agent.name}")
            seen.add(key)
        return self


# =========================================================================== tools

SearchBackend = Literal["auto", "duckduckgo", "brave", "tavily", "searxng", "gemini"]


class FilesystemToolConfig(StrictModel):
    max_read_chars: int = Field(60_000, ge=1_000)
    max_file_bytes: int = Field(5_000_000, ge=1_000)
    max_list_entries: int = Field(400, ge=10)
    max_search_results: int = Field(200, ge=1)
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            ".idea",
            ".vs",
            "*.pyc",
            ".DS_Store",
        ]
    )


class TerminalToolConfig(StrictModel):
    shell: Literal["auto", "cmd", "powershell", "pwsh", "bash", "sh"] = "auto"
    default_timeout_s: int = Field(120, ge=1)
    max_timeout_s: int = Field(1_800, ge=1)
    max_output_chars: int = Field(16_000, ge=500)
    env_passthrough: list[str] = Field(default_factory=list)


class WebToolConfig(StrictModel):
    search_backend: SearchBackend = "auto"
    searxng_url: str | None = None
    brave_api_key_env: str = "BRAVE_API_KEY"
    tavily_api_key_env: str = "TAVILY_API_KEY"
    gemini_search_model: str | None = None
    max_results: int = Field(6, ge=1, le=20)
    fetch_max_chars: int = Field(20_000, ge=500)
    fetch_max_bytes: int = Field(3_000_000, ge=10_000)
    timeout_s: float = Field(20.0, gt=0)
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MultiAIOrchestrator/0.1"
    allow_private_networks: bool = False


class TestsToolConfig(StrictModel):
    command: str | None = None
    timeout_s: int = Field(900, ge=5)


class GitToolConfig(StrictModel):
    enabled: bool = True
    auto_checkpoint: bool = True
    auto_branch: bool = False
    branch_prefix: str = "mao/"
    auto_commit: bool = False


class ToolsConfig(StrictModel):
    enabled_groups: list[str] = Field(
        default_factory=lambda: ["filesystem", "terminal", "web", "git", "tests", "collaboration"]
    )
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)
    terminal: TerminalToolConfig = Field(default_factory=TerminalToolConfig)
    web: WebToolConfig = Field(default_factory=WebToolConfig)
    tests: TestsToolConfig = Field(default_factory=TestsToolConfig)
    git: GitToolConfig = Field(default_factory=GitToolConfig)


# =========================================================================== settings


class LimitsConfig(StrictModel):
    max_tokens: int | None = Field(2_000_000, ge=1_000)
    max_cost_usd: float | None = Field(5.0, ge=0)
    max_llm_calls: int | None = Field(None, ge=1)
    max_agents: int = Field(20, ge=1, le=500)
    max_rounds: int = Field(3, ge=1, le=20)
    max_parallel_agents: int = Field(4, ge=1, le=64)
    max_steps_per_task: int = Field(30, ge=2, le=500)
    max_fix_iterations: int = Field(3, ge=0, le=20)
    max_review_iterations: int = Field(2, ge=0, le=10)
    max_node_attempts: int = Field(2, ge=1, le=5)
    max_consultations_per_task: int = Field(3, ge=0, le=20)
    max_ram_percent: float | None = Field(92.0, ge=10, le=100)
    on_limit: Literal["ask", "stop"] = "ask"
    soft_limit_ratio: float = Field(0.8, gt=0, lt=1)


class ContextConfig(StrictModel):
    compaction_threshold: float = Field(0.75, gt=0.1, lt=1)
    max_tool_result_chars: int = Field(12_000, ge=500)
    max_shared_context_tokens: int = Field(12_000, ge=500)
    default_max_output_tokens: int = Field(32_000, ge=256)
    summary_model: str = "auto"


class OrchestrationConfig(StrictModel):
    strategy: str = "standard"
    base_rounds: int = Field(2, ge=1, le=20)  # discussion rounds without mode influence (capped by limits.max_rounds)
    auto_model_selection: bool = True
    model_policy: Literal["quality", "balanced", "cheap"] = "balanced"
    orchestrator_model: str = "auto"
    debate_enabled: bool = True
    consensus_threshold: float = Field(0.67, gt=0, le=1)
    investigation_max_agents: int = Field(4, ge=0, le=20)
    max_critics: int = Field(3, ge=0, le=20)
    auto_add_quality_gates: bool = True
    require_plan_approval: bool = True
    vote_weights: dict[str, float] = Field(
        default_factory=lambda: {"security": 1.3, "architect": 1.2, "reviewer": 1.1}
    )


class UIConfig(StrictModel):
    refresh_per_second: int = Field(4, ge=1, le=30)
    show_banner: bool = True
    debug: bool = False
    max_event_lines: int = Field(10, ge=3, le=100)


class PrivacyConfig(StrictModel):
    local_only: bool = False


class ModesSettings(StrictModel):
    default: list[str] = Field(default_factory=lambda: ["productive"])  # z. B. ["productive:80", "strict:60"]
    orchestrator: str = "balanced"
    max_active: int = Field(6, ge=1, le=12)
    voice: bool = True  # personality comments from the orchestrator in the interface


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Settings(StrictModel):
    workspace: str | None = None
    response_language: str = "English"
    logs_dir: str = "logs"
    secrets_file: str | None = None
    aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "gpt": "openai",
            "chatgpt": "openai",
            "gemini": "gemini",
            "grok": "xai",
            "claude": "anthropic",
            "local": "ollama",
            "llama": "ollama",
        }
    )
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    modes: ModesSettings = Field(default_factory=ModesSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class AppConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    settings: Settings = Field(default_factory=Settings)
    providers: ProvidersFile = Field(default_factory=ProvidersFile)
    agents: AgentsFile = Field(default_factory=AgentsFile)
    roles: RolesFile = Field(default_factory=RolesFile)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
