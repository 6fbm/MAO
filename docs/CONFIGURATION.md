# Configuration

All files live in `config/` inside the program folder (`--home`, the `MAO_HOME` environment variable, or
the installation folder). Missing files are created from templates at start (`mao init`; `mao init
--force` resets them). Every file is validated strictly with pydantic: unknown keys and invalid values
produce a clear error message including the path.

Strings may contain environment variables: `${NAME}` or `${NAME:-default}`.
Changes made through the CLI (`/config set`, `/max-cost`, `/workspace`, `/agents …`) are written back to
the files; simple values are replaced while keeping the comments.

## settings.yaml

| Key | Meaning |
|---|---|
| `workspace` | default workspace |
| `response_language` | language for plans, reports and summaries |
| `logs_dir` | where sessions are stored (relative to the program folder, or absolute) |
| `secrets_file` | path of the encrypted key file (default `%LOCALAPPDATA%\MultiAIOrchestrator\secrets.dat` on Windows, `~/.config/mao/secrets.dat` elsewhere) |
| `aliases` | short names → provider or `provider/model` |
| `limits.max_tokens` / `max_cost_usd` / `max_llm_calls` | budgets per session (`null` = unlimited) |
| `limits.max_agents` | maximum number of agent instances |
| `limits.max_rounds` | discussion rounds for plan critique and debates |
| `limits.max_parallel_agents` | agents working at the same time |
| `limits.max_steps_per_task` | tool steps per agent task |
| `limits.max_fix_iterations` / `max_review_iterations` | quality loops |
| `limits.max_node_attempts` | attempts per plan step (with a different agent) |
| `limits.max_consultations_per_task` | follow-up questions one agent may ask per task |
| `limits.max_ram_percent` | above this RAM usage, no further steps are started in parallel |
| `limits.on_limit` | `ask` or `stop` |
| `context.compaction_threshold` | share of the context window at which compaction starts |
| `context.max_tool_result_chars` | truncation of long tool results |
| `context.max_shared_context_tokens` | budget for shared context per task |
| `context.default_max_output_tokens` | output cap per model call |
| `context.summary_model` | model used for compaction (`auto` = cheap/fast) |
| `orchestration.auto_model_selection` | automatic model choice on/off |
| `orchestration.model_policy` | `quality`, `balanced`, `cheap` |
| `orchestration.orchestrator_model` | model of the internal orchestrator agent (triage, judge) |
| `orchestration.debate_enabled` | critique rounds and debates |
| `orchestration.consensus_threshold` | weighted share of approval needed for consensus |
| `orchestration.investigation_max_agents` | parallel investigations in PLAN mode |
| `orchestration.max_critics` | critics per round |
| `orchestration.auto_add_quality_gates` | add test/review/final steps automatically |
| `orchestration.require_plan_approval` | RUN only with a confirmed plan |
| `orchestration.vote_weights` | vote weights per role |
| `privacy.local_only` | use local models only |
| `ui.*` | refresh rate, banner, debug, number of event lines |
| `logging.level` | level of detail in the log files |

## providers.yaml

```yaml
providers:
  <name>:
    type: openai_responses | openai_chat | anthropic | gemini | ollama | mock
    enabled: true
    base_url: https://…
    requires_api_key: true
    api_keys:
      env: [OPENAI_API_KEY, OPENAI_API_KEY_2]   # environment variables (several keys → rotation)
      use_secret_store: true                    # also use keys from /providers key add
    local: false            # local server (privacy, local_only)
    selectable: true        # may the automatic selection use this provider?
    timeout_s: 180
    max_retries: 3
    rate_limit: {max_concurrency: 4, requests_per_minute: null, tokens_per_minute: null}
    headers: {}             # additional, non-secret headers
    auto_discover: false    # detect models through the API at start
    default_model: …
    options: {}             # provider specific, e.g. num_ctx/keep_alive (Ollama), anthropic_version, max_tokens_param (openai_chat)
    models:
      <key>:
        id: <api-model-id>           # default: the key
        context_window: 128000
        max_output_tokens: 8192
        tier: fast | balanced | strong | frontier
        capabilities: [tools, coding, reasoning, vision, web_search, long_context, json]
        tool_calling: native | prompt | none
        supports_temperature: true
        reasoning: false             # OpenAI: request encrypted reasoning items
        pricing: {input_per_mtok: 0, output_per_mtok: 0, cached_input_per_mtok: null,
                  long_context_threshold: null, long_input_per_mtok: null, long_output_per_mtok: null}
        extra_body: {}               # copied into the request unchanged (e.g. reasoning: {effort: low})
```

Example of an additional OpenAI-compatible service:

```yaml
  groq:
    type: openai_chat
    base_url: https://api.groq.com/openai/v1
    api_keys: {env: [GROQ_API_KEY]}
    models:
      llama-large:
        id: <model-id-as-documented-by-the-vendor>
        context_window: 128000
        tier: balanced
        capabilities: [tools, coding]
```

## agents.yaml

```yaml
agents:
  - name: coder             # instances: coder-1 … coder-N when count > 1
    role: coder
    model: auto             # auto | provider | provider/model | alias
    fallback_models: []     # empty = automatic
    count: 2
    enabled: true
    permissions: {filesystem: write, terminal: true, internet: false, git: read}
    tools: [filesystem, terminal, git, tests, collaboration]
    capabilities: null      # overrides the capabilities of the role
    system_prompt: null     # replaces the role prompt
    extra_instructions: null
    temperature: null
    max_steps: null
    max_output_tokens: null
```

## roles.yaml

New roles, or adjustments to built-in ones (only the fields you give are overridden): `description`,
`capabilities`, `system_prompt`, `tools`, `permissions`, `preferred_tier`,
`required_model_capabilities`, `max_steps`. The capabilities drive how work is distributed (e.g.
`coding` for implementation, `testing`, `review`, `security`, `research`).

## tools.yaml

| Area | Important keys |
|---|---|
| `enabled_groups` | globally active tool groups: filesystem, terminal, web, git, tests, collaboration |
| `filesystem` | `max_read_chars`, `max_file_bytes`, `max_list_entries`, `ignore_patterns` |
| `terminal` | `shell` (auto/cmd/powershell/pwsh/bash/sh), timeouts, `max_output_chars`, `env_passthrough` |
| `web` | `search_backend` (auto/duckduckgo/brave/tavily/searxng/gemini), `searxng_url`, fetch limits, `allow_private_networks` |
| `tests` | `command` (null = detect automatically), `timeout_s` |
| `git` | `auto_checkpoint`, `auto_branch`, `branch_prefix`, `auto_commit` |

## permissions.yaml

| Key | Meaning |
|---|---|
| `defaults` | base permissions of every agent (then the role, then the agent) |
| `ceiling` | global ceiling; it can only take permissions away |
| `approval.<action>` | `allow`/`ask`/`deny` for create, overwrite, edit, mkdir, move, delete, internet, git_write, git_destructive, sensitive_read, outside_workspace; `execute` additionally supports `ask_risky` |
| `auto_approve` | grant every `ask` approval automatically (critical actions excepted) |
| `sensitive_patterns` | files that need approval to be read and are never searched |
| `protected_patterns` | never writable (default `.git`) |
| `command_allowlist` | commands that run without asking (only without chaining or redirection) |
| `command_denylist` | always blocked |
| `plan_mode_commands` | read-only commands allowed in PLAN mode |
| `scrub_env_patterns` | environment variables removed from terminal processes |
