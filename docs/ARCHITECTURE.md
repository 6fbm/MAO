# Architecture — Multi AI Orchestrator (`mao`)

As of 2026-09-14

## 1. Goals and priorities

1. Real functionality — every operation shown corresponds to a real model or tool call.
2. Reliability — one failing agent or provider does not abort a session.
3. Security — work stays inside the workspace, permissions are granular, risky actions need approval,
   no secrets in logs.
4. Good agent coordination — task graph, targeted information hand-off, debate and consensus, quality loops.
5. Efficient token and cost management — budgets, context routing instead of "everything to everyone",
   compaction.
6. Extensibility — providers, tools, roles, strategies and commands through registries.
7. Speed — independent tasks run in parallel.
8. A terminal interface you can actually follow.

## 2. Technology decisions

| Area | Decision | Reason |
|---|---|---|
| Language | Python ≥ 3.11 | Best AI ecosystem support, `asyncio` for concurrency, present on the target system |
| HTTP | `httpx` (async), direct REST calls | No heavy or unstable SDK dependencies, full control over retries and errors, testable with `MockTransport` |
| Configuration | YAML + `pydantic` v2 | Readable, strictly validated (typos are caught) |
| Terminal UI | `rich` (live dashboard) + `prompt_toolkit` (input, history, completion) | Works in Windows Terminal, classic CMD and any POSIX terminal |
| Secrets | Windows DPAPI (encrypted per user) + environment variables | No plaintext keys on disk, no extra dependency |
| Tests | `pytest` + `pytest-asyncio` | Standard |

### Provider APIs (per official documentation, checked 2026-09-14)

| Provider | API | Notes |
|---|---|---|
| OpenAI | Responses API `POST /v1/responses` | `store:false`; reasoning items with `encrypted_content` are handed back inside the tool loop |
| xAI (Grok) | Responses API `POST https://api.x.ai/v1/responses` | Chat Completions is "legacy" at xAI |
| Anthropic | Messages API `POST /v1/messages` | Thinking is always on for Opus 5 / Sonnet 5; thinking blocks are returned unchanged |
| Google Gemini | `models.generateContent` | "Legacy" but fully supported; `thoughtSignature` is returned without loss |
| Ollama | native `POST /api/chat` | `num_ctx` is set explicitly; model detection via `/api/tags` + `/api/show` |
| llama.cpp / LM Studio / vLLM / OpenRouter | OpenAI-compatible Chat Completions | generic provider |

Models without native tool calling use a text-based tool protocol (```` ```tool ```` blocks).

## 3. Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│ CLI / REPL / live dashboard (mao.cli)   — display and input only     │
├──────────────────────────────────────────────────────────────────────┤
│ Orchestrator (mao.orchestration)                                     │
│   PlanningPipeline · TaskGraph · GraphScheduler · DebateEngine ·     │
│   QualityLoops · AgentAssigner · PlanEstimator · RunControl          │
├──────────────────────────────────────────────────────────────────────┤
│ Agents (mao.agents)             │ Communication (mao.messaging)      │
│   Roles · AgentManager ·        │   MessageBus · Blackboard ·        │
│   AgentRuntime (tool loop)      │   ContextRouter                    │
├─────────────────────────────────┼────────────────────────────────────┤
│ Context (mao.context)           │ Tools (mao.tools) + security       │
│   compaction, budgets           │   Registry · Executor · Sandbox ·  │
│                                 │   RiskAssessor · ApprovalGateway   │
├─────────────────────────────────┴────────────────────────────────────┤
│ LLMGateway (mao.providers.gateway)                                   │
│   model resolution · retries · key rotation · rate limits ·          │
│   fallback · token/cost accounting · budget check                    │
├──────────────────────────────────────────────────────────────────────┤
│ Providers: OpenAIResponses · Anthropic · Gemini · Ollama ·           │
│            OpenAIChat · Mock (tests/demo only, never auto-picked)    │
├──────────────────────────────────────────────────────────────────────┤
│ Core: types · error classes · EventBus · configuration · sessions    │
└──────────────────────────────────────────────────────────────────────┘
```

The core layers know nothing about the terminal. They publish **events**; the dashboard, the session
logger and the tests subscribe to them. User interaction (approvals, budget questions) goes through
narrow protocols (`ApprovalHandler`, `LimitHandler`) that the CLI implements.

## 4. Components

### 4.1 Provider → model → agent → orchestrator

- **Provider** (`providers/base.py`): a single attempt `complete(request, api_key)`, plus `list_models()`
  and `health_check()`. It translates neutral types (`ChatMessage`, `ToolSpec`, `Usage`) into the wire
  format and classifies HTTP errors (`RateLimitError`, `AuthenticationError`, `ContextLengthError`, …).
- **ModelCatalog**: models from `providers.yaml` plus models discovered at runtime (Ollama, `/v1/models`).
  References: `provider/model`, aliases (`gpt`, `gemini`, `grok`, `claude`, `local`) or `auto`.
- **ModelSelector**: picks a model for a role or task by capabilities, tier, cost, availability, privacy
  (`local_only`) and provider diversity. Can be switched off (`auto_model_selection: false`).
- **LLMGateway**: the only path to a model. Budget check → key from the `KeyPool` → rate limiter → call
  with timeout → error handling (retry with backoff and jitter, `Retry-After`, key rotation, request
  adjustment, fallback model chain) → record usage and cost → events.

### 4.2 Agents

- **Role** = capabilities + system prompt + default tools + default permissions + preferred tier.
  Built in: planner, architect, researcher, coder, reviewer, tester, debugger, security, critic,
  documentation, project_manager, performance. Freely extensible through `roles.yaml`.
- **AgentConfig** (`agents.yaml`): name, role, model, fallbacks, count (`count: 5` → `coder-1 … coder-5`),
  permission overrides, tools, own system prompt.
- **AgentRuntime**: a real tool loop — model call → execute tool calls (reading ones in parallel, writing
  ones sequentially) → results back → … until an answer or `max_steps`. Structured output is requested as
  JSON, extracted robustly, and repaired once if it fails.

### 4.3 Agent-to-agent communication

- **MessageBus**: structured messages (`sender`, `recipients`, `kind` = info/question/answer/proposal/
  critique/vote/decision/finding/result/warning, `topic`, `reply_to`), persisted as `messages.jsonl`.
- **Blackboard**: shared knowledge store (findings, research with source URLs, risks, decisions, results
  per task node).
- **ContextRouter**: decides per agent and task what goes into the prompt — decisions, directly addressed
  messages, results of dependencies, relevant findings (keyword scoring) — within a token budget. An
  agent pulls details when it needs them via the `read_board` tool (compact push, pull on demand).
- **Communication tools**: `send_message`, `post_finding`, `read_board`, `consult_agent` (a synchronous
  question to another agent, depth 1, budgeted).

### 4.4 Context management

- Tool results are truncated (head and tail kept) and secrets are redacted.
- When an agent's history crosses the threshold (75 % of the context window by default) it is **rebuilt**:
  original task + a summary of the progress (from a cheap model) + the most recent tool results. This
  keeps the history valid for every provider (Anthropic and Gemini reject edited prefixes that carry
  thinking signatures).
- A `ContextLengthError` from the provider triggers a more aggressive compaction and another attempt.

### 4.5 Orchestration

**PLAN mode** (no lasting changes — writing tools are hard-blocked, the terminal only runs approved
read-only commands):
1. Workspace profile (no LLM): file tree, languages, test commands, git status.
2. Triage (planner): understanding, complexity, which investigations are needed.
3. Investigation (in parallel, different roles): findings → blackboard.
4. Draft plan (planner): steps with dependencies, roles, risks, acceptance criteria.
5. Critique rounds (reviewer/critic/security, in parallel) → consensus check → revision (up to `max_rounds`).
6. Task graph validation (cycles, unknown dependencies, missing quality gates are added), agent
   assignment, token and cost estimate.
7. The user: run / change (feedback → revision) / discard.

**RUN mode**:
1. Checkpoint: backups of every file that will be touched (always), plus an optional git checkpoint or branch.
2. `GraphScheduler`: ready nodes run in parallel (up to `max_parallel_agents`), assignment by role,
   capabilities, permissions and load; a failure is retried with a different agent; when a node fails for
   good its dependants are skipped.
3. Node types with special handling: `decision` → debate; `test` → test-fix cycle (deterministic test run
   → debugger → coder → test again, up to `max_fix_iterations`); `review` → review cycle (reviewer and
   security → coder fixes the blocking points).
4. Final synthesis and result report (changes, tests, agents, tokens, cost, duration, git).

**Debate and consensus** (`DebateEngine`): proposals (in parallel) → critique with stance, severity and
evidence → consensus check (weighted approval ≥ threshold, no critical objections) → revision → vote
(scores, role weights) → a judge agent checks the evidence itself (it may read files and consult other
agents) and decides with a rationale. A unanimous vote skips the judge call.

### 4.6 Security

- `WorkspaceSandbox`: every path is resolved (including symlinks and junctions) and must lie inside the
  workspace; UNC and device paths, alternate data streams and reserved names are rejected; `.git` is
  protected internally.
- Permissions per agent: `read`, `write`, `delete`, `execute`, `internet`, `git` (none/read/write),
  `consult`; plus a global ceiling.
- `RiskAssessor` rates commands and actions (denylist → blocked; deletion, `git reset --hard`, downloads,
  paths outside → high).
- `ApprovalGateway`: rules per kind of action (`allow`/`ask`/`deny`/`ask_risky`), one prompt at a time,
  "always allow for this session".
- Terminal processes get an environment **without** API keys or tokens; stdin is closed; a timeout kills
  the whole process tree.
- `fetch_url` blocks private and local networks (SSRF protection).
- Secrets: DPAPI-encrypted, only fingerprints are shown, and a global `Redactor` covers logs, transcripts
  and tool output.

### 4.7 Tokens, cost, budgets

`ResourceMonitor` measures CPU and RAM (psutil) and, where available, the NVIDIA GPU (`nvidia-smi`, on a
background thread). The dashboard displays these values; above `limits.max_ram_percent` the scheduler
starts no further parallel steps (it never blocks when nothing is running). Local providers are limited
to one concurrent call by default.

`UsageTracker` records every call (agent, provider, model, input/output/cached/reasoning, cost, latency).
Cost is either estimated from the prices in `providers.yaml` or taken from the provider when it reports
real cost. `BudgetGuard` checks `max_tokens`, `max_cost_usd` and `max_llm_calls` before every call; when a
limit is hit, the run stops in a controlled way or asks (+50 %).

### 4.8 Sessions and logging

`logs/session_<date>_<time>_<id>/` contains `session.json`, `plan.json`, `messages.jsonl`, `events.jsonl`,
`blackboard.json`, `decisions.json`, `tokens.json`, `changes.json`, `backups/`, plus `orchestration.log`,
`agents.log`, `tools.log` and `errors.log`. Sessions can be loaded again; nodes that were interrupted are
restarted when the session resumes.

## 5. Error handling (example flow)

```
Gemini 503 ─► retry (backoff 1.5s, jitter) ─► attempt 2/3 ─► 503 ─► attempt 3/3 ─► 503
           ─► the agent's fallback chain: anthropic/claude-sonnet-5 ─► success
429 with several keys ─► key goes into cooldown (Retry-After) ─► next key immediately
401 ─► key marked invalid ─► next key ─► none left ─► fallback model
context too long ─► compaction ─► another attempt
agent fails for good ─► node is reassigned to another agent ─► otherwise the node is "failed",
                       its dependants are "skipped", and the rest keeps running
```

## 6. Extension points

| Extension | How |
|---|---|
| New provider | Subclass `Provider`, add `@register_provider("type")`, reference it in `providers.yaml` with `type: type` |
| New tool | Subclass `Tool` (name, JSON schema, required permissions, `run`), register it in `tools/registry.py` |
| New role | `roles.yaml` (no code) |
| New strategy | `orchestration/strategies.py` — register the strategy, select it in `settings.yaml` |
| New CLI command | A function with `@command(...)` in `cli/commands/` |
| New search backend | `tools/web.py` — register the backend function |

## 7. Directory layout

```
config/                 YAML configuration (settings, providers, agents, roles, tools, permissions)
docs/                   documentation
src/mao/
  core/                 types, errors, EventBus, JSON and text helpers
  config/               schema, loader, templates
  security/             secrets (DPAPI), redaction, sandbox, risk, permissions, approvals
  providers/            provider implementations, KeyPool, rate limits, gateway
  models/               catalog, selection, detection of local models
  tokens/               estimation, tracking, budget
  tools/                tool system (filesystem, terminal, web, git, tests, communication)
  workspace/            workspace profile, change tracking and backups, git helpers
  messaging/            MessageBus, Blackboard, ContextRouter
  context/              compaction
  agents/               roles, agent runtime, AgentManager
  orchestration/        planner, task graph, scheduler, debate, quality loops, orchestrator
  sessions/             session store, session logger
  cli/                  REPL, commands, dashboard
tests/                  unit, integration and end-to-end tests
```
