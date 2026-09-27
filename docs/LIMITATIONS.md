# Known limits and deliberate decisions

As of 2026-09-14

## Providers

- **Cloud APIs without a live test during development**: no API keys were available while this was
  built. OpenAI/xAI (Responses API), Anthropic (Messages API) and Gemini (generateContent) are
  implemented according to the official documentation, and the wire formats are covered by unit tests.
  Run `/providers test` and a small task before using them for real work. Deviations in individual
  parameters are partly corrected at runtime (e.g. an unsupported `temperature`, `include`, JSON mode, or
  a history without thinking/reasoning signatures).
- **Gemini** uses `generateContent` (labelled "legacy" by Google, but fully supported). The newer
  Interactions API is not wired up because its documentation was incomplete at the time of development;
  an additional provider can be added through the provider interface.
- **Server-side tools** offered by the vendors (e.g. web search in the Responses API) are not used;
  research goes through this project's own `web_search`/`fetch_url` tools (optionally with Gemini
  grounding as the search backend).
- **No streaming**: replies appear once a call completes; the dashboard shows states, not token by token.
- **Prices and context limits** in `providers.yaml` are snapshots (some marked as assumptions, e.g.
  long-context thresholds at xAI, context limits of new Gemini models). Cost figures are estimates
  unless a provider reports the real cost.
- **Token estimates before a call** are heuristic (≈ 3.5 characters per token); after the call the
  numbers reported by the provider are used.

## Orchestration

- Quality and cost depend heavily on the models in use. Small local models often fail to follow JSON
  formats reliably; the system repairs and retries, but will mark a result as incomplete if it has to.
- The plan estimate (tokens/cost) is based on typical step sizes, not on a simulation.
- Agent follow-up questions (`consult_agent`) have depth 1 and a limit per task, to avoid cost explosions.
- Resource management: concurrency per provider (`max_concurrency`, 1 by default for local ones), rate
  limits, budgets, context windows, and a RAM threshold before further parallel steps are started. CPU
  and GPU utilisation is displayed but not used for control; local servers (Ollama) manage VRAM
  themselves.

## Security

- The command risk assessment is a heuristic — not a shell parser and not a sandbox. Commands run with
  the permissions of the user. For strong isolation, use a VM, a container or the Windows Sandbox.
- The SSRF protection of `fetch_url` checks resolved addresses before every request; DNS rebinding
  between the check and the connection is theoretically possible.
- A program may be started from an absolute path outside the workspace (e.g. a Python interpreter); its
  arguments may not point to paths outside without counting as high risk.

## Interface

- The live view needs a real console (Windows Terminal or CMD on Windows, any terminal elsewhere). In
  pipes, in CI or with `--plain`, output is line by line; approvals are then answered over stdin (no
  input means reject).
- Configuration changes generally take effect from the next session; limits (`/max-*`) apply immediately.
- When complex YAML structures (lists, new keys) are written back, the comments of that file are lost;
  simple values are replaced while keeping comments.

## Web search

- Without an API key, DuckDuckGo's HTML search is used. It is unofficial, it may refuse requests, and
  this is not worked around. For reliable research, configure `BRAVE_API_KEY`, `TAVILY_API_KEY` or your
  own SearXNG instance.
