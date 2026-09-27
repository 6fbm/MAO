# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Providers for DeepSeek, Qwen (Alibaba Model Studio), Kimi (Moonshot AI), Mistral AI, GLM (Z.ai),
  Groq and Together AI. All of them speak the OpenAI-compatible protocol, so they are configuration
  only - no new provider code. Endpoints and prices checked on 2026-09-27.
- `install.sh` and `mao.sh` as the Linux/macOS counterparts to `install.cmd` and `mao.cmd`.
- CI on GitHub Actions: the test suite on Linux, macOS and Windows against Python 3.11–3.14.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE` (MIT) and a vulnerability reporting section in
  `docs/SECURITY.md`.
- English `README.md`.

### Changed
- The project is English-only: the README, the documentation, the configuration templates and
  every user-facing string in the CLI were translated from German. `response_language` now
  defaults to `English`; set it back to any language you want your agents to answer in.
- The local `config/` is no longer tracked by git. It is generated on first start from
  `src/mao/config/templates/`, which is the versioned source.

## [0.1.0]

### Added
- PLAN mode: triage, parallel investigations, draft plan, critique rounds with consensus check, judge
  decision, plan with risks, agent assignment and token/cost estimate.
- RUN mode: task graph execution with parallelism, test→debug→fix and review→fix cycles, debates,
  final report.
- Providers: OpenAI (Responses API), xAI Grok, Anthropic Claude, Google Gemini, Ollama, and
  OpenAI-compatible endpoints (llama.cpp, LM Studio, vLLM, OpenRouter …), with several API keys per
  provider, rotation, retries and fallback models.
- Agents and roles configurable via YAML, several instances per model, agent-to-agent messages and a
  shared blackboard.
- Security: workspace sandbox, per-agent permissions, approval dialogs for risky actions, secret
  redaction in logs, DPAPI-encrypted key store on Windows.
- Token and cost tracking with budgets, resumable sessions, detailed logs, backups with rollback and
  git checkpoints.
- Live dashboard with keyboard control, and an offline demo mode (`--demo`).
