"""The agent tool loop: model call -> tool calls -> results -> ... -> final answer."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from mao.agents.agent import Agent, AgentState, AgentTask, AgentTaskResult
from mao.agents.manager import AgentManager
from mao.config.schema import PermissionsConfig, Settings, ToolsConfig
from mao.context.compaction import ConversationCompactor
from mao.core.errors import (
    BudgetExceededError,
    ContextLengthError,
    OperationCancelled,
    ProviderError,
    ProviderUnavailableError,
)
from mao.core.events import EventBus
from mao.core.jsonutil import extract_json
from mao.core.types import ChatMessage, CompletionRequest, FinishReason, RunMode, ToolCall, Usage
from mao.messaging.router import ContextRouter, RoutingRequest
from mao.orchestration.control import RunControl
from mao.providers.gateway import LLMGateway
from mao.security.approval import ApprovalGateway
from mao.security.sandbox import WorkspaceSandbox
from mao.tools.base import CollaborationPort, ToolContext
from mao.tools.executor import ToolExecutor
from mao.workspace.changes import ChangeTracker
from mao.workspace.git import GitRepo

MAX_JSON_REPAIRS = 2
MAX_LENGTH_CONTINUATIONS = 2
MAX_CONTEXT_RETRIES = 2
READONLY_GROUPS = {"filesystem", "git", "web", "collaboration"}


@dataclass
class RuntimeServices:
    gateway: LLMGateway
    executor: ToolExecutor
    router: ContextRouter
    compactor: ConversationCompactor
    manager: AgentManager
    bus: EventBus
    control: RunControl
    settings: Settings
    tools_config: ToolsConfig
    permissions_config: PermissionsConfig
    sandbox: WorkspaceSandbox
    approval: ApprovalGateway
    changes: ChangeTracker
    http: httpx.AsyncClient | None = None
    collaboration: CollaborationPort | None = None
    git: GitRepo | None = None
    web: object | None = None
    on_test_result: object | None = None
    mode: RunMode = RunMode.PLAN
    extras: dict = field(default_factory=dict)


def _base_prompt(agent: Agent, mode: RunMode, workspace: Path, language: str) -> str:
    if mode is RunMode.PLAN:
        mode_rules = (
            "You are in PLAN mode: do NOT create, modify, move or delete anything and do not run state-changing commands. "
            "Only read, analyze and report."
        )
    else:
        mode_rules = (
            "You are in RUN mode: you may change the workspace within your permissions. Make focused, minimal changes, keep "
            "the project working, and never delete data unless the task clearly requires it."
        )
    return f"""You are "{agent.name}", role "{agent.role.name}", in a multi-agent AI software team coordinated by an orchestrator.
Environment: {sys.platform} machine, workspace root: {workspace}. All relative paths are relative to that root.

Rules:
- Work only inside the workspace. Inspect real files with tools instead of guessing. Never invent files, APIs, results or test outcomes.
- Be concise and factual. Support statements with evidence (file paths with line numbers, command output, URLs).
- Share findings other agents need via post_finding; use send_message for handovers. Read the shared context you receive.
- If something fails or is impossible, say so clearly and explain why.
- Write all human-readable text (summaries, reports, plan items) in {language}. Keep code and identifiers unchanged.
- {mode_rules}"""


class AgentRuntime:
    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    # ------------------------------------------------------------------ prompt building

    def _tool_context(self, agent: Agent, task: AgentTask) -> ToolContext:
        s = self.services
        mode = RunMode.PLAN if task.readonly else s.mode
        return ToolContext(
            agent_name=agent.name,
            agent_role=agent.role.name,
            permissions=agent.permissions,
            mode=mode,
            sandbox=s.sandbox,
            approval=s.approval,
            changes=s.changes,
            bus=s.bus,
            tools_config=s.tools_config,
            permissions_config=s.permissions_config,
            http=s.http,
            collaboration=s.collaboration,
            git=s.git,
            web=s.web,  # type: ignore[arg-type]
            node_id=task.node_id,
            consult_depth=task.consult_depth,
            on_test_result=s.on_test_result,  # type: ignore[arg-type]
        )

    def system_prompt(self, agent: Agent, ctx: ToolContext, tool_names: list[str]) -> str:
        s = self.services
        parts = [
            _base_prompt(agent, ctx.mode, s.sandbox.root, s.settings.response_language),
            f"## Your role: {agent.role.name}\n{agent.system_prompt}",
        ]
        if agent.extra_instructions:
            parts.append(f"## Additional instructions\n{agent.extra_instructions.strip()}")
        if tool_names:
            parts.append(
                "## Tools\nAvailable: " + ", ".join(tool_names) + ".\n"
                "Read files before editing them. Prefer edit_file for small changes. Batch independent read-only tool calls."
            )
        else:
            parts.append("## Tools\nNo tools are available for this task; answer from the provided context.")
        if agent.notes:
            parts.append("## Notes from your earlier tasks in this session\n" + "\n".join(f"- {n}" for n in agent.notes[-5:]))
        return "\n\n".join(parts)

    def task_prompt(self, agent: Agent, task: AgentTask) -> str:
        s = self.services
        parts: list[str] = []
        if task.include_shared_context:
            shared = s.router.build(
                RoutingRequest(
                    agent_name=agent.name,
                    role=agent.role.name,
                    capabilities=agent.capabilities,
                    query_text=task.shared_context_query or f"{task.title} {task.instructions[:500]}",
                    node_id=task.node_id,
                    depends_on=task.depends_on,
                    budget_tokens=s.settings.context.max_shared_context_tokens,
                )
            )
            if shared:
                parts.append(shared)
        parts.append(f"## Your task: {task.title}\n{task.instructions.strip()}")
        if task.output == "json":
            schema = task.output_schema or "{}"
            parts.append(
                "## Output format\nWhen you are finished, reply with ONE JSON object (no other text) with this structure:\n"
                f"{schema}\nWrite human-readable string values in {s.settings.response_language}."
            )
        return "\n\n".join(parts)

    def _max_output(self, agent: Agent) -> int:
        return agent.max_output_tokens or self.services.settings.context.default_max_output_tokens

    # ------------------------------------------------------------------ execution

    async def _execute_calls(self, calls: list[ToolCall], ctx: ToolContext, allowed: set[str]) -> list[ChatMessage]:
        executor = self.services.executor
        tools = [executor.registry.get(call.name) for call in calls]
        if len(calls) > 1 and all(t is not None and t.parallel_safe for t in tools):
            results = await asyncio.gather(*(executor.execute(call, ctx, allowed) for call in calls))
        else:
            results = []
            for call in calls:
                await self.services.control.checkpoint()
                results.append(await executor.execute(call, ctx, allowed))
        return [
            ChatMessage.tool_result(call, result.content or "(empty)", is_error=not result.ok)
            for call, result in zip(calls, results, strict=True)
        ]

    def _set_state(self, agent: Agent, state: AgentState, activity: str, track: bool) -> None:
        if track:
            self.services.manager.set_state(agent, state, activity)

    async def run(self, agent: Agent, task: AgentTask, *, track_state: bool = True) -> AgentTaskResult:
        s = self.services
        if not agent.available:
            return AgentTaskResult(task_id=task.id, agent=agent.name, status="failed", error=f"Agent offline: {agent.offline_reason}")
        ctx = self._tool_context(agent, task)
        groups = list(agent.tool_groups)
        if task.tool_groups is not None:
            groups = [g for g in groups if g in task.tool_groups]
        if task.readonly:
            groups = [g for g in groups if g in READONLY_GROUPS]
        tools = s.executor.tools_for(ctx, groups)
        if task.consult_depth >= 1:
            tools = [t for t in tools if t.name != "consult_agent"]
        specs = [t.spec() for t in tools]
        allowed = {t.name for t in tools}
        system = self.system_prompt(agent, ctx, sorted(allowed))
        messages: list[ChatMessage] = [ChatMessage.user(self.task_prompt(agent, task))]
        max_steps = task.max_steps or agent.max_steps
        usage = Usage()
        partial_text: list[str] = []
        tool_call_count = 0
        json_repairs = length_continues = context_retries = 0
        model_used: str | None = None
        if track_state:
            agent.busy = True
        agent.stats.tasks += 1
        self._set_state(agent, AgentState.THINKING, task.title, track_state)
        try:
            step = 0
            while step < max_steps:
                step += 1
                await s.control.checkpoint()
                messages = await s.compactor.maybe_compact(agent, system, messages, specs)
                request = CompletionRequest(
                    model="",
                    system=system,
                    messages=messages,
                    tools=specs,
                    temperature=agent.temperature,
                    max_output_tokens=self._max_output(agent),
                    json_mode=task.output == "json" and not specs,
                    metadata={
                        "agent": agent.name,
                        "role": agent.role.name,
                        "purpose": task.purpose,
                        "task_title": task.title,
                        "node_id": task.node_id,
                        "step": step,
                        "output": task.output,
                    },
                )
                self._set_state(agent, AgentState.THINKING, task.title, track_state)
                try:
                    result = await s.gateway.complete(
                        request, model_ref=agent.model_ref or "", agent=agent.name, purpose=task.purpose, fallbacks=agent.fallback_refs
                    )
                except ContextLengthError:
                    if context_retries >= MAX_CONTEXT_RETRIES:
                        raise
                    context_retries += 1
                    step -= 1
                    messages = await s.compactor.maybe_compact(agent, system, messages, specs, force=True, aggressive=True)
                    continue
                agent.stats.llm_calls += 1
                response = result.response
                model_used = result.entry.ref
                usage = usage + response.usage
                messages.append(response.message)

                if response.message.tool_calls:
                    tool_call_count += len(response.message.tool_calls)
                    agent.stats.tool_calls += len(response.message.tool_calls)
                    names = ", ".join(c.name for c in response.message.tool_calls)
                    self._set_state(agent, AgentState.TOOL, f"{task.title}: {names}", track_state)
                    messages.extend(await self._execute_calls(response.message.tool_calls, ctx, allowed))
                    continue

                text = response.message.content or ""
                if response.finish_reason is FinishReason.LENGTH and length_continues < MAX_LENGTH_CONTINUATIONS:
                    length_continues += 1
                    partial_text.append(text)
                    messages.append(ChatMessage.user("Your answer was cut off by the output limit. Continue exactly where you stopped, without repeating."))
                    continue
                full_text = "".join(partial_text) + text
                if task.output == "json":
                    data = extract_json(full_text)
                    if data is None and json_repairs < MAX_JSON_REPAIRS:
                        json_repairs += 1
                        partial_text = []
                        messages.append(ChatMessage.user("Your reply did not contain valid JSON. Reply with ONLY the JSON object in the required structure."))
                        continue
                    status = "ok" if data is not None else "incomplete"
                    return self._finish(agent, task, status, full_text, data, step, tool_call_count, usage, model_used, track_state)
                return self._finish(agent, task, "ok", full_text, None, step, tool_call_count, usage, model_used, track_state)

            # step limit reached: ask for a final answer
            messages.append(
                ChatMessage.user(
                    f"You reached the step limit ({max_steps}). Do not call any more tools. Give your final answer now"
                    + (" as the required JSON object" if task.output == "json" else "")
                    + ", and state clearly what is unfinished."
                )
            )
            messages = await s.compactor.maybe_compact(agent, system, messages, specs)
            request = CompletionRequest(
                model="", system=system, messages=messages, tools=specs, temperature=agent.temperature,
                max_output_tokens=self._max_output(agent), metadata={"agent": agent.name, "purpose": task.purpose, "final": True},
            )
            result = await s.gateway.complete(request, model_ref=agent.model_ref or "", agent=agent.name, purpose=task.purpose, fallbacks=agent.fallback_refs)
            usage = usage + result.response.usage
            text = result.response.message.content or ""
            data = extract_json(text) if task.output == "json" else None
            return self._finish(agent, task, "incomplete", text or "Step limit reached.", data, max_steps, tool_call_count, usage, result.entry.ref, track_state)
        except (OperationCancelled, BudgetExceededError):
            self._set_state(agent, AgentState.IDLE, "", track_state)
            raise
        except (ProviderUnavailableError, ProviderError) as exc:
            agent.stats.failures += 1
            detail = str(exc)
            if isinstance(exc, ProviderUnavailableError) and exc.attempts:
                detail += " | " + " | ".join(exc.attempts[-4:])
            self._set_state(agent, AgentState.FAILED, detail[:120], track_state)
            return AgentTaskResult(task_id=task.id, agent=agent.name, status="failed", error=detail, usage=usage, tool_calls=tool_call_count, model=model_used)
        finally:
            if track_state:
                agent.busy = False

    def _finish(
        self,
        agent: Agent,
        task: AgentTask,
        status: str,
        text: str,
        data: object,
        steps: int,
        tool_calls: int,
        usage: Usage,
        model: str | None,
        track_state: bool,
    ) -> AgentTaskResult:
        self._set_state(agent, AgentState.DONE if status == "ok" else AgentState.IDLE, "", track_state)
        if status == "ok":
            agent.remember(f"{task.title}: {text.strip()[:160]}")
        return AgentTaskResult(
            task_id=task.id,
            agent=agent.name,
            status=status,  # type: ignore[arg-type]
            text=text.strip(),
            data=data,
            steps=steps,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
        )
