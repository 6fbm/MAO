"""Keeps agent conversations inside the model context window.

Instead of editing earlier turns (which breaks Anthropic thinking signatures
and Gemini thought signatures), a long conversation is *rebased*: a new first
user message contains the original task, a dense progress summary and the
latest tool results. The result is valid for every provider.
"""

from __future__ import annotations

import json

from mao.agents.agent import Agent
from mao.config.schema import ContextConfig
from mao.core.errors import BudgetExceededError, ConfigError, OperationCancelled, ProviderError, ProviderUnavailableError
from mao.core.events import ContextEvent, EventBus
from mao.core.text import truncate_middle
from mao.core.types import ChatMessage, CompletionRequest, MessageRole, ToolSpec
from mao.models.catalog import ModelCatalog
from mao.models.selector import ModelSelector, SelectionRequest
from mao.providers.gateway import LLMGateway
from mao.tokens.estimator import ESTIMATOR, TokenEstimator

SUMMARY_SYSTEM = (
    "You compress the working transcript of an AI agent so it can continue its task with less context. "
    "Preserve precisely: the goal, files inspected (paths and the key facts learned), changes made (exact files and what changed), "
    "commands/tests run and their outcomes, decisions, errors and open problems, and the concrete next steps. "
    "Be dense and factual, use bullet points, at most about 600 words."
)


def render_transcript(messages: list[ChatMessage], max_chars: int = 60_000) -> str:
    lines: list[str] = []
    for message in messages:
        if message.role is MessageRole.ASSISTANT:
            if message.content:
                lines.append(f"AGENT: {message.content}")
            for call in message.tool_calls:
                lines.append(f"AGENT CALLS {call.name}({json.dumps(call.arguments, ensure_ascii=False)[:500]})")
        elif message.role is MessageRole.TOOL:
            status = "ERROR" if message.is_error else "RESULT"
            lines.append(f"{status} {message.name}: {truncate_middle(message.content, 1_500)[0]}")
        else:
            lines.append(f"USER: {truncate_middle(message.content, 2_000)[0]}")
    return truncate_middle("\n".join(lines), max_chars)[0]


def fallback_summary(messages: list[ChatMessage]) -> str:
    lines = ["(automatic short summary without a model)"]
    for message in messages:
        for call in message.tool_calls:
            lines.append(f"- {call.name}({json.dumps(call.arguments, ensure_ascii=False)[:200]})")
        if message.role is MessageRole.TOOL:
            lines.append(f"  -> {'Error' if message.is_error else 'ok'}: {message.content[:200]}")
        if message.role is MessageRole.ASSISTANT and message.content:
            lines.append(f"- Note: {message.content[:300]}")
    return "\n".join(lines[-80:])


class ConversationCompactor:
    def __init__(
        self,
        *,
        gateway: LLMGateway,
        catalog: ModelCatalog,
        selector: ModelSelector,
        config: ContextConfig,
        bus: EventBus,
        estimator: TokenEstimator = ESTIMATOR,
    ) -> None:
        self.gateway = gateway
        self.catalog = catalog
        self.selector = selector
        self.config = config
        self.bus = bus
        self.estimator = estimator

    def budget(self, agent: Agent) -> int:
        try:
            entry = self.catalog.resolve(agent.model_ref or "")
        except ConfigError:
            return 32_000
        reserve = min(entry.config.max_output_tokens, agent.max_output_tokens or self.config.default_max_output_tokens)
        return max(2_048, entry.config.context_window - reserve)

    def needs_compaction(self, agent: Agent, system: str, messages: list[ChatMessage], tools: list[ToolSpec], *, aggressive: bool = False) -> tuple[bool, int, int]:
        available = self.budget(agent)
        used = self.estimator.count_request(messages, system, tools)
        threshold = int(available * (0.5 if aggressive else self.config.compaction_threshold))
        return used > threshold, used, available

    async def maybe_compact(
        self,
        agent: Agent,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        *,
        force: bool = False,
        aggressive: bool = False,
    ) -> list[ChatMessage]:
        if len(messages) < 2:
            return messages
        needed, used, available = self.needs_compaction(agent, system, messages, tools, aggressive=aggressive)
        if not (needed or force):
            return messages
        last = messages[-1]
        if last.role is MessageRole.ASSISTANT and last.tool_calls:
            return messages  # not a safe point: results for these calls are still missing
        task_message = messages[0]
        history = messages[1:]
        trailing_results: list[ChatMessage] = []
        for message in reversed(history):
            if message.role is not MessageRole.TOOL:
                break
            trailing_results.insert(0, message)
        summary = await self._summarize(agent, task_message, history)
        char_budget = int(available * self.estimator.chars_per_token * 0.35)
        task_text = truncate_middle(task_message.content, max(4_000, char_budget))[0]
        recap = ["## Progress so far (compacted)", summary]
        if trailing_results:
            recap.append("## Most recent tool results")
            recap.extend(f"[{m.name}] {truncate_middle(m.content, 3_000)[0]}" for m in trailing_results)
        recap.append("Continue the task on this basis. Do not repeat steps that are already done.")
        rebased = [ChatMessage.user(task_text + "\n\n" + "\n\n".join(recap))]
        after = self.estimator.count_request(rebased, system, tools)
        self.bus.publish(ContextEvent(agent=agent.name, action="compacted", before_tokens=used, after_tokens=after))
        return rebased

    async def _summarize(self, agent: Agent, task_message: ChatMessage, history: list[ChatMessage]) -> str:
        if self.config.summary_model.lower() == "auto":
            entry = self.selector.select(SelectionRequest(purpose="summary", preferred_tier="fast"))
            model_ref = entry.ref if entry else agent.model_ref
        else:
            model_ref = self.config.summary_model
        if not model_ref:
            return fallback_summary(history)
        request = CompletionRequest(
            model="",
            system=SUMMARY_SYSTEM,
            messages=[
                ChatMessage.user(
                    f"Task of the agent:\n{truncate_middle(task_message.content, 4_000)[0]}\n\nTranscript:\n{render_transcript(history)}"
                )
            ],
            max_output_tokens=2_000,
            metadata={"agent": agent.name, "purpose": "compaction"},
        )
        fallbacks = [agent.model_ref] if agent.model_ref and agent.model_ref != model_ref else []
        try:
            result = await self.gateway.complete(request, model_ref=model_ref, agent=agent.name, purpose="compaction", fallbacks=fallbacks)
        except (OperationCancelled, BudgetExceededError):
            raise
        except (ProviderUnavailableError, ProviderError):
            return fallback_summary(history)
        return result.response.message.content.strip() or fallback_summary(history)
