"""Provider-neutral runtime types.

Every provider converts these types to and from its own wire format. Opaque
provider state that must be replayed verbatim (Anthropic thinking blocks,
Gemini thought signatures, OpenAI reasoning items) travels in
``ChatMessage.provider_data`` so a multi-turn tool loop stays valid.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class RunMode(str, Enum):
    PLAN = "plan"
    RUN = "run"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # set when the model produced arguments that are not valid JSON
    parse_error: str | None = None


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # TOOL messages: id of the call this result answers and the tool name
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    # {"format": "<provider wire format>", ...} – replayed only to the same format
    provider_data: dict[str, Any] | None = None

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> ChatMessage:
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call: ToolCall, content: str, *, is_error: bool = False) -> ChatMessage:
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=call.id,
            name=call.name,
            is_error=is_error,
        )


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    # True when the provider did not report usage and we estimated it
    estimated: bool = False
    # provider-reported cost in USD, if the API returns one (e.g. OpenRouter)
    reported_cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.reported_cost_usd is not None or other.reported_cost_usd is not None:
            cost = (self.reported_cost_usd or 0.0) + (other.reported_cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            estimated=self.estimated or other.estimated,
            reported_cost_usd=cost,
        )


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    OTHER = "other"


class CompletionRequest(BaseModel):
    model: str  # provider API model id
    messages: list[ChatMessage]
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    temperature: float | None = None
    max_output_tokens: int | None = None
    json_mode: bool = False
    # merged into the provider request body (provider specific knobs)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # never sent to a provider: used for routing, logging and the mock provider
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    message: ChatMessage
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    raw_finish_reason: str | None = None
    model: str = ""
    provider: str = ""
    response_id: str | None = None
    latency_s: float = 0.0
