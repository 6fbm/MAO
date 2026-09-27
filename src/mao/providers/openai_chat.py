"""OpenAI-compatible Chat Completions (``POST /chat/completions``).

Used for local and third-party servers that speak this de-facto standard:
llama.cpp ``llama-server``, LM Studio, vLLM, OpenRouter, Groq, DeepSeek, …
"""

from __future__ import annotations

import json
import time
from typing import Any

from mao.config.schema import ModelConfig
from mao.core.errors import InvalidResponseError
from mao.core.types import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    MessageRole,
    ToolCall,
    Usage,
    new_id,
)
from mao.providers.base import DiscoveredModel, Provider, register_provider
from mao.providers.common import deep_merge, parse_arguments, strip_think

_FINISH = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


@register_provider("openai_chat")
class OpenAIChatProvider(Provider):
    default_base_url = "https://api.openai.com/v1"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = self.base_headers()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def build_body(self, request: CompletionRequest, model: ModelConfig) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if message.role is MessageRole.USER:
                messages.append({"role": "user", "content": message.content})
            elif message.role is MessageRole.ASSISTANT:
                entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                        }
                        for call in message.tool_calls
                    ]
                    if not message.content:
                        entry["content"] = None
                messages.append(entry)
            elif message.role is MessageRole.TOOL:
                messages.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content})
        body: dict[str, Any] = {"model": request.model, "messages": messages, "stream": False}
        if request.max_output_tokens:
            body[self.config.options.get("max_tokens_param", "max_tokens")] = request.max_output_tokens
        if request.temperature is not None and model.supports_temperature and not request.metadata.get("omit_temperature"):
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
        elif request.json_mode and self.config.options.get("json_mode", True) and not request.metadata.get("omit_json_mode"):
            body["response_format"] = {"type": "json_object"}
        deep_merge(body, model.extra_body)
        deep_merge(body, request.extra_body)
        return body

    def parse_response(self, data: Any, request: CompletionRequest) -> CompletionResponse:
        if not isinstance(data, dict) or not data.get("choices"):
            raise InvalidResponseError("Response contains no 'choices'", provider=self.name)
        choice = data["choices"][0]
        raw_message = choice.get("message") or {}
        content = raw_message.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        calls: list[ToolCall] = []
        for raw_call in raw_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            arguments, error = parse_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=raw_call.get("id") or new_id("call_"),
                    name=function.get("name", ""),
                    arguments=arguments,
                    parse_error=error,
                )
            )
        raw_finish = choice.get("finish_reason")
        finish = _FINISH.get(raw_finish or "", FinishReason.OTHER)
        if calls:
            finish = FinishReason.TOOL_CALLS
        usage_data = data.get("usage") or {}
        reported_cost = usage_data.get("cost")
        usage = Usage(
            input_tokens=int(usage_data.get("prompt_tokens") or 0),
            output_tokens=int(usage_data.get("completion_tokens") or 0),
            cached_input_tokens=int((usage_data.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            reasoning_tokens=int((usage_data.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            reported_cost_usd=float(reported_cost) if isinstance(reported_cost, (int, float)) else None,
        )
        message = ChatMessage(role=MessageRole.ASSISTANT, content=strip_think(content).strip(), tool_calls=calls)
        return CompletionResponse(
            message=message,
            usage=usage,
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            model=data.get("model") or request.model,
            provider=self.name,
            response_id=data.get("id"),
        )

    async def complete(
        self,
        request: CompletionRequest,
        *,
        api_key: str | None,
        model: ModelConfig,
        timeout: float,
    ) -> CompletionResponse:
        started = time.monotonic()
        data = await self._request_json(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(api_key),
            body=self.build_body(request, model),
            timeout=timeout,
        )
        response = self.parse_response(data, request)
        response.latency_s = time.monotonic() - started
        return response

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        data = await self._request_json("GET", f"{self.base_url}/models", headers=self._headers(api_key), timeout=15)
        models = []
        for entry in (data or {}).get("data", []):
            if isinstance(entry, dict) and entry.get("id"):
                context = entry.get("context_length") or (entry.get("meta") or {}).get("n_ctx_train")
                models.append(
                    DiscoveredModel(
                        id=entry["id"],
                        context_window=int(context) if isinstance(context, (int, float)) else None,
                        capabilities=["local"] if self.config.local else [],
                        tool_calling="native",
                    )
                )
        return sorted(models, key=lambda m: m.id)
