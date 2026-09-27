"""Anthropic Messages API (``POST /v1/messages``).

Current models think by default. Thinking blocks (with their signatures) are
kept in ``provider_data`` and sent back unchanged within a tool loop, as the
API requires. When history comes from another model or was rebuilt, it is
converted to plain text/tool blocks without thinking, which adaptive
thinking accepts.
"""

from __future__ import annotations

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
)
from mao.providers.base import DiscoveredModel, Provider, register_provider
from mao.providers.common import deep_merge, native_history

DEFAULT_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 16_000

_STOP = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    "refusal": FinishReason.CONTENT_FILTER,
    "pause_turn": FinishReason.OTHER,
}


@register_provider("anthropic")
class AnthropicProvider(Provider):
    default_base_url = "https://api.anthropic.com/v1"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = self.base_headers()
        headers["anthropic-version"] = str(self.config.options.get("anthropic_version", DEFAULT_VERSION))
        betas = self.config.options.get("beta_headers")
        if betas:
            headers["anthropic-beta"] = ",".join(betas) if isinstance(betas, list) else str(betas)
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _messages(self, request: CompletionRequest) -> list[dict[str, Any]]:
        plain = bool(request.metadata.get("plain_history"))
        out: list[dict[str, Any]] = []

        def append(role: str, blocks: list[dict[str, Any]]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for message in request.messages:
            if message.role is MessageRole.USER:
                append("user", [{"type": "text", "text": message.content or "(empty)"}])
            elif message.role is MessageRole.ASSISTANT:
                native = None if plain else native_history(
                    message, wire_format=self.wire_format, provider=self.name, model=request.model
                )
                if native and native.get("content"):
                    blocks = [dict(block) for block in native["content"]]
                else:
                    blocks = []
                    if message.content:
                        blocks.append({"type": "text", "text": message.content})
                    for call in message.tool_calls:
                        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
                    if not blocks:
                        blocks.append({"type": "text", "text": "(no output)"})
                append("assistant", blocks)
            elif message.role is MessageRole.TOOL:
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content or "(empty)",
                }
                if message.is_error:
                    block["is_error"] = True
                append("user", [block])
        return out

    def build_body(self, request: CompletionRequest, model: ModelConfig) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens or min(model.max_output_tokens, DEFAULT_MAX_TOKENS),
            "messages": self._messages(request),
        }
        if request.system:
            body["system"] = request.system
        if request.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in request.tools
            ]
        if request.temperature is not None and model.supports_temperature and not request.metadata.get("omit_temperature"):
            body["temperature"] = request.temperature
        deep_merge(body, model.extra_body)
        deep_merge(body, request.extra_body)
        return body

    def parse_response(self, data: Any, request: CompletionRequest) -> CompletionResponse:
        if not isinstance(data, dict) or data.get("type") == "error":
            raise InvalidResponseError(f"Unexpected response: {str(data)[:200]}", provider=self.name)
        content = data.get("content") or []
        texts = [block.get("text", "") for block in content if block.get("type") == "text"]
        calls = [
            ToolCall(id=block["id"], name=block.get("name", ""), arguments=block.get("input") or {})
            for block in content
            if block.get("type") == "tool_use" and block.get("id")
        ]
        raw_stop = data.get("stop_reason")
        finish = _STOP.get(raw_stop or "", FinishReason.OTHER)
        if calls and finish is FinishReason.STOP:
            finish = FinishReason.TOOL_CALLS
        usage_data = data.get("usage") or {}
        cache_read = int(usage_data.get("cache_read_input_tokens") or 0)
        cache_write = int(usage_data.get("cache_creation_input_tokens") or 0)
        usage = Usage(
            input_tokens=int(usage_data.get("input_tokens") or 0) + cache_read + cache_write,
            output_tokens=int(usage_data.get("output_tokens") or 0),
            cached_input_tokens=cache_read,
        )
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="\n".join(t for t in texts if t).strip(),
            tool_calls=calls,
            provider_data={"format": self.wire_format, "provider": self.name, "model": request.model, "content": content},
        )
        return CompletionResponse(
            message=message,
            usage=usage,
            finish_reason=finish,
            raw_finish_reason=raw_stop,
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
            f"{self.base_url}/messages",
            headers=self._headers(api_key),
            body=self.build_body(request, model),
            timeout=timeout,
        )
        response = self.parse_response(data, request)
        response.latency_s = time.monotonic() - started
        return response

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        models: list[DiscoveredModel] = []
        params: dict[str, Any] = {"limit": 1000}
        for _page in range(10):
            data = await self._request_json(
                "GET", f"{self.base_url}/models", headers=self._headers(api_key), params=params, timeout=30
            )
            for entry in (data or {}).get("data", []):
                capabilities = ["tools"]
                caps = entry.get("capabilities") or {}
                if (caps.get("image_input") or {}).get("supported"):
                    capabilities.append("vision")
                if (caps.get("thinking") or {}).get("supported"):
                    capabilities.append("reasoning")
                models.append(
                    DiscoveredModel(
                        id=entry["id"],
                        display_name=entry.get("display_name"),
                        context_window=entry.get("max_input_tokens") or None,
                        max_output_tokens=entry.get("max_tokens") or None,
                        capabilities=capabilities,
                        tool_calling="native",
                    )
                )
            if not data.get("has_more") or not data.get("last_id"):
                break
            params = {"limit": 1000, "after_id": data["last_id"]}
        return models
