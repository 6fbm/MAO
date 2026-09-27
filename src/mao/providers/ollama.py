"""Ollama native API (``POST /api/chat``) with model discovery.

``options.num_ctx`` is always set explicitly: Ollama's small default context
would otherwise silently truncate agent prompts.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from mao.config.schema import ModelConfig
from mao.core.errors import InvalidResponseError, ProviderError
from mao.core.types import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    MessageRole,
    ToolCall,
    Usage,
)
from mao.providers.base import DiscoveredModel, Provider, ProviderHealth, register_provider
from mao.providers.common import deep_merge, parse_arguments, strip_think

DEFAULT_NUM_CTX = 8192


@register_provider("ollama")
class OllamaProvider(Provider):
    default_base_url = "http://localhost:11434"

    def num_ctx(self, model: ModelConfig) -> int:
        configured = self.config.options.get("num_ctx")
        if configured:
            return min(model.context_window, int(configured))
        return model.context_window

    def build_body(self, request: CompletionRequest, model: ModelConfig) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if message.role is MessageRole.USER:
                messages.append({"role": "user", "content": message.content})
            elif message.role is MessageRole.ASSISTANT:
                entry: dict[str, Any] = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {"type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                        for call in message.tool_calls
                    ]
                messages.append(entry)
            elif message.role is MessageRole.TOOL:
                messages.append({"role": "tool", "content": message.content, "tool_name": message.name or ""})
        options: dict[str, Any] = {"num_ctx": self.num_ctx(model)}
        if request.temperature is not None and model.supports_temperature and not request.metadata.get("omit_temperature"):
            options["temperature"] = request.temperature
        if request.max_output_tokens:
            options["num_predict"] = request.max_output_tokens
        body: dict[str, Any] = {"model": request.model, "messages": messages, "stream": False, "options": options}
        keep_alive = self.config.options.get("keep_alive")
        if keep_alive:
            body["keep_alive"] = keep_alive
        if "think" in self.config.options:
            body["think"] = self.config.options["think"]
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
        elif request.json_mode and not request.metadata.get("omit_json_mode"):
            body["format"] = "json"
        deep_merge(body, model.extra_body)
        deep_merge(body, request.extra_body)
        return body

    def parse_response(self, data: Any, request: CompletionRequest) -> CompletionResponse:
        if not isinstance(data, dict) or "message" not in data:
            raise InvalidResponseError(f"Unexpected Ollama response: {str(data)[:200]}", provider=self.name)
        raw_message = data.get("message") or {}
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(raw_message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            arguments, error = parse_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=raw_call.get("id") or f"ollama_{index}_{uuid.uuid4().hex[:8]}",
                    name=function.get("name", ""),
                    arguments=arguments,
                    parse_error=error,
                )
            )
        done_reason = data.get("done_reason")
        finish = FinishReason.LENGTH if done_reason == "length" else FinishReason.STOP
        if calls:
            finish = FinishReason.TOOL_CALLS
        usage = Usage(
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=strip_think(raw_message.get("content") or "").strip(),
            tool_calls=calls,
        )
        return CompletionResponse(
            message=message,
            usage=usage,
            finish_reason=finish,
            raw_finish_reason=done_reason,
            model=data.get("model") or request.model,
            provider=self.name,
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
            f"{self.base_url}/api/chat",
            headers=self.base_headers(),
            body=self.build_body(request, model),
            timeout=timeout,
        )
        response = self.parse_response(data, request)
        response.latency_s = time.monotonic() - started
        return response

    # ------------------------------------------------------------------ discovery

    async def _show(self, name: str) -> dict[str, Any]:
        return await self._request_json(
            "POST", f"{self.base_url}/api/show", headers=self.base_headers(), body={"model": name}, timeout=30
        )

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        tags = await self._request_json("GET", f"{self.base_url}/api/tags", headers=self.base_headers(), timeout=15)
        entries = [e for e in (tags or {}).get("models", []) if isinstance(e, dict) and e.get("name")]

        async def describe(entry: dict[str, Any]) -> DiscoveredModel:
            name = entry["name"]
            details = dict(entry.get("details") or {})
            try:
                info = await self._show(name)
            except ProviderError as exc:
                details["error"] = str(exc)
                return DiscoveredModel(id=name, size_bytes=entry.get("size"), details=details, capabilities=["local"])
            raw_caps = [str(c) for c in info.get("capabilities") or []]
            capabilities = ["local"]
            if "tools" in raw_caps:
                capabilities.append("tools")
            if "vision" in raw_caps:
                capabilities.append("vision")
            if "thinking" in raw_caps:
                capabilities.append("reasoning")
            context = None
            for key, value in (info.get("model_info") or {}).items():
                if key.endswith(".context_length") and isinstance(value, (int, float)):
                    context = int(value)
                    break
            details["raw_capabilities"] = raw_caps
            return DiscoveredModel(
                id=name,
                context_window=context,
                capabilities=capabilities,
                tool_calling="native" if "tools" in raw_caps else "prompt",
                size_bytes=entry.get("size"),
                details=details,
            )

        return list(await asyncio.gather(*(describe(e) for e in entries)))

    async def running_models(self) -> list[dict[str, Any]]:
        data = await self._request_json("GET", f"{self.base_url}/api/ps", headers=self.base_headers(), timeout=10)
        return list((data or {}).get("models", []))

    async def health_check(self, *, api_key: str | None) -> ProviderHealth:
        started = time.monotonic()
        try:
            version = await self._request_json(
                "GET", f"{self.base_url}/api/version", headers=self.base_headers(), timeout=5
            )
            tags = await self._request_json("GET", f"{self.base_url}/api/tags", headers=self.base_headers(), timeout=10)
        except ProviderError as exc:
            return ProviderHealth(ok=False, message=f"Ollama not reachable: {exc}", latency_s=time.monotonic() - started)
        count = len((tags or {}).get("models", []))
        return ProviderHealth(
            ok=True,
            message=f"Ollama {version.get('version', '?')}",
            latency_s=time.monotonic() - started,
            models=count,
        )
