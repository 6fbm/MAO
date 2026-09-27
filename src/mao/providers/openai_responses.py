"""OpenAI Responses API (``POST /v1/responses``) – used for OpenAI and xAI.

Requests are stateless (``store: false``). For reasoning models the encrypted
reasoning items are requested and replayed with the next request of the same
tool loop, as recommended by the official function-calling guide.
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
from mao.providers.common import deep_merge, native_history, parse_arguments


@register_provider("openai_responses")
class OpenAIResponsesProvider(Provider):
    default_base_url = "https://api.openai.com/v1"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = self.base_headers()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # ------------------------------------------------------------------ request

    def _input_items(self, request: CompletionRequest) -> list[dict[str, Any]]:
        plain = bool(request.metadata.get("plain_history"))
        items: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role is MessageRole.USER:
                items.append({"role": "user", "content": message.content})
            elif message.role is MessageRole.ASSISTANT:
                native = None if plain else native_history(
                    message, wire_format=self.wire_format, provider=self.name, model=request.model
                )
                if native and isinstance(native.get("output"), list):
                    items.extend(native["output"])
                    continue
                if message.content:
                    items.append({"role": "assistant", "content": message.content})
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        }
                    )
            elif message.role is MessageRole.TOOL:
                items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content})
        return items

    def build_body(self, request: CompletionRequest, model: ModelConfig) -> dict[str, Any]:
        body: dict[str, Any] = {"model": request.model, "input": self._input_items(request), "store": False}
        if request.system:
            body["instructions"] = request.system
        if request.tools:
            body["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in request.tools
            ]
        if request.max_output_tokens:
            body["max_output_tokens"] = request.max_output_tokens
        if request.temperature is not None and model.supports_temperature and not request.metadata.get("omit_temperature"):
            body["temperature"] = request.temperature
        if request.json_mode and not request.tools and not request.metadata.get("omit_json_mode"):
            body["text"] = {"format": {"type": "json_object"}}
        if (
            model.reasoning
            and self.config.options.get("include_encrypted_reasoning", True)
            and not request.metadata.get("omit_include")
        ):
            body["include"] = ["reasoning.encrypted_content"]
        deep_merge(body, model.extra_body)
        deep_merge(body, request.extra_body)
        return body

    # ------------------------------------------------------------------ response

    def parse_response(self, data: Any, request: CompletionRequest) -> CompletionResponse:
        if not isinstance(data, dict):
            raise InvalidResponseError("Unerwartetes Responseformat", provider=self.name)
        if data.get("error"):
            raise InvalidResponseError(f"Error in the response: {data['error']}", provider=self.name)
        output = data.get("output") or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for item in output:
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "refusal":
                        texts.append(part.get("refusal", ""))
            elif item_type == "function_call":
                arguments, error = parse_arguments(item.get("arguments"))
                calls.append(
                    ToolCall(
                        id=item.get("call_id") or item.get("id") or new_id("call_"),
                        name=item.get("name", ""),
                        arguments=arguments,
                        parse_error=error,
                    )
                )
        usage_data = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("input_tokens") or 0),
            output_tokens=int(usage_data.get("output_tokens") or 0),
            cached_input_tokens=int((usage_data.get("input_tokens_details") or {}).get("cached_tokens") or 0),
            reasoning_tokens=int((usage_data.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
        )
        status = data.get("status")
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            finish = {"max_output_tokens": FinishReason.LENGTH, "content_filter": FinishReason.CONTENT_FILTER}.get(
                reason, FinishReason.OTHER
            )
        elif status == "failed":
            raise InvalidResponseError(f"Responsestatus 'failed': {data.get('error')}", provider=self.name)
        else:
            finish = FinishReason.TOOL_CALLS if calls else FinishReason.STOP
        # reasoning items without encrypted content cannot be replayed statelessly
        replay = [item for item in output if item.get("type") != "reasoning" or item.get("encrypted_content")]
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="\n".join(t for t in texts if t).strip(),
            tool_calls=calls,
            provider_data={"format": self.wire_format, "provider": self.name, "model": request.model, "output": replay},
        )
        return CompletionResponse(
            message=message,
            usage=usage,
            finish_reason=finish,
            raw_finish_reason=status,
            model=data.get("model") or request.model,
            provider=self.name,
            response_id=data.get("id"),
        )

    # ------------------------------------------------------------------ api

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
            f"{self.base_url}/responses",
            headers=self._headers(api_key),
            body=self.build_body(request, model),
            timeout=timeout,
        )
        response = self.parse_response(data, request)
        response.latency_s = time.monotonic() - started
        return response

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        data = await self._request_json("GET", f"{self.base_url}/models", headers=self._headers(api_key), timeout=30)
        models = []
        for entry in (data or {}).get("data", []):
            if isinstance(entry, dict) and entry.get("id"):
                models.append(DiscoveredModel(id=entry["id"], details={"owned_by": entry.get("owned_by")}))
        return sorted(models, key=lambda m: m.id)
