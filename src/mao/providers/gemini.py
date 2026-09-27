"""Google Gemini ``models.generateContent``.

``thoughtSignature`` values returned with function calls are replayed
verbatim for the same model. For history that did not come from this model,
the documented placeholder signature is attached to the first function call
of each step so Gemini 3 validation accepts it.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from mao.config.schema import ModelConfig
from mao.core.errors import ContentFilterError, InvalidResponseError
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
from mao.providers.common import deep_merge, native_history, parse_arguments

PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"
GENERATED_ID_PREFIX = "gemini_call_"

_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "anyOf",
    "title",
    "propertyOrdering",
}

_FINISH = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
    "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    "UNEXPECTED_TOOL_CALL": FinishReason.ERROR,
}


def sanitize_schema(schema: Any) -> Any:
    """Reduce JSON Schema to the OpenAPI subset accepted by ``parameters``."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        out["type"] = non_null[0] if non_null else "string"
        if "null" in raw_type:
            out["nullable"] = True
    elif raw_type:
        out["type"] = raw_type
    for key, value in schema.items():
        if key == "type" or key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: sanitize_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = sanitize_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            out[key] = [sanitize_schema(sub) for sub in value]
        else:
            out[key] = value
    if "required" in out:
        props = out.get("properties") or {}
        out["required"] = [name for name in out["required"] if name in props]
        if not out["required"]:
            out.pop("required")
    return out


@register_provider("gemini")
class GeminiProvider(Provider):
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = self.base_headers()
        if api_key:
            headers["x-goog-api-key"] = api_key
        return headers

    @staticmethod
    def _model_path(model_id: str) -> str:
        return model_id if model_id.startswith("models/") else f"models/{model_id}"

    # ------------------------------------------------------------------ request

    def _contents(self, request: CompletionRequest) -> list[dict[str, Any]]:
        plain = bool(request.metadata.get("plain_history"))
        out: list[dict[str, Any]] = []

        def append(role: str, parts: list[dict[str, Any]]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["parts"].extend(parts)
            else:
                out.append({"role": role, "parts": list(parts)})

        for message in request.messages:
            if message.role is MessageRole.USER:
                append("user", [{"text": message.content or " "}])
            elif message.role is MessageRole.ASSISTANT:
                native = None if plain else native_history(
                    message, wire_format=self.wire_format, provider=self.name, model=request.model
                )
                if native and native.get("parts"):
                    append("model", [dict(part) for part in native["parts"]])
                    continue
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append({"text": message.content})
                for index, call in enumerate(message.tool_calls):
                    function_call: dict[str, Any] = {"name": call.name, "args": call.arguments}
                    if call.id and not call.id.startswith(GENERATED_ID_PREFIX):
                        function_call["id"] = call.id
                    part: dict[str, Any] = {"functionCall": function_call}
                    if index == 0:
                        part["thoughtSignature"] = PLACEHOLDER_SIGNATURE
                    parts.append(part)
                append("model", parts or [{"text": " "}])
            elif message.role is MessageRole.TOOL:
                payload = {"error": message.content} if message.is_error else {"result": message.content}
                function_response: dict[str, Any] = {"name": message.name or "tool", "response": payload}
                if message.tool_call_id and not message.tool_call_id.startswith(GENERATED_ID_PREFIX):
                    function_response["id"] = message.tool_call_id
                append("user", [{"functionResponse": function_response}])
        return out

    def build_body(self, request: CompletionRequest, model: ModelConfig) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": self._contents(request)}
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        if request.tools:
            declarations = []
            for tool in request.tools:
                declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
                parameters = sanitize_schema(tool.parameters)
                if parameters.get("properties"):
                    declaration["parameters"] = parameters
                declarations.append(declaration)
            body["tools"] = [{"functionDeclarations": declarations}]
        generation: dict[str, Any] = {}
        if request.max_output_tokens:
            generation["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None and model.supports_temperature and not request.metadata.get("omit_temperature"):
            generation["temperature"] = request.temperature
        if request.json_mode and not request.tools and not request.metadata.get("omit_json_mode"):
            generation["responseMimeType"] = "application/json"
        if generation:
            body["generationConfig"] = generation
        deep_merge(body, model.extra_body)
        deep_merge(body, request.extra_body)
        return body

    # ------------------------------------------------------------------ response

    @staticmethod
    def _usage(data: dict[str, Any]) -> Usage:
        meta = data.get("usageMetadata") or {}
        thoughts = int(meta.get("thoughtsTokenCount") or 0)
        return Usage(
            input_tokens=int(meta.get("promptTokenCount") or 0) + int(meta.get("toolUsePromptTokenCount") or 0),
            output_tokens=int(meta.get("candidatesTokenCount") or 0) + thoughts,
            cached_input_tokens=int(meta.get("cachedContentTokenCount") or 0),
            reasoning_tokens=thoughts,
        )

    def parse_response(self, data: Any, request: CompletionRequest) -> CompletionResponse:
        if not isinstance(data, dict):
            raise InvalidResponseError("Unerwartetes Responseformat", provider=self.name)
        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = (data.get("promptFeedback") or {}).get("blockReason")
            if block_reason:
                raise ContentFilterError(f"Prompt blocked by Gemini: {block_reason}", provider=self.name)
            raise InvalidResponseError("Response contains no candidates", provider=self.name)
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if "functionCall" in part:
                function_call = part["functionCall"] or {}
                arguments, error = parse_arguments(function_call.get("args"))
                calls.append(
                    ToolCall(
                        id=function_call.get("id") or f"{GENERATED_ID_PREFIX}{uuid.uuid4().hex[:10]}",
                        name=function_call.get("name", ""),
                        arguments=arguments,
                        parse_error=error,
                    )
                )
            elif "text" in part and not part.get("thought"):
                texts.append(part["text"])
        raw_finish = candidate.get("finishReason")
        finish = _FINISH.get(raw_finish or "", FinishReason.OTHER if raw_finish else FinishReason.STOP)
        if calls:
            finish = FinishReason.TOOL_CALLS
        if finish is FinishReason.ERROR and not texts:
            raise InvalidResponseError(f"Gemini reports {raw_finish}", provider=self.name)
        if finish is FinishReason.CONTENT_FILTER and not texts:
            raise ContentFilterError(f"Response blocked by Gemini: {raw_finish}", provider=self.name)
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="".join(texts).strip(),
            tool_calls=calls,
            provider_data={"format": self.wire_format, "provider": self.name, "model": request.model, "parts": parts},
        )
        return CompletionResponse(
            message=message,
            usage=self._usage(data),
            finish_reason=finish,
            raw_finish_reason=raw_finish,
            model=data.get("modelVersion") or request.model,
            provider=self.name,
            response_id=data.get("responseId"),
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
            f"{self.base_url}/{self._model_path(request.model)}:generateContent",
            headers=self._headers(api_key),
            body=self.build_body(request, model),
            timeout=timeout,
        )
        response = self.parse_response(data, request)
        response.latency_s = time.monotonic() - started
        return response

    async def grounded_search(
        self, query: str, *, api_key: str | None, model_id: str, timeout: float
    ) -> tuple[str, list[dict[str, str]], Usage]:
        """Answer a research query with Google Search grounding; returns text, sources and usage."""
        body = {
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "tools": [{"google_search": {}}],
        }
        data = await self._request_json(
            "POST",
            f"{self.base_url}/{self._model_path(model_id)}:generateContent",
            headers=self._headers(api_key),
            body=body,
            timeout=timeout,
        )
        candidates = (data or {}).get("candidates") or []
        if not candidates:
            raise InvalidResponseError("No search results from Gemini", provider=self.name)
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        chunks = (candidate.get("groundingMetadata") or {}).get("groundingChunks") or []
        sources = [
            {"url": chunk["web"].get("uri", ""), "title": chunk["web"].get("title", "")}
            for chunk in chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("web"), dict)
        ]
        return text, sources, self._usage(data)

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        models: list[DiscoveredModel] = []
        params: dict[str, Any] = {"pageSize": 1000}
        for _page in range(10):
            data = await self._request_json(
                "GET", f"{self.base_url}/models", headers=self._headers(api_key), params=params, timeout=30
            )
            for entry in (data or {}).get("models", []):
                if "generateContent" not in (entry.get("supportedGenerationMethods") or []):
                    continue
                model_id = str(entry.get("name", "")).removeprefix("models/")
                capabilities = ["tools", "web_search"]
                if entry.get("thinking"):
                    capabilities.append("reasoning")
                models.append(
                    DiscoveredModel(
                        id=model_id,
                        display_name=entry.get("displayName"),
                        context_window=entry.get("inputTokenLimit"),
                        max_output_tokens=entry.get("outputTokenLimit"),
                        capabilities=capabilities,
                        tool_calling="native",
                    )
                )
            token = (data or {}).get("nextPageToken")
            if not token:
                break
            params = {"pageSize": 1000, "pageToken": token}
        return sorted(models, key=lambda m: m.id)
