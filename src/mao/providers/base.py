"""Provider interface and registry.

A provider performs exactly one attempt of a model call and converts between
the neutral types and its wire format. Retries, key rotation, rate limits,
budgets and fallbacks live in :mod:`mao.providers.gateway`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from mao import __version__
from mao.config.schema import ModelConfig, ProviderConfig
from mao.core.errors import ConfigError, InvalidResponseError, ProviderError
from mao.core.types import CompletionRequest, CompletionResponse
from mao.providers.http import map_http_error, map_transport_error


class DiscoveredModel(BaseModel):
    id: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    tool_calling: str | None = None
    size_bytes: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    ok: bool
    message: str = ""
    latency_s: float | None = None
    models: int | None = None


class Provider(ABC):
    type_name: ClassVar[str] = ""
    default_base_url: ClassVar[str] = ""

    def __init__(self, name: str, config: ProviderConfig, http: httpx.AsyncClient) -> None:
        self.name = name
        self.config = config
        self.http = http

    @property
    def base_url(self) -> str:
        return (self.config.base_url or self.default_base_url).rstrip("/")

    @property
    def wire_format(self) -> str:
        return self.type_name

    # ------------------------------------------------------------------ interface

    @abstractmethod
    async def complete(
        self,
        request: CompletionRequest,
        *,
        api_key: str | None,
        model: ModelConfig,
        timeout: float,
    ) -> CompletionResponse:
        """Perform a single request. Raise a classified ``ProviderError`` on failure."""

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        return [
            DiscoveredModel(id=cfg.id or key, context_window=cfg.context_window, capabilities=list(cfg.capabilities))
            for key, cfg in self.config.models.items()
        ]

    async def health_check(self, *, api_key: str | None) -> ProviderHealth:
        started = time.monotonic()
        try:
            models = await self.list_models(api_key=api_key)
        except ProviderError as exc:
            return ProviderHealth(ok=False, message=str(exc), latency_s=time.monotonic() - started)
        return ProviderHealth(ok=True, message="reachable", latency_s=time.monotonic() - started, models=len(models))

    # ------------------------------------------------------------------ helpers

    def base_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": f"multi-ai-orchestrator/{__version__}"}
        headers.update(self.config.headers)
        return headers

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self.http.request(method, url, json=body, headers=headers, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            raise map_transport_error(self.name, exc) from exc
        if response.status_code >= 400:
            raise map_http_error(self.name, response)
        try:
            return response.json()
        except ValueError as exc:
            raise InvalidResponseError("Response is not valid JSON", provider=self.name) from exc


_REGISTRY: dict[str, type[Provider]] = {}


def register_provider(type_name: str):
    def decorator(cls: type[Provider]) -> type[Provider]:
        cls.type_name = type_name
        _REGISTRY[type_name] = cls
        return cls

    return decorator


def load_builtin_providers() -> None:
    # importing registers the classes
    from mao.providers import anthropic, gemini, mock, ollama, openai_chat, openai_responses  # noqa: F401


def provider_types() -> list[str]:
    load_builtin_providers()
    return sorted(_REGISTRY)


def create_provider(name: str, config: ProviderConfig, http: httpx.AsyncClient) -> Provider:
    load_builtin_providers()
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise ConfigError(
            f"Provider '{name}': unknown type '{config.type}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return cls(name, config, http)
