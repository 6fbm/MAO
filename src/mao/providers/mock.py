"""Deterministic provider for tests and the explicitly labelled offline demo.

The mock provider is ``selectable: false`` in the default configuration, so
the automatic model selection never uses it. Its output is always marked as
simulated in the UI.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable

from mao.config.schema import ModelConfig
from mao.core.types import ChatMessage, CompletionRequest, CompletionResponse, Usage
from mao.providers.base import DiscoveredModel, Provider, register_provider
from mao.tokens.estimator import ESTIMATOR

Responder = Callable[[CompletionRequest], CompletionResponse | str | Awaitable[CompletionResponse | str]]


def _default_responder(request: CompletionRequest) -> str:
    return "OK (mock response - simulated)"


@register_provider("mock")
class MockProvider(Provider):
    responder: Responder | None = None

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.calls: list[CompletionRequest] = []
        self.responder = None

    def set_responder(self, responder: Responder | None) -> None:
        self.responder = responder

    async def complete(
        self,
        request: CompletionRequest,
        *,
        api_key: str | None,
        model: ModelConfig,
        timeout: float,
    ) -> CompletionResponse:
        started = time.monotonic()
        self.calls.append(request)
        result = (self.responder or _default_responder)(request)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            result = CompletionResponse(message=ChatMessage.assistant(result))
        result.provider = self.name
        result.model = request.model
        if result.usage.input_tokens == 0 and result.usage.output_tokens == 0:
            result.usage = Usage(
                input_tokens=ESTIMATOR.count_request(request.messages, request.system, request.tools),
                output_tokens=ESTIMATOR.count_message(result.message),
                estimated=True,
            )
        result.latency_s = time.monotonic() - started
        return result

    async def list_models(self, *, api_key: str | None) -> list[DiscoveredModel]:
        return [DiscoveredModel(id=cfg.id or key, context_window=cfg.context_window) for key, cfg in self.config.models.items()]
