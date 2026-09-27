"""Application container: configuration, providers, tools, sessions and the orchestrator."""

from __future__ import annotations

import asyncio
import logging

import httpx

from mao.config.loader import ConfigManager
from mao.config.schema import AppConfig
from mao.core.events import EventBus
from mao.orchestration.orchestrator import Orchestrator
from mao.orchestration.resources import ResourceMonitor
from mao.paths import AppPaths
from mao.providers.hub import ProviderHub
from mao.security.approval import ApprovalHandler, deny_all
from mao.security.redaction import REDACTOR, RedactingFilter
from mao.security.secrets import SecretStore
from mao.sessions.store import SessionStore
from mao.tokens.budget import LimitHandler
from mao.tools.registry import build_default_registry


def configure_logging() -> None:
    root = logging.getLogger("mao")
    root.setLevel(logging.DEBUG)
    if not any(isinstance(f, RedactingFilter) for f in root.filters):
        root.addFilter(RedactingFilter(REDACTOR))
    # never let HTTP libraries log request details (headers could contain keys)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


class AppContext:
    def __init__(self, paths: AppPaths, config_manager: ConfigManager, config: AppConfig, *, demo: bool = False) -> None:
        configure_logging()
        self.paths = paths
        self.config_manager = config_manager
        self.config = config
        self.demo = demo
        self.redactor = REDACTOR
        self.bus = EventBus()
        self.secrets = SecretStore(paths.secrets_file(config.settings.secrets_file), self.redactor)
        self.http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            timeout=httpx.Timeout(60.0, connect=15.0),
        )
        self.tool_registry = build_default_registry()
        self.resources: ResourceMonitor = ResourceMonitor()
        self.store = SessionStore(paths.logs_dir(config.settings.logs_dir))
        self.approval_handler: ApprovalHandler = deny_all
        self.limit_handler: LimitHandler | None = None
        self.discovery_messages: dict[str, str] = {}
        self.hub = self._build_hub()
        self.orchestrator = Orchestrator(self)

    @classmethod
    async def create(cls, paths: AppPaths, *, demo: bool = False, discover: bool = True) -> AppContext:
        manager = ConfigManager(paths.config_dir)
        manager.ensure_templates()
        config = manager.load()
        if demo:
            from mao.demo import apply_demo_config

            apply_demo_config(config)
        app = cls(paths, manager, config, demo=demo)
        if discover:
            await app.refresh_models()
        return app

    def _build_hub(self) -> ProviderHub:
        hub = ProviderHub(self.config, secrets=self.secrets, http=self.http, redactor=self.redactor)
        if self.demo:
            from mao.demo import install_demo_responder

            install_demo_responder(hub)
        return hub

    async def refresh_models(self, timeout_s: float = 12.0) -> dict[str, str]:
        try:
            self.discovery_messages = await asyncio.wait_for(self.hub.autodiscover(), timeout_s)
        except asyncio.TimeoutError:
            self.discovery_messages = {"*": "Model detection: timeout"}
        return self.discovery_messages

    async def reload_config(self) -> None:
        config = self.config_manager.load()
        if self.demo:
            from mao.demo import apply_demo_config

            apply_demo_config(config)
        self.config = config
        self.secrets = SecretStore(self.paths.secrets_file(config.settings.secrets_file), self.redactor)
        self.store = SessionStore(self.paths.logs_dir(config.settings.logs_dir))
        self.hub = self._build_hub()
        await self.refresh_models()

    def set_handlers(self, approval: ApprovalHandler, limit: LimitHandler | None) -> None:
        self.approval_handler = approval
        self.limit_handler = limit

    async def aclose(self) -> None:
        await self.orchestrator.close_active()
        await self.http.aclose()
