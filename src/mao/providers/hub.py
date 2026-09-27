"""Process-wide owner of provider instances, key pools, limiters and the catalog."""

from __future__ import annotations

import asyncio
import os

import httpx

from mao.config.schema import AppConfig
from mao.core.errors import ConfigError, NoAvailableKeyError, ProviderError
from mao.models.catalog import CatalogEntry, ModelCatalog
from mao.providers.base import Provider, ProviderHealth, create_provider
from mao.providers.keypool import KeyPool
from mao.providers.ratelimit import ProviderLimiter
from mao.security.redaction import Redactor
from mao.security.secrets import SecretStore


class ProviderHub:
    def __init__(
        self,
        config: AppConfig,
        *,
        secrets: SecretStore,
        http: httpx.AsyncClient,
        redactor: Redactor,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self.http = http
        self.redactor = redactor
        self.catalog = ModelCatalog(config.providers.providers, config.settings)
        self.providers: dict[str, Provider] = {}
        self.keypools: dict[str, KeyPool] = {}
        self.limiters: dict[str, ProviderLimiter] = {}
        self.health: dict[str, ProviderHealth] = {}
        self.secret_store_error: str | None = None
        for name, provider_config in config.providers.providers.items():
            self.providers[name] = create_provider(name, provider_config, http)
            self.limiters[name] = ProviderLimiter(provider_config.rate_limit)
            self.reload_keys(name)

    def _collect_keys(self, name: str) -> list[tuple[str, str]]:
        provider_config = self.config.providers.providers[name]
        keys: list[tuple[str, str]] = []
        for env_name in provider_config.api_keys.env:
            value = os.environ.get(env_name, "").strip()
            if value:
                self.redactor.add_secret(value)
                keys.append((value, f"env:{env_name}"))
        if provider_config.api_keys.use_secret_store:
            try:
                for value in self.secrets.keys(name):
                    keys.append((value, "secret-store"))
            except ConfigError as exc:
                self.secret_store_error = str(exc)
        return keys

    def reload_keys(self, name: str) -> KeyPool:
        provider_config = self.config.providers.providers[name]
        pool = KeyPool(name, self._collect_keys(name), requires_key=provider_config.requires_api_key)
        self.keypools[name] = pool
        self.catalog.set_key_availability(name, pool.is_usable())
        return pool

    def _api_key(self, name: str) -> str | None:
        pool = self.keypools[name]
        try:
            state = pool.acquire()
        except NoAvailableKeyError:
            return None
        pool.release(state)
        return state.key if state else None

    async def check_health(self, names: list[str] | None = None) -> dict[str, ProviderHealth]:
        targets = [
            n
            for n in (names or list(self.providers))
            if self.config.providers.providers[n].enabled and self.keypools[n].is_usable()
        ]

        async def one(name: str) -> tuple[str, ProviderHealth]:
            health = await self.providers[name].health_check(api_key=self._api_key(name))
            return name, health

        results = dict(await asyncio.gather(*(one(n) for n in targets)))
        for name, health in results.items():
            self.health[name] = health
            # only local servers are marked unreachable: a cloud hiccup should not disable a provider
            if self.config.providers.providers[name].local:
                self.catalog.set_reachability(name, health.ok, health.message)
        return results

    async def discover(self, name: str) -> list[CatalogEntry]:
        if name not in self.providers:
            raise ConfigError(f"Unknown provider: {name}")
        models = await self.providers[name].list_models(api_key=self._api_key(name))
        return self.catalog.add_discovered(name, models)

    async def autodiscover(self) -> dict[str, str]:
        """Discover models of enabled auto-discover providers. Returns provider -> message."""
        messages: dict[str, str] = {}

        async def one(name: str) -> None:
            try:
                added = await self.discover(name)
                self.catalog.set_reachability(name, True, "")
                messages[name] = f"{len(added)} models detected"
            except ProviderError as exc:
                if self.config.providers.providers[name].local:
                    self.catalog.set_reachability(name, False, str(exc))
                messages[name] = f"not reachable: {exc}"

        targets = [
            name
            for name, cfg in self.config.providers.providers.items()
            if cfg.enabled and cfg.auto_discover and self.keypools[name].is_usable()
        ]
        await asyncio.gather(*(one(n) for n in targets))
        return messages
