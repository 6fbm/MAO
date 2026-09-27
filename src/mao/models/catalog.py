"""Configured and discovered models, availability and reference resolution.

References accepted everywhere a model is configured:
``provider/model`` · ``provider`` (its default model) · alias from settings
(``gpt``, ``gemini``, ``grok``, ``claude``, ``local``) · a bare model id.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from mao.config.schema import TIER_RANK, ModelConfig, PricingConfig, ProviderConfig, Settings
from mao.core.errors import ConfigError
from mao.providers.base import DiscoveredModel


@dataclass
class CatalogEntry:
    provider: str
    key: str
    config: ModelConfig
    provider_config: ProviderConfig
    source: str = "config"
    note: str = ""

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.key}"

    @property
    def api_id(self) -> str:
        return self.config.id or self.key

    @property
    def local(self) -> bool:
        return self.provider_config.local

    @property
    def label(self) -> str:
        return self.config.display_name or self.api_id


@dataclass
class ProviderStatus:
    name: str
    enabled: bool
    has_keys: bool
    reachable: bool | None = None
    message: str = ""
    discovered: int = 0
    extra: dict[str, str] = field(default_factory=dict)


class ModelCatalog:
    def __init__(self, providers: dict[str, ProviderConfig], settings: Settings) -> None:
        self._providers = providers
        self._settings = settings
        self._lock = threading.Lock()
        self._entries: dict[str, CatalogEntry] = {}
        self._status: dict[str, ProviderStatus] = {}
        for name, provider_config in providers.items():
            self._status[name] = ProviderStatus(
                name=name, enabled=provider_config.enabled, has_keys=not provider_config.requires_api_key
            )
            for key, model_config in provider_config.models.items():
                self._entries[f"{name}/{key}"] = CatalogEntry(name, key, model_config, provider_config)

    # ------------------------------------------------------------------ status

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers)

    def provider_config(self, name: str) -> ProviderConfig:
        if name not in self._providers:
            raise ConfigError(f"Unknown provider: {name}")
        return self._providers[name]

    def set_key_availability(self, provider: str, has_keys: bool) -> None:
        if provider in self._status:
            self._status[provider].has_keys = has_keys or not self._providers[provider].requires_api_key

    def set_reachability(self, provider: str, reachable: bool | None, message: str = "") -> None:
        if provider in self._status:
            self._status[provider].reachable = reachable
            self._status[provider].message = message

    def status(self, provider: str) -> ProviderStatus:
        return self._status[provider]

    def statuses(self) -> list[ProviderStatus]:
        return list(self._status.values())

    # ------------------------------------------------------------------ entries

    def entries(self, *, provider: str | None = None, include_disabled: bool = True) -> list[CatalogEntry]:
        with self._lock:
            items = list(self._entries.values())
        if provider is not None:
            items = [e for e in items if e.provider == provider]
        if not include_disabled:
            items = [e for e in items if e.config.enabled and e.provider_config.enabled]
        return items

    def is_available(self, entry: CatalogEntry) -> bool:
        status = self._status.get(entry.provider)
        if status is None:
            return False
        if not (entry.config.enabled and entry.provider_config.enabled and status.has_keys):
            return False
        if status.reachable is False:
            return False
        return not (self._settings.privacy.local_only and not entry.local)

    def unavailable_reason(self, entry: CatalogEntry) -> str:
        status = self._status.get(entry.provider)
        if status is None:
            return "unknown provider"
        if not entry.provider_config.enabled:
            return "provider disabled"
        if not entry.config.enabled:
            return entry.note or "Model disabled"
        if not status.has_keys:
            return "no API key"
        if status.reachable is False:
            return status.message or "not reachable"
        if self._settings.privacy.local_only and not entry.local:
            return "local_only is active"
        return ""

    def available_entries(self, *, selectable_only: bool = True) -> list[CatalogEntry]:
        return [
            e
            for e in self.entries()
            if self.is_available(e) and (e.provider_config.selectable or not selectable_only)
        ]

    def add_discovered(self, provider: str, models: list[DiscoveredModel]) -> list[CatalogEntry]:
        provider_config = self.provider_config(provider)
        options = provider_config.options
        num_ctx = options.get("num_ctx")
        added: list[CatalogEntry] = []
        with self._lock:
            known_ids = {e.api_id: e for e in self._entries.values() if e.provider == provider}
            for model in models:
                if model.id in known_ids and known_ids[model.id].source == "config":
                    continue
                broken = "error" in model.details
                context = model.context_window or (int(num_ctx) if num_ctx else 32_000)
                if num_ctx and provider_config.local:
                    context = min(context, int(num_ctx))
                context = max(1_024, context)
                max_output = model.max_output_tokens or (min(8_192, context // 2) if provider_config.local else 16_000)
                capabilities = set(model.capabilities)
                if provider_config.local:
                    capabilities.add("local")
                tool_calling = model.tool_calling or ("native" if "tools" in capabilities else "prompt")
                tier = options.get("discovered_tier", "fast" if provider_config.local else "balanced")
                if tier not in TIER_RANK:
                    tier = "balanced"
                config = ModelConfig(
                    id=model.id,
                    display_name=model.display_name,
                    enabled=not broken,
                    context_window=context,
                    max_output_tokens=max(16, max_output),
                    tier=tier,
                    capabilities=sorted(capabilities),
                    tool_calling=tool_calling,
                    pricing=PricingConfig(),
                )
                entry = CatalogEntry(
                    provider,
                    model.id,
                    config,
                    provider_config,
                    source="discovered",
                    note=f"broken: {model.details['error']}" if broken else "",
                )
                self._entries[entry.ref] = entry
                added.append(entry)
        self._status[provider].discovered = len(added)
        return added

    # ------------------------------------------------------------------ resolution

    def resolve(self, reference: str) -> CatalogEntry:
        ref = reference.strip()
        if not ref:
            raise ConfigError("Empty model reference")
        alias = self._settings.aliases.get(ref.lower())
        if alias:
            ref = alias
        entries = self.entries()
        if "/" in ref:
            provider, key = ref.split("/", 1)
            if provider in self._providers:
                exact = self._entries.get(f"{provider}/{key}")
                if exact is not None:
                    return exact
                by_id = [e for e in entries if e.provider == provider and e.api_id == key]
                if by_id:
                    return by_id[0]
                known = ", ".join(e.key for e in entries if e.provider == provider) or "none"
                raise ConfigError(
                    f"Model '{key}' is with provider '{provider}' unknown (known: {known}). "
                    f"Tip: /models discover {provider}"
                )
        if ref in self._providers:
            provider_config = self._providers[ref]
            if provider_config.default_model and f"{ref}/{provider_config.default_model}" in self._entries:
                return self._entries[f"{ref}/{provider_config.default_model}"]
            candidates = [e for e in entries if e.provider == ref and e.config.enabled]
            if not candidates:
                raise ConfigError(f"Provider '{ref}' has no (working) models. Tip: /models discover {ref}")
            candidates.sort(key=lambda e: (not self.is_available(e), -TIER_RANK[e.config.tier], e.key))
            return candidates[0]
        matches = [e for e in entries if e.key == ref or e.api_id == ref]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            available = [e for e in matches if self.is_available(e)]
            if len(available) == 1:
                return available[0]
            raise ConfigError(f"Ambiguous model reference '{ref}': {', '.join(e.ref for e in matches)}")
        raise ConfigError(f"Unknown model, provider or alias: '{reference}'")
