"""Automatic model selection by task profile, capabilities, cost and diversity."""

from __future__ import annotations

from dataclasses import dataclass, field

from mao.config.schema import TIER_RANK
from mao.models.catalog import CatalogEntry, ModelCatalog

# desired tier and capabilities that earn a bonus, per task purpose
TASK_PROFILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "planning": ("strong", ("reasoning",)),
    "architecture": ("strong", ("reasoning", "long_context")),
    "analysis": ("balanced", ("reasoning", "long_context")),
    "research": ("balanced", ("web_search", "long_context")),
    "coding": ("strong", ("coding",)),
    "review": ("strong", ("coding", "reasoning")),
    "testing": ("balanced", ("coding",)),
    "debugging": ("strong", ("coding", "reasoning")),
    "security": ("strong", ("reasoning", "coding")),
    "critique": ("balanced", ("reasoning",)),
    "documentation": ("balanced", ()),
    "management": ("balanced", ("reasoning",)),
    "performance": ("strong", ("coding", "reasoning")),
    "judge": ("strong", ("reasoning",)),
    "summary": ("fast", ()),
}

_COST_WEIGHT = {"quality": 0.5, "balanced": 2.5, "cheap": 6.0}
_OVER_TIER_FACTOR = {"quality": 0.6, "balanced": -0.6, "cheap": -1.8}


@dataclass
class SelectionRequest:
    purpose: str = "analysis"
    preferred_tier: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    local_only: bool = False
    exclude_refs: set[str] = field(default_factory=set)
    provider_usage: dict[str, int] = field(default_factory=dict)
    policy: str | None = None


def blended_price(entry: CatalogEntry) -> float:
    pricing = entry.config.pricing
    return pricing.input_per_mtok * 0.75 + pricing.output_per_mtok * 0.25


class ModelSelector:
    def __init__(self, catalog: ModelCatalog, policy: str = "balanced") -> None:
        self.catalog = catalog
        self.policy = policy

    def rank(self, request: SelectionRequest) -> list[tuple[float, CatalogEntry]]:
        desired_name, bonus_caps = TASK_PROFILES.get(request.purpose, TASK_PROFILES["analysis"])
        desired = TIER_RANK[request.preferred_tier or desired_name]
        policy = request.policy or self.policy
        candidates = [
            e
            for e in self.catalog.available_entries(selectable_only=True)
            if e.ref not in request.exclude_refs and (e.local or not request.local_only)
        ]
        max_price = max((blended_price(e) for e in candidates), default=0.0)
        ranked: list[tuple[float, CatalogEntry]] = []
        for entry in candidates:
            caps = set(entry.config.capabilities)
            if any(cap not in caps for cap in request.required_capabilities if cap != "tools"):
                continue
            if "tools" in request.required_capabilities and entry.config.tool_calling == "none":
                continue
            tier = TIER_RANK[entry.config.tier]
            if tier >= desired:
                fit = 10.0 + (tier - desired) * _OVER_TIER_FACTOR.get(policy, -0.6)
            else:
                fit = 10.0 - (desired - tier) * 3.0
            bonus = sum(1.5 for cap in bonus_caps if cap in caps)
            cost_penalty = _COST_WEIGHT.get(policy, 2.5) * (blended_price(entry) / max_price) if max_price > 0 else 0.0
            tool_bonus = 1.0 if entry.config.tool_calling == "native" else -2.0
            diversity = -0.8 * request.provider_usage.get(entry.provider, 0)
            ranked.append((fit + bonus - cost_penalty + tool_bonus + diversity, entry))
        ranked.sort(key=lambda item: (-item[0], item[1].ref))
        return ranked

    def select(self, request: SelectionRequest) -> CatalogEntry | None:
        ranked = self.rank(request)
        return ranked[0][1] if ranked else None

    def fallbacks(self, primary: CatalogEntry, request: SelectionRequest, count: int = 2) -> list[CatalogEntry]:
        """Next best models, preferring other providers (a provider outage hits all its models)."""
        ranked = [e for _, e in self.rank(SelectionRequest(**{**request.__dict__, "provider_usage": {}})) if e.ref != primary.ref]
        other_providers = [e for e in ranked if e.provider != primary.provider]
        same_provider = [e for e in ranked if e.provider == primary.provider]
        result: list[CatalogEntry] = []
        seen_providers: set[str] = set()
        for entry in other_providers:
            if entry.provider not in seen_providers:
                result.append(entry)
                seen_providers.add(entry.provider)
            if len(result) >= count:
                return result
        for entry in same_provider + other_providers:
            if entry not in result:
                result.append(entry)
            if len(result) >= count:
                break
        return result
