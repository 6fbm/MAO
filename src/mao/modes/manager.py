"""Active mode state: global, per agent, orchestrator and temporary modes."""

from __future__ import annotations

import re

from mao.config.schema import Settings
from mao.core.errors import ConfigError
from mao.core.events import EventBus, ModeChangedEvent
from mao.modes.profile import BehaviorProfile, ResolvedMode, StrategyProfile, build_strategy, render_prompt, resolve
from mao.modes.registry import ModeRegistry
from mao.modes.schema import ActiveMode, ModeState

_INTENSITY_RE = re.compile(r"^(\d{1,3})%?$")


class ModeManager:
    def __init__(self, registry: ModeRegistry, settings: Settings, bus: EventBus) -> None:
        self.registry = registry
        self.settings = settings
        self.bus = bus
        self.state = ModeState()
        self.warnings: list[str] = []
        self.reset(publish=False)

    # ------------------------------------------------------------------ parsing

    def parse_entries(self, tokens: list[str]) -> list[ActiveMode]:
        """'productive 90 strict funny 20' -> [productive:90, strict:<default>, funny:20]."""
        entries: list[ActiveMode] = []
        index = 0
        cleaned = [t.strip().strip(",+").lower() for t in tokens if t.strip().strip(",+")]
        while index < len(cleaned):
            token = cleaned[index]
            if _INTENSITY_RE.match(token):
                raise ConfigError(f"Intensity '{token}' without a preceding mode")
            name, _, inline = token.partition(":")
            definition = self.registry.get(name)
            intensity = definition.default_intensity
            if inline:
                intensity = self._parse_intensity(inline)
            elif index + 1 < len(cleaned) and _INTENSITY_RE.match(cleaned[index + 1]):
                intensity = self._parse_intensity(cleaned[index + 1])
                index += 1
            entries = [e for e in entries if e.name != definition.name] + [ActiveMode(name=definition.name, intensity=intensity)]
            index += 1
        if not entries:
            raise ConfigError("No mode given. Overview: /mode")
        if len(entries) > self.settings.modes.max_active:
            raise ConfigError(f"At most {self.settings.modes.max_active} concurrent modes (settings.modes.max_active)")
        return entries

    @staticmethod
    def _parse_intensity(text: str) -> int:
        match = _INTENSITY_RE.match(text.strip())
        if not match or not 0 <= int(match.group(1)) <= 100:
            raise ConfigError(f"Invalid intensity '{text}' (0–100)")
        return int(match.group(1))

    def _entries_from_settings(self, values: list[str], fallback: str) -> list[ActiveMode]:
        try:
            return self.parse_entries([part for value in values for part in value.replace(":", " ").split()])
        except ConfigError as exc:
            self.warnings.append(f"settings.modes: {exc} - using '{fallback}'")
            return [ActiveMode(name=fallback, intensity=self.registry.get(fallback).default_intensity)]

    # ------------------------------------------------------------------ changes

    def _changed(self, scope: str, entries: list[ActiveMode] | None, target: str | None = None) -> None:
        names = [e.name for e in entries or []]
        self.bus.publish(ModeChangedEvent(scope=scope, target=target, modes=names, label=self.label()))

    def set_global(self, entries: list[ActiveMode]) -> None:
        self.state.global_modes = entries
        self._changed("global", entries)

    def set_orchestrator(self, entries: list[ActiveMode]) -> None:
        self.state.orchestrator_modes = entries
        self._changed("orchestrator", entries, "orchestrator")

    def set_agent(self, target: str, entries: list[ActiveMode]) -> None:
        key = target.lstrip("@").lower()
        if key == "orchestrator":
            self.set_orchestrator(entries)
            return
        self.state.agent_modes[key] = entries
        self._changed("agent", entries, key)

    def clear_agent(self, target: str) -> bool:
        key = target.lstrip("@").lower()
        if key == "orchestrator":
            self.state.orchestrator_modes = self._entries_from_settings([self.settings.modes.orchestrator], "balanced")
            self._changed("orchestrator", self.state.orchestrator_modes, "orchestrator")
            return True
        removed = self.state.agent_modes.pop(key, None) is not None
        if removed:
            self._changed("agent", [], key)
        return removed

    def set_temporary(self, entries: list[ActiveMode]) -> None:
        self.state.temporary_pending = entries
        self._changed("temporary", entries)

    def begin_cycle(self) -> bool:
        """A planning cycle starts: a pending temporary mode becomes active for it."""
        if self.state.temporary_pending is None:
            return False
        self.state.temporary_active = self.state.temporary_pending
        self.state.temporary_pending = None
        self._changed("temporary_active", self.state.temporary_active)
        return True

    def end_cycle(self) -> bool:
        """The planning cycle ended: restore the previous (global) modes."""
        if self.state.temporary_active is None:
            return False
        self.state.temporary_active = None
        self._changed("restored", self.state.global_modes)
        return True

    def reset(self, *, publish: bool = True) -> None:
        self.warnings = []
        self.state = ModeState(
            global_modes=self._entries_from_settings(self.settings.modes.default, "productive"),
            orchestrator_modes=self._entries_from_settings([self.settings.modes.orchestrator], "balanced"),
        )
        if publish:
            self._changed("reset", self.state.global_modes)

    # ------------------------------------------------------------------ resolution

    def effective_global(self) -> list[ActiveMode]:
        return self.state.temporary_active or self.state.global_modes

    def _resolve(self, entries: list[ActiveMode]) -> list[ResolvedMode]:
        resolved = []
        for entry in entries:
            if self.registry.exists(entry.name):
                resolved.append(ResolvedMode(self.registry.get(entry.name), entry.intensity))
        return resolved

    def profile(self) -> BehaviorProfile:
        return resolve(self._resolve(self.effective_global()))

    def strategy(self) -> StrategyProfile:
        return build_strategy(self.settings, self.profile())

    def entries_for_agent(self, name: str, config_name: str = "", role: str = "", internal: bool = False) -> list[ActiveMode]:
        if internal:
            # the orchestrator keeps its own (neutral) behavior but speaks with the global personality
            personality = [
                e for e in self.effective_global()
                if self.registry.exists(e.name) and self.registry.get(e.name).kind == "personality"
            ]
            own = list(self.state.orchestrator_modes)
            return own + [p for p in personality if p.name not in {o.name for o in own}]
        for key in (name.lower(), config_name.lower(), role.lower()):
            if key and key in self.state.agent_modes:
                return self.state.agent_modes[key]
        return self.effective_global()

    def profile_for_agent(self, name: str, config_name: str = "", role: str = "", internal: bool = False) -> BehaviorProfile:
        return resolve(self._resolve(self.entries_for_agent(name, config_name, role, internal)))

    def prompt_for_agent(self, name: str, config_name: str = "", role: str = "", internal: bool = False) -> str:
        return render_prompt(self._resolve(self.entries_for_agent(name, config_name, role, internal)))

    def label(self, entries: list[ActiveMode] | None = None) -> str:
        active = entries if entries is not None else self.effective_global()
        text = " + ".join(e.name.upper() for e in active) or "–"
        if entries is None and self.state.temporary_active:
            text += " (temporary)"
        return text

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return self.state.model_dump(mode="json")

    def load_dict(self, data: dict | None) -> None:
        if not data:
            return
        state = ModeState.model_validate(data)

        def known(entries: list[ActiveMode] | None) -> list[ActiveMode] | None:
            if entries is None:
                return None
            return [e for e in entries if self.registry.exists(e.name)]

        self.state = ModeState(
            global_modes=known(state.global_modes) or self.state.global_modes,
            orchestrator_modes=known(state.orchestrator_modes) or self.state.orchestrator_modes,
            agent_modes={k: v for k, v in ((k, known(v) or []) for k, v in state.agent_modes.items()) if v},
            temporary_pending=known(state.temporary_pending),
            temporary_active=None,
        )
        self._changed("restored", self.state.global_modes)
