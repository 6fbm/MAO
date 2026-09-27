"""Loads built-in modes (package) and user modes (<home>/modes/*.yaml)."""

from __future__ import annotations

import difflib
import os
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from mao.config.loader import format_validation_error
from mao.core.errors import ConfigError
from mao.modes.schema import ModeDefinition

BUILTIN_ORDER = [
    "productive", "creative", "professional", "strict", "funny", "chaotic", "debate", "critical",
    "researcher", "coder", "hacker", "autonomous", "efficient", "safe", "evil", "balanced",
]


def _builtin_files() -> list[tuple[str, str]]:
    folder = resources.files("mao.modes").joinpath("builtin")
    return sorted((entry.name, entry.read_text(encoding="utf-8")) for entry in folder.iterdir() if entry.name.endswith(".yaml"))


def parse_mode(text: str, source: str) -> ModeDefinition:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML error in {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: the root element must be a mapping")
    data.pop("builtin", None)
    data.pop("source", None)
    try:
        definition = ModeDefinition.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(source, exc)) from exc
    definition.source = source
    return definition


class ModeRegistry:
    def __init__(self, user_dir: Path | None) -> None:
        self.user_dir = user_dir
        self._modes: dict[str, ModeDefinition] = {}
        self._builtin: set[str] = set()
        self.errors: list[str] = []
        self.reload()

    def reload(self) -> None:
        modes: dict[str, ModeDefinition] = {}
        builtin: set[str] = set()
        errors: list[str] = []
        for file_name, text in _builtin_files():
            definition = parse_mode(text, f"builtin/{file_name}")
            definition.builtin = True
            modes[definition.name] = definition
            builtin.add(definition.name)
        if self.user_dir is not None and self.user_dir.is_dir():
            for path in sorted(self.user_dir.glob("*.yaml")):
                try:
                    definition = parse_mode(path.read_text(encoding="utf-8"), str(path))
                except ConfigError as exc:
                    errors.append(str(exc))
                    continue
                if path.stem != definition.name:
                    errors.append(f"{path}: file name and name ({definition.name}) do not match - ignored")
                    continue
                definition.builtin = definition.name in builtin
                modes[definition.name] = definition
        self._modes, self._builtin, self.errors = modes, builtin, errors

    def ensure_user_templates(self) -> list[Path]:
        """Copy built-in modes into <home>/modes so they can be viewed and adapted."""
        if self.user_dir is None:
            return []
        self.user_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for file_name, text in _builtin_files():
            target = self.user_dir / file_name
            if not target.exists():
                target.write_text(text, encoding="utf-8")
                written.append(target)
        if written:
            self.reload()
        return written

    # ------------------------------------------------------------------ queries

    def exists(self, name: str) -> bool:
        return name.lower() in self._modes

    def get(self, name: str) -> ModeDefinition:
        key = name.strip().lower()
        if key in self._modes:
            return self._modes[key]
        suggestions = difflib.get_close_matches(key, list(self._modes), n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else " overview: /mode"
        raise ConfigError(f"Unknown mode '{name}'.{hint}")

    def is_builtin(self, name: str) -> bool:
        return name.lower() in self._builtin

    def all(self) -> list[ModeDefinition]:
        ordered = [self._modes[n] for n in BUILTIN_ORDER if n in self._modes]
        others = sorted((m for n, m in self._modes.items() if n not in BUILTIN_ORDER), key=lambda m: m.name)
        return ordered + others

    def custom(self) -> list[ModeDefinition]:
        return [m for m in self.all() if not m.builtin]

    # ------------------------------------------------------------------ persistence

    def save_custom(self, definition: ModeDefinition) -> Path:
        if self.user_dir is None:
            raise ConfigError("No folder configured for custom modes")
        if definition.name in self._builtin:
            raise ConfigError(f"'{definition.name}' is a built-in mode - please pick another name (or modes/{definition.name}.yaml directly)")
        self.user_dir.mkdir(parents=True, exist_ok=True)
        data = definition.model_dump(mode="json", exclude={"builtin", "source"})
        default_effects = definition.effects.__class__().model_dump(mode="json")
        data["effects"] = {k: v for k, v in data["effects"].items() if v != default_effects.get(k)}
        header = f"# Custom mode '{definition.name}' - created with /mode create. See docs/MODES.md for the fields\n"
        path = self.user_dir / f"{definition.name}.yaml"
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
        parse_mode(tmp.read_text(encoding="utf-8"), str(path))  # validate before replacing
        os.replace(tmp, path)
        self.reload()
        return path

    def delete_custom(self, name: str) -> Path:
        key = name.lower()
        if key in self._builtin:
            raise ConfigError(f"'{key}' is a built-in mode and cannot be deleted")
        if self.user_dir is None or not (self.user_dir / f"{key}.yaml").exists():
            raise ConfigError(f"No custom mode '{name}' found")
        path = self.user_dir / f"{key}.yaml"
        path.unlink()
        self.reload()
        return path
