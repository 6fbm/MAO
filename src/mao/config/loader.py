"""Loading, validating and saving the YAML configuration files."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from mao.config.schema import (
    AgentsFile,
    AppConfig,
    PermissionsConfig,
    ProvidersFile,
    RolesFile,
    Settings,
    ToolsConfig,
)
from mao.core.errors import ConfigError

CONFIG_FILES: dict[str, tuple[str, type[BaseModel]]] = {
    "settings": ("settings.yaml", Settings),
    "providers": ("providers.yaml", ProvidersFile),
    "agents": ("agents.yaml", AgentsFile),
    "roles": ("roles.yaml", RolesFile),
    "tools": ("tools.yaml", ToolsConfig),
    "permissions": ("permissions.yaml", PermissionsConfig),
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_KEY_LINE_RE = re.compile(r"""^(?P<q>['"]?)(?P<key>[^'":#\n]+?)(?P=q)\s*:(?P<rest>.*)$""")


def interpolate_env(value: Any) -> Any:
    """Replace ``${VAR}`` / ``${VAR:-default}`` in string values."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


def template_text(file_name: str) -> str:
    return resources.files("mao.config").joinpath("templates", file_name).read_text(encoding="utf-8")


def format_validation_error(file_name: str, exc: ValidationError) -> str:
    lines = [f"Invalid configuration in {file_name}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def yaml_scalar(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10_000)
    if text.endswith("...\n"):
        text = text[: -len("...\n")]
    return text.strip()


def parse_scalar(text: str) -> Any:
    """Interpret CLI input like YAML: ``true`` -> bool, ``5`` -> int, ``null`` -> None."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def set_scalar_in_yaml_text(text: str, path: list[str], new_value: str) -> str | None:
    """Replace a scalar value in YAML text while keeping comments and layout.

    Returns ``None`` when the key path is not a simple block-style scalar; the
    caller then falls back to rewriting the whole file.
    """
    lines = text.splitlines(keepends=True)
    start, end = 0, len(lines)
    parent_indent = -1

    def content_indent(line: str) -> int | None:
        stripped = line.lstrip(" ")
        if not stripped.strip() or stripped.startswith("#"):
            return None
        return len(line) - len(stripped)

    for depth, key in enumerate(path):
        child_indent: int | None = None
        found_index: int | None = None
        for index in range(start, end):
            indent = content_indent(lines[index])
            if indent is None:
                continue
            if indent <= parent_indent:
                break
            if child_indent is None:
                child_indent = indent
            if indent != child_indent:
                continue
            match = _KEY_LINE_RE.match(lines[index].strip(" ").rstrip("\r\n"))
            if match and match.group("key").strip() == key:
                found_index = index
                break
        if found_index is None or child_indent is None:
            return None
        line = lines[found_index]
        match = _KEY_LINE_RE.match(line.strip(" ").rstrip("\r\n"))
        assert match is not None
        rest = match.group("rest")
        is_last = depth == len(path) - 1
        if is_last:
            value_part = rest.strip()
            if not value_part or value_part.startswith("#"):
                return None  # block mapping/list follows
            if value_part[0] in "{[|>&*!":
                return None
            comment = ""
            comment_match = re.search(r"\s+#.*$", rest)
            if comment_match and value_part[0] not in "'\"":
                comment = comment_match.group(0)
            newline = "\n" if line.endswith("\n") else ""
            if line.endswith("\r\n"):
                newline = "\r\n"
            lines[found_index] = " " * child_indent + f"{key}: {new_value}{comment}{newline}"
            return "".join(lines)
        if rest.strip() and not rest.strip().startswith("#"):
            return None  # inline mapping – not handled
        parent_indent = child_indent
        start = found_index + 1
        block_end = start
        while block_end < end:
            indent = content_indent(lines[block_end])
            if indent is not None and indent <= child_indent:
                break
            block_end += 1
        end = block_end
    return None


class ConfigManager:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir

    def path_for(self, section: str) -> Path:
        if section not in CONFIG_FILES:
            raise ConfigError(f"Unknown configuration section: {section}")
        return self.config_dir / CONFIG_FILES[section][0]

    # ------------------------------------------------------------------ templates

    def ensure_templates(self, overwrite: bool = False) -> list[Path]:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for file_name, _model in CONFIG_FILES.values():
            target = self.config_dir / file_name
            if overwrite or not target.exists():
                target.write_text(template_text(file_name), encoding="utf-8")
                written.append(target)
        return written

    # ------------------------------------------------------------------ reading

    def read_text(self, section: str) -> str:
        path = self.path_for(section)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_raw(self, section: str) -> dict[str, Any]:
        path = self.path_for(section)
        if not path.exists():
            return {}
        return self._parse_yaml(path.name, path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_yaml(file_name: str, text: str) -> dict[str, Any]:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML error in {file_name}: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"{file_name}: the root element must be a mapping")
        return data

    def _validate(self, section: str, raw: dict[str, Any]) -> BaseModel:
        file_name, model_cls = CONFIG_FILES[section]
        try:
            return model_cls.model_validate(interpolate_env(raw))
        except ValidationError as exc:
            raise ConfigError(format_validation_error(file_name, exc)) from exc

    def load_section(self, section: str) -> BaseModel:
        return self._validate(section, self.read_raw(section))

    def load(self) -> AppConfig:
        sections = {name: self.load_section(name) for name in CONFIG_FILES}
        return AppConfig(**sections)

    # ------------------------------------------------------------------ writing

    def _write_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def save_raw(self, section: str, data: dict[str, Any]) -> BaseModel:
        model = self._validate(section, data)
        header = f"# {CONFIG_FILES[section][0]} - saved automatically by mao (template: mao init --show)\n"
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110)
        self._write_atomic(self.path_for(section), header + body)
        return model

    def update_raw(self, section: str, mutate: Callable[[dict[str, Any]], None]) -> BaseModel:
        raw = self.read_raw(section)
        mutate(raw)
        return self.save_raw(section, raw)

    def set_value(self, dotted_path: str, value: Any) -> BaseModel:
        """Set ``section.key.sub`` (e.g. ``settings.limits.max_cost_usd``) preserving comments."""
        parts = [p for p in dotted_path.split(".") if p]
        if len(parts) < 2:
            raise ConfigError("The path must have the form <section>.<key>[.<subkey>]")
        section, key_path = parts[0], parts[1:]
        path = self.path_for(section)
        text = self.read_text(section)
        if text:
            new_text = set_scalar_in_yaml_text(text, key_path, yaml_scalar(value))
            if new_text is not None:
                raw = self._parse_yaml(path.name, new_text)
                model = self._validate(section, raw)
                self._write_atomic(path, new_text)
                return model

        def mutate(raw: dict[str, Any]) -> None:
            node = raw
            for key in key_path[:-1]:
                child = node.get(key)
                if not isinstance(child, dict):
                    child = {}
                    node[key] = child
                node = child
            node[key_path[-1]] = value

        return self.update_raw(section, mutate)
