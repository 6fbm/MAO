from __future__ import annotations

from pathlib import Path

import pytest

from mao.config.loader import ConfigManager, set_scalar_in_yaml_text
from mao.config.schema import PermissionOverrides, PermissionSet
from mao.core.errors import ConfigError
from mao.security.permissions import Capability, granted_capabilities, resolve_permissions


def test_templates_are_valid(config_dir: Path) -> None:
    config = ConfigManager(config_dir).load()
    assert {"openai", "anthropic", "gemini", "xai", "ollama", "mock"} <= set(config.providers.providers)
    assert config.providers.providers["mock"].selectable is False
    assert config.providers.providers["openai"].models["gpt-5.6-terra"].pricing.input_per_mtok == 2.0
    assert config.providers.providers["anthropic"].models["claude-haiku-4-5"].id == "claude-haiku-4-5-20251001"
    names = [a.name for a in config.agents.agents]
    assert "coder" in names and "planner" in names
    assert config.settings.limits.max_rounds == 3
    assert config.permissions.approval.delete == "ask"


def test_set_value_keeps_comments(config_dir: Path) -> None:
    manager = ConfigManager(config_dir)
    manager.set_value("settings.limits.max_cost_usd", 12.5)
    text = (config_dir / "settings.yaml").read_text(encoding="utf-8")
    assert "max_cost_usd: 12.5" in text
    assert "# estimated cost per session in USD" in text
    assert manager.load().settings.limits.max_cost_usd == 12.5


def test_set_value_top_level_path(config_dir: Path) -> None:
    manager = ConfigManager(config_dir)
    manager.set_value("settings.workspace", r"C:\Projects\Demo")
    assert manager.load().settings.workspace == r"C:\Projects\Demo"


def test_set_value_invalid_is_rejected_without_writing(config_dir: Path) -> None:
    manager = ConfigManager(config_dir)
    before = (config_dir / "settings.yaml").read_text(encoding="utf-8")
    with pytest.raises(ConfigError):
        manager.set_value("settings.limits.max_rounds", 0)
    assert (config_dir / "settings.yaml").read_text(encoding="utf-8") == before


def test_set_value_missing_key_falls_back_to_rewrite(config_dir: Path) -> None:
    manager = ConfigManager(config_dir)
    manager.set_value("tools.web.searxng_url", "http://localhost:8888")
    assert manager.load().tools.web.searxng_url == "http://localhost:8888"


def test_unknown_field_is_an_error(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    (tmp_path / "settings.yaml").write_text("limits:\n  max_tokenz: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_tokenz"):
        manager.load_section("settings")


def test_env_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAO_TEST_URL", "http://search.local")
    manager = ConfigManager(tmp_path)
    (tmp_path / "tools.yaml").write_text("web:\n  searxng_url: ${MAO_TEST_URL}\n", encoding="utf-8")
    assert manager.load_section("tools").web.searxng_url == "http://search.local"


def test_scalar_edit_ignores_block_values() -> None:
    text = "a:\n  b:\n    - 1\n  c: 2 # note\n"
    assert set_scalar_in_yaml_text(text, ["a", "b"], "3") is None
    assert set_scalar_in_yaml_text(text, ["a", "c"], "5") == "a:\n  b:\n    - 1\n  c: 5 # note\n"


def test_permission_resolution_and_ceiling() -> None:
    defaults = PermissionOverrides(filesystem="read", terminal=False, internet=False, git="read")
    role = PermissionOverrides(filesystem="write", terminal=True, git="write")
    agent = PermissionOverrides(internet=True)
    perms = resolve_permissions(defaults, role, agent)
    assert perms.write and perms.execute and perms.internet and not perms.delete and perms.git == "write"
    limited = resolve_permissions(defaults, role, agent, ceiling=PermissionOverrides(terminal=False, git="read"))
    assert not limited.execute and limited.git == "read" and limited.write
    caps = granted_capabilities(limited)
    assert Capability.WRITE in caps and Capability.GIT_WRITE not in caps and Capability.GIT_READ in caps


def test_filesystem_full_grants_delete() -> None:
    perms = PermissionOverrides(filesystem="full").apply(PermissionSet())
    assert perms.read and perms.write and perms.delete
