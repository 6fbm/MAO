from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mao.config.loader import ConfigManager
from mao.core.events import Event, EventBus


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def recorded_events(event_bus: EventBus) -> list[Event]:
    events: list[Event] = []
    event_bus.subscribe(events.append)
    return events


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    return root


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "config"
    ConfigManager(directory).ensure_templates()
    return directory


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
