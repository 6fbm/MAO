"""Resolution of the application home and its well-known locations."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "MultiAIOrchestrator"


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mao"


def _source_checkout_root() -> Path | None:
    # src/mao/paths.py -> project root when running from a checkout / editable install
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "mao").is_dir():
        return root
    return None


@dataclass(frozen=True)
class AppPaths:
    home: Path
    user_data: Path

    @classmethod
    def discover(cls, explicit_home: str | None = None) -> AppPaths:
        if explicit_home:
            home = Path(explicit_home).expanduser()
        elif os.environ.get("MAO_HOME"):
            home = Path(os.environ["MAO_HOME"]).expanduser()
        else:
            home = _source_checkout_root() or (Path.home() / ".mao")
        return cls(home=home.resolve(), user_data=_user_data_dir())

    @property
    def config_dir(self) -> Path:
        return self.home / "config"

    def logs_dir(self, configured: str = "logs") -> Path:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else self.home / path

    def secrets_file(self, configured: str | None = None) -> Path:
        if configured:
            path = Path(configured).expanduser()
            return path if path.is_absolute() else self.home / path
        return self.user_data / "secrets.dat"

    @property
    def history_file(self) -> Path:
        return self.user_data / "history.txt"
