"""CLI tests: the real entry point in a subprocess (plain output mode)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mao.demo import create_demo_workspace


def run_cli(tmp_path: Path, *args: str, input_text: str = "", timeout: int = 300) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "LOCALAPPDATA": str(tmp_path / "localappdata"),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            "PYTHONIOENCODING": "utf-8",
            "COLUMNS": "160",
            "NO_COLOR": "1",
        }
    )
    env.pop("MAO_HOME", None)
    return subprocess.run(
        [sys.executable, "-m", "mao", "--home", str(tmp_path / "home"), "--plain", "--no-discover", *args],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
    )


def test_init_creates_config(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "init")
    assert result.returncode == 0, result.stderr
    for name in ("settings.yaml", "providers.yaml", "agents.yaml", "roles.yaml", "tools.yaml", "permissions.yaml"):
        assert (tmp_path / "home" / "config" / name).exists()


def test_basic_commands(tmp_path: Path) -> None:
    result = run_cli(
        tmp_path,
        "-c", "/help",
        "-c", "/config validate",
        "-c", "/max-cost 3",
        "-c", "/providers",
        "-c", "/agents roles",
        "-c", "/models anthropic",
        "-c", "/plna something",
    )
    out = result.stdout
    assert result.returncode == 0, result.stderr[-3_000:] + out[-3_000:]
    assert "/plan" in out and "/max-rounds" in out
    assert "The configuration is valid" in out
    assert "$3.00" in out
    assert "anthropic" in out and "no key" in out
    assert "security" in out and "project_manager" in out
    assert "claude-sonnet-5" in out
    assert "Did you mean: /plan" in out
    assert "max_cost_usd: 3.0" in (tmp_path / "home" / "config" / "settings.yaml").read_text(encoding="utf-8")


def test_demo_plan_and_run_via_cli(tmp_path: Path) -> None:
    workspace = create_demo_workspace(tmp_path / "ws")
    result = run_cli(
        tmp_path,
        "--demo",
        "-c", f"/workspace {workspace}",
        "-c", "/plan Fix all errors in the calculator",
        "-c", "/tokens",
        "-c", "/sessions",
        "-c", "/changes",
        "-c", "/messages 5",
        input_text="y\n",
    )
    out = result.stdout
    assert result.returncode == 0, result.stderr[-3_000:] + out[-3_000:]
    assert "Plan created" in out and "TASK COMPLETED" in out
    assert "return a + b" in (workspace / "calculator.py").read_text(encoding="utf-8")
    assert "~ calculator.py" in out and "completed" in out
    assert "Tokens pro Agent" in out
