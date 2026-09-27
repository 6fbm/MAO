"""Shell command execution with approval, secret-free environment and hard timeouts."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mao.config.schema import PermissionsConfig
from mao.core.errors import PermissionDeniedError
from mao.core.text import truncate_middle
from mao.core.types import RunMode
from mao.security.permissions import Capability
from mao.security.risk import ActionKind, ProposedAction, assess_command, matches_patterns
from mao.tools.base import Tool, ToolContext, ToolResult, object_schema

READ_CHUNK = 64 * 1024
MAX_CAPTURE_BYTES = 4 * 1024 * 1024


@dataclass
class CommandOutcome:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool


def scrubbed_environment(patterns: list[str], passthrough: list[str]) -> dict[str, str]:
    allowed = {name.upper() for name in passthrough}
    env = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper not in allowed and any(fnmatch.fnmatchcase(upper, p.upper()) for p in patterns):
            continue
        env[key] = value
    env["PYTHONIOENCODING"] = "utf-8"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def resolve_shell(setting: str) -> str:
    if setting != "auto":
        return setting
    if sys.platform == "win32":
        return "cmd"
    return "bash" if shutil.which("bash") else "sh"


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        if sys.platform == "win32":
            try:
                return data.decode("oem", errors="replace")
            except LookupError:
                pass
        return data.decode("utf-8", errors="replace")


async def _kill_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(process.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), 10)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError, asyncio.TimeoutError):
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 10)
    except asyncio.TimeoutError:
        pass


async def _drain(stream: asyncio.StreamReader | None, buffer: bytearray) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(READ_CHUNK)
        if not chunk:
            return
        if len(buffer) < MAX_CAPTURE_BYTES:
            buffer.extend(chunk[: MAX_CAPTURE_BYTES - len(buffer)])


async def run_shell_command(
    command: str,
    *,
    cwd: Path,
    shell: str,
    timeout_s: float,
    env: dict[str, str],
) -> CommandOutcome:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": env,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    started = time.monotonic()
    if shell == "cmd":
        process = await asyncio.create_subprocess_shell(command, **kwargs)
    elif shell in ("powershell", "pwsh"):
        executable = "powershell.exe" if shell == "powershell" else "pwsh"
        process = await asyncio.create_subprocess_exec(executable, "-NoProfile", "-NonInteractive", "-Command", command, **kwargs)
    else:
        process = await asyncio.create_subprocess_exec(shell, "-c", command, **kwargs)
    stdout_buffer, stderr_buffer = bytearray(), bytearray()
    readers = asyncio.gather(_drain(process.stdout, stdout_buffer), _drain(process.stderr, stderr_buffer))
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
        await _kill_tree(process)
    except asyncio.CancelledError:
        await _kill_tree(process)
        raise
    try:
        await asyncio.wait_for(readers, 5)
    except asyncio.TimeoutError:
        readers.cancel()
    return CommandOutcome(
        exit_code=process.returncode,
        stdout=_decode(bytes(stdout_buffer)),
        stderr=_decode(bytes(stderr_buffer)),
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
    )


async def authorize_command(ctx: ToolContext, command: str, cwd_rel: str) -> None:
    """Denylist, PLAN-mode restrictions and approval – shared by run_command and run_tests."""
    config: PermissionsConfig = ctx.permissions_config
    assessment = assess_command(
        command,
        workspace_root=ctx.sandbox.root,
        denylist=config.command_denylist,
        allowlist=config.command_allowlist,
    )
    if assessment.denied:
        raise PermissionDeniedError(f"Command blocked (denylist): {command}")
    if ctx.mode is RunMode.PLAN:
        if assessment.chained or not matches_patterns(command, config.plan_mode_commands):
            raise PermissionDeniedError(
                "Only read-only commands are allowed in PLAN mode (e.g. "
                + ", ".join(config.plan_mode_commands[:6])
                + ")"
            )
        return
    await ctx.approval.require(
        ProposedAction(
            kind=ActionKind.EXECUTE,
            agent=ctx.agent_name,
            target=command,
            detail=f"Working directory: {cwd_rel}",
            risk=assessment.risk,
            reasons=assessment.reasons,
        )
    )


def format_outcome(command: str, outcome: CommandOutcome, max_chars: int) -> str:
    status = "TIMEOUT" if outcome.timed_out else f"exit code {outcome.exit_code}"
    parts = [f"$ {command}", f"[{status}, {outcome.duration_s:.1f}s]"]
    half = max_chars // 2
    if outcome.stdout.strip():
        parts.append("--- stdout ---\n" + truncate_middle(outcome.stdout.rstrip(), half)[0])
    if outcome.stderr.strip():
        parts.append("--- stderr ---\n" + truncate_middle(outcome.stderr.rstrip(), half)[0])
    if not outcome.stdout.strip() and not outcome.stderr.strip():
        parts.append("(no output)")
    return "\n".join(parts)


class RunCommandTool(Tool):
    name = "run_command"
    group = "terminal"
    description = (
        "Run a non-interactive shell command in the workspace (Windows: cmd.exe). stdin is closed, "
        "so commands must not wait for input. Returns exit code, stdout and stderr."
    )
    parameters = object_schema(
        {
            "command": {"type": "string", "description": "Command line"},
            "cwd": {"type": "string", "description": "Relative working directory, default '.'"},
            "timeout_s": {"type": "integer", "description": "Timeout in seconds"},
        },
        ["command"],
    )
    required = frozenset({Capability.EXECUTE})
    mutating = True
    plan_allowed = True
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args["command"]).strip()
        if not command:
            return ToolResult.failure("Empty command")
        cwd = ctx.sandbox.resolve(args.get("cwd") or ".")
        if not cwd.is_dir():
            return ToolResult.failure(f"Working directory does not exist: {args.get('cwd')}")
        cwd_rel = ctx.sandbox.relative(cwd)
        await authorize_command(ctx, command, cwd_rel)
        terminal = ctx.tools_config.terminal
        try:
            timeout = int(args.get("timeout_s") or terminal.default_timeout_s)
        except (TypeError, ValueError):
            timeout = terminal.default_timeout_s
        timeout = max(1, min(timeout, terminal.max_timeout_s))
        outcome = await run_shell_command(
            command,
            cwd=cwd,
            shell=resolve_shell(terminal.shell),
            timeout_s=timeout,
            env=scrubbed_environment(ctx.permissions_config.scrub_env_patterns, terminal.env_passthrough),
        )
        ok = outcome.exit_code == 0 and not outcome.timed_out
        return ToolResult(
            ok=ok,
            content=format_outcome(command, outcome, terminal.max_output_chars),
            data={"exit_code": outcome.exit_code, "timed_out": outcome.timed_out},
        )


TOOLS = [RunCommandTool]
