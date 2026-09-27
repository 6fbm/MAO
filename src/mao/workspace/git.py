"""Thin async wrapper around the git command line."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

from mao.core.errors import ToolError

_BRANCH_RE = re.compile(r"^(?!.*\.\.)(?!/)(?!.*//)[A-Za-z0-9._/\-]{1,100}(?<!/)(?<!\.lock)$")


def valid_branch_name(name: str) -> bool:
    return bool(_BRANCH_RE.match(name)) and not name.startswith("-")


class GitRepo:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    async def detect(cls, path: Path) -> GitRepo | None:
        if shutil.which("git") is None:
            return None
        code, out, _ = await cls._exec(path, "rev-parse", "--show-toplevel", timeout=15)
        if code != 0 or not out.strip():
            return None
        return cls(Path(out.strip()))

    @staticmethod
    async def _exec(cwd: Path, *args: str, timeout: float = 60) -> tuple[int, str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_OPTIONAL_LOCKS"] = "0"
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(cwd),
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            return 127, "", str(exc)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return 124, "", f"git {' '.join(args)}: timeout"
        return process.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")

    async def run(self, *args: str, timeout: float = 60) -> tuple[int, str, str]:
        return await self._exec(self.root, *args, timeout=timeout)

    async def run_checked(self, *args: str, timeout: float = 60) -> str:
        code, out, err = await self.run(*args, timeout=timeout)
        if code != 0:
            raise ToolError(f"git {' '.join(args)} failed: {(err or out).strip()[:500]}")
        return out

    async def status(self) -> str:
        return await self.run_checked("status", "--short", "--branch")

    async def is_clean(self) -> bool:
        out = await self.run_checked("status", "--porcelain")
        return not out.strip()

    async def current_branch(self) -> str:
        out = await self.run_checked("rev-parse", "--abbrev-ref", "HEAD")
        return out.strip()

    async def head(self) -> str | None:
        code, out, _ = await self.run("rev-parse", "HEAD")
        return out.strip() if code == 0 else None

    async def diff(self, paths: list[str] | None = None, *, staged: bool = False, stat: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if stat:
            args.append("--stat")
        if paths:
            args += ["--", *paths]
        return await self.run_checked(*args)

    async def log(self, count: int = 10) -> str:
        return await self.run_checked("log", f"-{max(1, min(count, 100))}", "--oneline", "--decorate")

    async def create_branch(self, name: str, *, checkout: bool = True) -> None:
        if not valid_branch_name(name):
            raise ToolError(f"Invalid branch name: {name}")
        if checkout:
            await self.run_checked("switch", "-c", name)
        else:
            await self.run_checked("branch", name)

    async def commit(self, message: str, paths: list[str] | None = None) -> str:
        await self.run_checked("add", "-A", "--", *(paths or ["."]))
        code, out, err = await self.run("commit", "-m", message)
        if code != 0:
            text = (err or out).strip()
            if "nothing to commit" in text:
                raise ToolError("Nothing to commit.")
            raise ToolError(f"Commit failed: {text[:500]}")
        return (await self.head()) or ""

    async def checkpoint(self, label: str) -> str | None:
        """Record the current state without touching the working tree or index."""
        code, out, _ = await self.run("stash", "create", f"mao checkpoint {label}")
        sha = out.strip() if code == 0 and out.strip() else await self.head()
        if not sha:
            return None
        safe_label = re.sub(r"[^A-Za-z0-9._\-]", "_", label)
        await self.run_checked("update-ref", f"refs/mao/checkpoints/{safe_label}", sha)
        return sha

    async def list_checkpoints(self) -> list[tuple[str, str]]:
        code, out, _ = await self.run("for-each-ref", "--format=%(refname) %(objectname)", "refs/mao/checkpoints")
        if code != 0:
            return []
        rows = []
        for line in out.splitlines():
            if " " in line:
                ref, sha = line.split(" ", 1)
                rows.append((ref.removeprefix("refs/mao/checkpoints/"), sha))
        return rows

    async def restore_paths(self, paths: list[str]) -> None:
        await self.run_checked("restore", "--", *paths)
