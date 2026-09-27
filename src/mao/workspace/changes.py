"""Tracks every file an agent creates, modifies, moves or deletes.

Before the first change to a path, the original is copied into the session's
backup folder. This enables precise diffs for reviewers, an accurate change
summary and a rollback that also covers untracked files (independent of git).
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mao.security.sandbox import WorkspaceSandbox

MAX_DIR_BACKUP_BYTES = 200 * 1024 * 1024


class FileChange(BaseModel):
    path: str
    kind: Literal["created", "modified", "deleted"]
    agent: str
    ts: float = Field(default_factory=time.time)
    backup: str | None = None
    is_dir: bool = False


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


class ChangeTracker:
    def __init__(self, sandbox: WorkspaceSandbox, backup_dir: Path, persist_path: Path | None = None) -> None:
        self.sandbox = sandbox
        self.backup_dir = backup_dir
        self.persist_path = persist_path
        self._changes: dict[str, FileChange] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ recording

    def _backup(self, rel: str, path: Path) -> str:
        destination = self.backup_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if _dir_size(path) > MAX_DIR_BACKUP_BYTES:
                raise OSError(f"Directory too large for a backup: {rel}")
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)
        return rel

    def before_write(self, path: Path, agent: str) -> None:
        rel = self.sandbox.relative(path)
        with self._lock:
            existing = self._changes.get(rel)
            if existing is None:
                if path.exists():
                    self._changes[rel] = FileChange(path=rel, kind="modified", agent=agent, backup=self._backup(rel, path), is_dir=path.is_dir())
                else:
                    self._changes[rel] = FileChange(path=rel, kind="created", agent=agent)
            elif existing.kind == "deleted":
                existing.kind = "modified" if existing.backup else "created"
                existing.agent = agent
            self._persist()

    def mark_directory_created(self, path: Path, agent: str) -> None:
        rel = self.sandbox.relative(path)
        with self._lock:
            if rel not in self._changes and not path.exists():
                self._changes[rel] = FileChange(path=rel, kind="created", agent=agent, is_dir=True)
                self._persist()

    def before_delete(self, path: Path, agent: str) -> None:
        rel = self.sandbox.relative(path)
        prefix = rel + "/"
        with self._lock:
            existing = self._changes.get(rel)
            if existing is None:
                self._changes[rel] = FileChange(
                    path=rel, kind="deleted", agent=agent, backup=self._backup(rel, path), is_dir=path.is_dir()
                )
            elif existing.kind == "created":
                del self._changes[rel]
            else:
                existing.kind = "deleted"
                existing.agent = agent
            if path.is_dir():
                for child_rel in [r for r in self._changes if r.startswith(prefix)]:
                    child = self._changes[child_rel]
                    if child.kind == "created":
                        del self._changes[child_rel]
                    else:
                        child.kind = "deleted"
            self._persist()

    # ------------------------------------------------------------------ queries

    def changes(self) -> list[FileChange]:
        with self._lock:
            return sorted(self._changes.values(), key=lambda c: c.path)

    def summary(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {"created": [], "modified": [], "deleted": []}
        for change in self.changes():
            if not change.is_dir:
                result[change.kind].append(change.path)
        return result

    def diff(self, rel: str | None = None, max_chars: int = 60_000) -> str:
        chunks: list[str] = []
        total = 0
        for change in self.changes():
            if change.is_dir or (rel is not None and change.path != rel):
                continue
            current = self.sandbox.root / change.path
            before = _read_text(self.backup_dir / change.backup) if change.backup else ""
            after = _read_text(current) if change.kind != "deleted" and current.exists() else ""
            if before is None or after is None:
                chunks.append(f"Binary file {change.kind}: {change.path}\n")
                continue
            lines = difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{change.path}" if change.kind != "created" else "/dev/null",
                tofile=f"b/{change.path}" if change.kind != "deleted" else "/dev/null",
            )
            text = "".join(lines)
            if not text:
                continue
            chunks.append(text if text.endswith("\n") else text + "\n")
            total += len(text)
            if total > max_chars:
                chunks.append("… (diff truncated)\n")
                break
        return "".join(chunks)

    # ------------------------------------------------------------------ rollback

    def rollback(self, paths: list[str] | None = None) -> list[str]:
        restored: list[str] = []
        with self._lock:
            targets = [c for c in self._changes.values() if paths is None or c.path in paths]
            # files before directories; deepest paths first for created entries
            targets.sort(key=lambda c: (c.is_dir, -c.path.count("/")))
            for change in targets:
                current = self.sandbox.root / change.path
                if change.kind == "created":
                    if current.is_dir():
                        try:
                            current.rmdir()
                        except OSError:
                            continue
                    elif current.exists():
                        current.unlink()
                elif change.backup:
                    source = self.backup_dir / change.backup
                    if current.exists():
                        if current.is_dir():
                            shutil.rmtree(current)
                        else:
                            current.unlink()
                    current.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        shutil.copytree(source, current)
                    else:
                        shutil.copy2(source, current)
                restored.append(change.path)
                del self._changes[change.path]
            self._persist()
        return restored

    # ------------------------------------------------------------------ persistence

    def _persist(self) -> None:
        if self.persist_path is None:
            return
        data = [c.model_dump() for c in self._changes.values()]
        tmp = self.persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.persist_path)

    def load(self) -> None:
        if self.persist_path is None or not self.persist_path.exists():
            return
        data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        with self._lock:
            self._changes = {entry["path"]: FileChange(**entry) for entry in data}
