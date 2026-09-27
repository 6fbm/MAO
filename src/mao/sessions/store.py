"""Sessions on disk: logs/session_<date>_<time>_<id>/…"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mao.core.errors import SessionError
from mao.orchestration.plan import PlanDocument

_DIR_RE = re.compile(r"^session_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(\d+)$")


class SessionStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SessionMeta(BaseModel):
    id: str
    dir_name: str
    kind: str = "task"  # task | debate
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    task: str
    status: SessionStatus = SessionStatus.CREATED
    workspace: str | None = None
    plan_approved: bool = False
    total_tokens: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    git_checkpoint: str | None = None
    git_branch: str | None = None
    summary: str = ""
    error: str | None = None


class SessionResult(BaseModel):
    status: str
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    bugs_fixed: int = 0
    tests: dict[str, Any] | None = None
    agents_used: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = True
    duration_s: float = 0.0
    nodes_done: int = 0
    nodes_failed: list[str] = Field(default_factory=list)
    nodes_skipped: list[str] = Field(default_factory=list)
    git: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    highlights: list[str] = Field(default_factory=list)
    remaining_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    error: str | None = None


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Session:
    def __init__(self, root: Path, meta: SessionMeta) -> None:
        self.meta = meta
        self.dir = root / meta.dir_name

    @property
    def id(self) -> str:
        return self.meta.id

    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def backups_dir(self) -> Path:
        return self.dir / "backups"

    def save_meta(self) -> None:
        self.meta.updated_at = time.time()
        _write_json(self.path("session.json"), self.meta.model_dump(mode="json"))

    def save_plan(self, plan: PlanDocument) -> None:
        _write_json(self.path("plan.json"), plan.model_dump(mode="json"))

    def load_plan(self) -> PlanDocument | None:
        path = self.path("plan.json")
        if not path.exists():
            return None
        return PlanDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save_result(self, result: SessionResult) -> None:
        _write_json(self.path("result.json"), result.model_dump(mode="json"))

    def load_result(self) -> SessionResult | None:
        path = self.path("result.json")
        if not path.exists():
            return None
        return SessionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def set_status(self, status: SessionStatus, *, error: str | None = None) -> None:
        self.meta.status = status
        if error is not None:
            self.meta.error = error
        self.save_meta()


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _existing_dirs(self) -> list[tuple[int, Path]]:
        if not self.root.exists():
            return []
        found = []
        for entry in self.root.iterdir():
            match = _DIR_RE.match(entry.name)
            if entry.is_dir() and match:
                found.append((int(match.group(1)), entry))
        return sorted(found)

    def create(self, task: str, workspace: str | None, *, kind: str = "task") -> Session:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self._existing_dirs()
        number = (existing[-1][0] + 1) if existing else 1
        session_id = f"{number:03d}"
        dir_name = f"session_{datetime.now():%Y-%m-%d_%H-%M-%S}_{session_id}"
        meta = SessionMeta(id=session_id, dir_name=dir_name, task=task, workspace=workspace, kind=kind)
        session = Session(self.root, meta)
        session.dir.mkdir(parents=True)
        session.backups_dir.mkdir()
        session.save_meta()
        return session

    def list(self) -> list[SessionMeta]:
        metas = []
        for _number, directory in self._existing_dirs():
            path = directory / "session.json"
            if not path.exists():
                continue
            try:
                metas.append(SessionMeta.model_validate(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, ValueError):
                continue
        return metas

    def load(self, session_id: str) -> Session:
        wanted = session_id.strip().lstrip("#")
        if not wanted.isdigit():
            raise SessionError(f"Invalid session ID: {session_id}")
        for number, directory in self._existing_dirs():
            if number == int(wanted):
                path = directory / "session.json"
                if not path.exists():
                    break
                meta = SessionMeta.model_validate(json.loads(path.read_text(encoding="utf-8")))
                return Session(self.root, meta)
        raise SessionError(f"Session {session_id} not found")

    def latest(self) -> Session | None:
        metas = self.list()
        return self.load(metas[-1].id) if metas else None
