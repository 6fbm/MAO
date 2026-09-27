"""Shared knowledge store: findings, research with sources, risks, decisions and results."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from mao.core.text import one_line, truncate_end
from mao.core.types import new_id
from mao.security.redaction import Redactor

# \w keeps accented characters, so words from non-English sources still tokenize.
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "not", "all", "can", "use",
    "has", "have", "had", "but", "its", "his", "her", "they", "them", "you", "your", "our", "out",
    "into", "than", "then", "when", "what", "which", "who", "how", "why", "any", "each", "some",
}


def tokenize(text: str | None) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS]


class Source(BaseModel):
    url: str
    title: str = ""


class BoardEntry(BaseModel):
    id: str = Field(default_factory=lambda: new_id("bb_"))
    ts: float = Field(default_factory=time.time)
    author: str
    kind: str
    title: str
    content: str
    summary: str = ""
    sources: list[Source] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    importance: int = 3
    node_id: str | None = None

    def render(self, *, full: bool = False, max_chars: int = 2_000) -> str:
        head = f"[{self.id}] ({self.kind}, {self.author}) {self.title}"
        body = truncate_end(self.content, max_chars) if full else self.summary
        lines = [head, body]
        if self.files:
            lines.append("Files: " + ", ".join(self.files[:10]))
        if self.sources:
            lines.append("Sources: " + ", ".join(s.url for s in self.sources[:8]))
        return "\n".join(line for line in lines if line)


class Blackboard:
    def __init__(self, redactor: Redactor, store_path: Path | None = None) -> None:
        self.redactor = redactor
        self.store_path = store_path
        self._entries: list[BoardEntry] = []
        self._lock = threading.Lock()

    def add(
        self,
        author: str,
        kind: str,
        title: str,
        content: str,
        *,
        sources: list[Source] | list[dict] | None = None,
        files: list[str] | None = None,
        tags: list[str] | None = None,
        importance: int = 3,
        node_id: str | None = None,
        summary: str | None = None,
    ) -> BoardEntry:
        clean = self.redactor.redact(content)
        entry = BoardEntry(
            author=author,
            kind=kind,
            title=one_line(self.redactor.redact(title), 160),
            content=clean,
            summary=one_line(summary or clean, 300),
            sources=[s if isinstance(s, Source) else Source(**s) for s in sources or []],
            files=files or [],
            tags=tags or [],
            importance=importance,
            node_id=node_id,
        )
        with self._lock:
            self._entries.append(entry)
            self._persist()
        return entry

    def get(self, entry_id: str) -> BoardEntry | None:
        with self._lock:
            return next((e for e in self._entries if e.id == entry_id), None)

    def entries(self, kind: str | None = None) -> list[BoardEntry]:
        with self._lock:
            return [e for e in self._entries if kind is None or e.kind == kind]

    def by_node(self, node_id: str, kind: str | None = None) -> list[BoardEntry]:
        return [e for e in self.entries(kind) if e.node_id == node_id]

    def search(
        self,
        query: str | None,
        *,
        kinds: list[str] | None = None,
        limit: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[BoardEntry]:
        candidates = [
            e for e in self.entries() if (kinds is None or e.kind in kinds) and (not exclude_ids or e.id not in exclude_ids)
        ]
        terms = set(tokenize(query))
        newest = max((e.ts for e in candidates), default=0.0)
        scored: list[tuple[float, BoardEntry]] = []
        for entry in candidates:
            if terms:
                title = tokenize(entry.title)
                body = tokenize(entry.summary + " " + entry.content[:4_000])
                extra = tokenize(" ".join(entry.files + entry.tags))
                score = sum(3 for w in title if w in terms) + sum(1 for w in body if w in terms) + sum(2 for w in extra if w in terms)
                if score == 0:
                    continue
            else:
                score = 1.0
            score += entry.importance * 0.5 + (1.0 if entry.ts == newest else 0.0)
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], -item[1].ts))
        return [entry for _, entry in scored[:limit]]

    def _persist(self) -> None:
        if self.store_path is None:
            return
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([e.model_dump() for e in self._entries], ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.store_path)

    def load(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        data = json.loads(self.store_path.read_text(encoding="utf-8"))
        with self._lock:
            self._entries = [BoardEntry.model_validate(item) for item in data]
