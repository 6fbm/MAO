"""CollaborationHub – the implementation behind the collaboration tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from mao.core.errors import ToolError
from mao.messaging.blackboard import Blackboard, Source
from mao.messaging.bus import MessageBus

ConsultHandler = Callable[[str, str, str, str | None, int], Awaitable[str]]


class CollaborationHub:
    def __init__(
        self,
        bus: MessageBus,
        board: Blackboard,
        *,
        consult_handler: ConsultHandler | None = None,
        max_consultations_per_task: int = 3,
    ) -> None:
        self.bus = bus
        self.board = board
        self.consult_handler = consult_handler
        self.max_consultations = max_consultations_per_task
        self._consultations: dict[tuple[str, str | None], int] = {}

    def send_message(self, sender: str, recipients: list[str], kind: str, content: str, topic: str | None = None) -> str:
        return self.bus.post(sender, recipients, kind, content, topic=topic).id

    def post_finding(
        self,
        author: str,
        title: str,
        content: str,
        *,
        kind: str = "finding",
        sources: list[str] | None = None,
        files: list[str] | None = None,
        importance: int = 3,
        node_id: str | None = None,
    ) -> str:
        entry = self.board.add(
            author,
            kind,
            title,
            content,
            sources=[Source(url=url) for url in sources or []],
            files=files,
            importance=importance,
            node_id=node_id,
        )
        self.bus.post(author, ["*"], "finding", f"{title}: {entry.summary}", topic=node_id, summary=f"{kind}: {title}")
        return entry.id

    def read_board(self, query: str | None, kind: str | None, limit: int) -> str:
        if query:
            exact = self.board.get(query.strip())
            if exact is not None:
                return exact.render(full=True, max_chars=12_000)
        entries = self.board.search(query, kinds=[kind] if kind else None, limit=limit)
        if not entries:
            return "No matching entries on the blackboard."
        return "\n\n---\n\n".join(e.render(full=True, max_chars=4_000) for e in entries)

    def record_sources(self, author: str, query: str, hits: list[dict[str, str]], node_id: str | None = None) -> None:
        if not hits:
            return
        lines = [f"- {h.get('title', '')}: {h.get('url', '')}\n  {h.get('snippet', '')[:300]}" for h in hits]
        self.board.add(
            author,
            "research",
            query,
            "\n".join(lines),
            sources=[Source(url=h.get("url", ""), title=h.get("title", "")) for h in hits if h.get("url")],
            importance=2,
            node_id=node_id,
            summary=f"{len(hits)} sources for: {query}",
        )

    async def consult(self, requester: str, target: str, question: str, node_id: str | None = None, depth: int = 0) -> str:
        if self.consult_handler is None:
            raise ToolError("Consultations are not available")
        key = (requester, node_id)
        if self._consultations.get(key, 0) >= self.max_consultations:
            raise ToolError(f"Limit of {self.max_consultations} consultations reached for this task")
        self._consultations[key] = self._consultations.get(key, 0) + 1
        question_message = self.bus.post(requester, [target], "question", question, topic=node_id)
        answer = await self.consult_handler(requester, target, question, node_id, depth + 1)
        self.bus.post(target, [requester], "answer", answer, topic=node_id, reply_to=question_message.id)
        return answer
