"""Session, log, message and blackboard commands."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text

from mao.cli.commands.base import CommandContext, command, require_args
from mao.cli.render import plan_view, result_view, sessions_table
from mao.core.errors import MaoError
from mao.core.text import one_line
from mao.sessions.logger import LOG_FILES


@command(
    "sessions",
    summary="List stored sessions, load one, or show its result",
    usage="/sessions | /sessions load <id> | /sessions show <id>",
    group="Sessions",
    subcommands=("load", "show"),
)
async def sessions_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    if not args:
        metas = app.store.list()
        if not metas:
            ctx.ui.info("No sessions yet.")
            return
        ctx.ui.print(sessions_table(metas[-30:]))
        return
    sub = args[0].lower()
    require_args(args, 2, f"/sessions {sub} <id>")
    if sub == "load":
        ctx.repl.ensure_idle()
        srt = await app.orchestrator.load_session(args[1])
        ctx.ui.success(f"Session {srt.session.id} loaded: {one_line(srt.session.meta.task, 80)} ({srt.session.meta.status.value})")
        if srt.plan is not None:
            ctx.ui.print(plan_view(srt.plan))
            ctx.ui.info("Run/resume with /run")
        return
    if sub == "show":
        session = app.store.load(args[1])
        meta = session.meta
        ctx.ui.print(Text(f"Session {meta.id} – {meta.status.value}", style="bold"))
        ctx.ui.print(f"Task:      {meta.task}")
        ctx.ui.print(f"Workspace: {meta.workspace}")
        ctx.ui.print(f"Created:   {datetime.fromtimestamp(meta.created_at):%Y-%m-%d %H:%M}   Folder: {session.dir}")
        if meta.error:
            ctx.ui.warn(meta.error)
        result = session.load_result()
        if result is not None:
            ctx.ui.print(result_view(result))
        elif (plan := session.load_plan()) is not None:
            ctx.ui.print(plan_view(plan))
        return
    raise MaoError(f"Unknown subcommand: {sub}")


@command(
    "logs",
    summary="Show the log files of the current session or of one session",
    usage="/logs [orchestration|agents|tools|errors] [--tail N] [--session ID]",
    group="Sessions",
    subcommands=LOG_FILES,
)
async def logs_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    app = ctx.app
    tokens = list(args)
    tail = 40
    session_id = None
    if "--tail" in tokens:
        index = tokens.index("--tail")
        tail = int(tokens[index + 1]) if index + 1 < len(tokens) else tail
        del tokens[index : index + 2]
    if "--session" in tokens:
        index = tokens.index("--session")
        session_id = tokens[index + 1] if index + 1 < len(tokens) else None
        del tokens[index : index + 2]
    name = tokens[0] if tokens else "orchestration"
    if name not in LOG_FILES:
        raise MaoError(f"Log must be one of {', '.join(LOG_FILES)}")
    if session_id:
        directory = app.store.load(session_id).dir
    elif app.orchestrator.active is not None:
        directory = app.orchestrator.active.session.dir
    else:
        latest = app.store.latest()
        if latest is None:
            raise MaoError("No sessions available.")
        directory = latest.dir
    path = directory / f"{name}.log"
    if not path.exists():
        ctx.ui.info(f"{path} does not exist (yet).")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]
    ctx.ui.print(Text(str(path), style="dim"))
    for line in lines:
        style = "red" if " ERROR " in line else ("yellow" if " WARNING " in line else "")
        ctx.ui.print(Text(line, style=style))


@command("messages", summary="Agent-to-agent communication of the active session", usage="/messages [count]", group="Sessions")
async def messages_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    count = int(args[0]) if args and args[0].isdigit() else 25
    messages = srt.messages.all()[-count:]
    if not messages:
        ctx.ui.info("No messages yet.")
    for message in messages:
        head = Text(f"{datetime.fromtimestamp(message.ts):%H:%M:%S} ", style="dim")
        head.append(f"{message.sender} → {', '.join(message.recipients)} ", style="bold")
        head.append(f"[{message.kind}]", style="cyan")
        ctx.ui.print(head)
        ctx.ui.print(Text("  " + message.content[:2_000].replace("\n", "\n  ")))


@command("board", summary="Shared blackboard (findings, research, decisions, results)", usage="/board [kind] [search]", group="Sessions")
async def board_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    kinds = {"finding", "research", "risk", "recommendation", "decision", "result", "workspace", "plan"}
    kind = args[0] if args and args[0] in kinds else None
    query = " ".join(args[1:] if kind else args) or None
    entries = srt.board.search(query, kinds=[kind] if kind else None, limit=20) if query else srt.board.entries(kind)[-20:]
    if not entries:
        ctx.ui.info("No entries.")
    for entry in entries:
        ctx.ui.print(Text(f"[{entry.id}] {entry.kind} · {entry.author} · {entry.title}", style="bold"))
        ctx.ui.print(Text("  " + one_line(entry.summary, 300)))
        if entry.sources:
            ctx.ui.print(Text("  Sources: " + ", ".join(s.url for s in entry.sources[:6]), style="dim"))


@command("decisions", summary="Decisions taken in the active session", group="Sessions")
async def decisions_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    decisions = srt.board.entries("decision")
    if not decisions:
        ctx.ui.info("No decisions yet.")
    for entry in decisions:
        ctx.ui.print(Text(f"{entry.title}", style="bold green"))
        ctx.ui.print(Text("  " + entry.content.replace("\n", "\n  ")))
