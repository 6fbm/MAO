"""Change tracking, diff, rollback and git commands."""

from __future__ import annotations

from rich.syntax import Syntax
from rich.text import Text

from mao.cli.commands.base import CommandContext, command, require_args
from mao.core.errors import MaoError


@command("changes", summary="Files changed by agents in the active session", group="Workspace & Git")
async def changes_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    changes = srt.changes.changes()
    if not changes:
        ctx.ui.info("No changes in this session.")
        return
    styles = {"created": ("+", "green"), "modified": ("~", "yellow"), "deleted": ("-", "red")}
    for change in changes:
        symbol, style = styles[change.kind]
        line = Text(f"{symbol} {change.path}", style=style)
        line.append(f"  ({change.kind}, {change.agent}{', Backup' if change.backup else ''})", style="dim")
        ctx.ui.print(line)


@command("diff", summary="Diff of all changes in the session (or of one file)", usage="/diff [path]", group="Workspace & Git")
async def diff_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.require_active()
    diff = srt.changes.diff(args[0] if args else None, max_chars=200_000)
    if not diff:
        ctx.ui.info("No diff.")
        return
    ctx.ui.print(Syntax(diff, "diff", theme="ansi_dark", word_wrap=False))


@command("rollback", summary="Restore the session's changes from the backups", usage="/rollback [path …]", group="Workspace & Git")
async def rollback_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    ctx.repl.ensure_idle()
    srt = ctx.app.orchestrator.require_active()
    changes = srt.changes.changes()
    targets = [c for c in changes if not args or c.path in args]
    if not targets:
        ctx.ui.info("Nothing to restore.")
        return
    for change in targets:
        ctx.ui.print(f"  {change.kind}: {change.path}")
    if not await ctx.ui.confirm(f"{len(targets)} change(s) (newly created files will be deleted)?", default=False):
        return
    restored = await ctx.app.orchestrator.rollback(args or None)
    ctx.ui.success(f"{len(restored)} path(s) restored.")


@command(
    "git",
    summary="Git status, diff, log, commit and checkpoints",
    usage="/git status | /git diff [path] | /git log | /git commit <message> | /git checkpoints",
    group="Workspace & Git",
    subcommands=("status", "diff", "log", "commit", "checkpoints"),
)
async def git_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    srt = ctx.app.orchestrator.active
    if srt is None or srt.git is None:
        from pathlib import Path

        from mao.workspace.git import GitRepo

        workspace = ctx.app.config.settings.workspace
        repo = await GitRepo.detect(Path(workspace)) if workspace else None
    else:
        repo = srt.git
    if repo is None:
        raise MaoError("The workspace is not a git repository (or git is missing).")
    sub = args[0].lower() if args else "status"
    if sub == "status":
        ctx.ui.print(await repo.status())
    elif sub == "diff":
        diff = await repo.diff(args[1:] or None)
        ctx.ui.print(Syntax(diff or "(no changes)", "diff", theme="ansi_dark"))
    elif sub == "log":
        ctx.ui.print(await repo.log(15))
    elif sub == "commit":
        require_args(args, 2, "/git commit <message>")
        message = raw.split(None, 1)[1]
        if not await ctx.ui.confirm(f"Commit all changes with message '{message}'?", default=True):
            return
        sha = await repo.commit(message)
        ctx.ui.success(f"Commit {sha[:10]} created.")
    elif sub == "checkpoints":
        rows = await repo.list_checkpoints()
        if not rows:
            ctx.ui.info("No mao checkpoints.")
        for label, sha in rows:
            ctx.ui.print(f"{label}  {sha[:12]}  (restore with e.g.: git stash apply {sha[:12]})")
    else:
        raise MaoError(f"Unknown subcommand: {sub}")
