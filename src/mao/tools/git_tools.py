"""Git tools for agents (status, diff, log, branch, commit, restore)."""

from __future__ import annotations

from typing import Any

from mao.core.text import truncate_middle
from mao.security.permissions import Capability
from mao.security.risk import ActionKind, ProposedAction, RiskLevel
from mao.tools.base import Tool, ToolContext, ToolResult, object_schema


def _no_repo() -> ToolResult:
    return ToolResult.failure("The workspace is not a git repository.")


class GitStatusTool(Tool):
    name = "git_status"
    group = "git"
    description = "Show git status (branch and changed files)."
    parameters = object_schema({})
    required = frozenset({Capability.GIT_READ})

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.git is None:
            return _no_repo()
        return ToolResult.success((await ctx.git.status()).strip() or "(clean)")


class GitDiffTool(Tool):
    name = "git_diff"
    group = "git"
    description = "Show the git diff of the working tree (optionally for one path or staged changes)."
    parameters = object_schema({"path": {"type": "string"}, "staged": {"type": "boolean"}})
    required = frozenset({Capability.GIT_READ})

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.git is None:
            return _no_repo()
        paths = [ctx.sandbox.relative(ctx.sandbox.resolve(args["path"]))] if args.get("path") else None
        diff = await ctx.git.diff(paths, staged=bool(args.get("staged")))
        return ToolResult.success(truncate_middle(diff, 30_000)[0] or "(no changes)")


class GitLogTool(Tool):
    name = "git_log"
    group = "git"
    description = "Show recent commits."
    parameters = object_schema({"max_count": {"type": "integer"}})
    required = frozenset({Capability.GIT_READ})

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.git is None:
            return _no_repo()
        return ToolResult.success(await ctx.git.log(int(args.get("max_count") or 10)))


class GitCreateBranchTool(Tool):
    name = "git_create_branch"
    group = "git"
    description = "Create and switch to a new branch."
    parameters = object_schema({"name": {"type": "string"}}, ["name"])
    required = frozenset({Capability.GIT_WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.git is None:
            return _no_repo()
        name = str(args["name"]).strip()
        await ctx.approval.require(ProposedAction(kind=ActionKind.GIT_WRITE, agent=ctx.agent_name, target=f"branch {name}", risk=RiskLevel.MEDIUM))
        await ctx.git.create_branch(name)
        return ToolResult.success(f"Branch created and checked out: {name}")


class GitCommitTool(Tool):
    name = "git_commit"
    group = "git"
    description = "Stage and commit changes (all changes or the given paths)."
    parameters = object_schema(
        {"message": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
        ["message"],
    )
    required = frozenset({Capability.GIT_WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.git is None:
            return _no_repo()
        paths = [ctx.sandbox.relative(ctx.sandbox.resolve(p)) for p in args.get("paths") or []] or None
        message = str(args["message"]).strip()
        if not message:
            return ToolResult.failure("The commit message must not be empty")
        await ctx.approval.require(ProposedAction(kind=ActionKind.GIT_WRITE, agent=ctx.agent_name, target=f"commit: {message}", risk=RiskLevel.MEDIUM))
        sha = await ctx.git.commit(message, paths)
        return ToolResult.success(f"Commit created: {sha[:10]} {message}")


class GitRestoreTool(Tool):
    name = "git_restore"
    group = "git"
    description = "Discard working-tree changes of the given files (restores the committed version). Destructive."
    parameters = object_schema({"paths": {"type": "array", "items": {"type": "string"}}}, ["paths"])
    required = frozenset({Capability.GIT_WRITE, Capability.WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.git is None:
            return _no_repo()
        resolved = [ctx.sandbox.resolve(p) for p in args.get("paths") or []]
        if not resolved:
            return ToolResult.failure("No paths given")
        rels = [ctx.sandbox.relative(p) for p in resolved]
        await ctx.approval.require(
            ProposedAction(kind=ActionKind.GIT_DESTRUCTIVE, agent=ctx.agent_name, target="git restore " + " ".join(rels), risk=RiskLevel.HIGH, reasons=["discards changes"])
        )
        for path in resolved:
            if path.exists():
                ctx.changes.before_write(path, ctx.agent_name)
        await ctx.git.restore_paths(rels)
        return ToolResult.success(f"Restored: {', '.join(rels)}")


TOOLS = [GitStatusTool, GitDiffTool, GitLogTool, GitCreateBranchTool, GitCommitTool, GitRestoreTool]
