"""File-system tools. Every path goes through the workspace sandbox."""

from __future__ import annotations

import asyncio
import codecs
import difflib
import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Any

from mao.core.errors import ToolError
from mao.security.permissions import Capability
from mao.security.risk import ActionKind, ProposedAction, RiskLevel
from mao.tools.base import Tool, ToolContext, ToolResult, object_schema
from mao.workspace.workspace import is_ignored, render_tree

_ENCODINGS = ("utf-8", "cp1252", "latin-1")


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def read_text_file(path: Path) -> tuple[str, str, str]:
    """Return (text, encoding, newline) or raise ToolError for binary files.

    The encoding is reported so a later write keeps it: a BOM is only written
    back if the file had one.
    """
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ToolError(f"Binary file cannot be read as text: {path.name}")
    newline = "\r\n" if b"\r\n" in data else "\n"
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace"), "utf-8-sig", newline
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding), encoding, newline
        except UnicodeDecodeError:
            continue
    raise ToolError("Unknown text encoding")  # pragma: no cover - latin-1 always decodes


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> None:
    normalized = text.replace("\r\n", "\n")
    if newline == "\r\n":
        normalized = normalized.replace("\n", "\r\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.mao-tmp")
    with open(tmp, "w", encoding=encoding, newline="") as handle:
        handle.write(normalized)
    os.replace(tmp, path)


def _walk_files(root: Path, ignore: list[str]):  # type: ignore[no-untyped-def]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_ignored(d, ignore) and not os.path.islink(os.path.join(dirpath, d))]
        for filename in filenames:
            if not is_ignored(filename, ignore):
                yield Path(dirpath) / filename


# ============================================================================ read-only


class ListDirectoryTool(Tool):
    name = "list_directory"
    group = "filesystem"
    description = "List files and folders as a tree (relative to the workspace). Build/vendor folders are skipped."
    parameters = object_schema(
        {
            "path": {"type": "string", "description": "Relative folder path, default '.'"},
            "depth": {"type": "integer", "description": "Tree depth 1-5 (default 2)"},
        }
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.sandbox.resolve(args.get("path") or ".")
        rel = ctx.sandbox.relative(target)
        if not target.exists():
            return ToolResult.failure(f"Path does not exist: {rel}")
        if target.is_file():
            return ToolResult.success(f"{rel} is a file ({target.stat().st_size} Bytes)")
        depth = _clamp(args.get("depth"), 1, 5, 2)
        fs = ctx.tools_config.filesystem
        tree = await asyncio.to_thread(render_tree, target, fs.ignore_patterns, depth=depth, max_entries=fs.max_list_entries)
        return ToolResult.success(f"{rel}/\n{tree}")


class ReadFileTool(Tool):
    name = "read_file"
    group = "filesystem"
    description = (
        "Read a text file. Lines are prefixed with their number and '| ' (the prefix is not part of the file). "
        "Use start_line/end_line for large files."
    )
    parameters = object_schema(
        {
            "path": {"type": "string", "description": "Relative file path"},
            "start_line": {"type": "integer", "description": "First line (1-based)"},
            "end_line": {"type": "integer", "description": "Last line (inclusive)"},
        },
        ["path"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.sandbox.resolve(args["path"])
        rel = ctx.sandbox.relative(path)
        if not path.is_file():
            return ToolResult.failure(f"File does not exist: {rel}")
        if ctx.sandbox.is_sensitive(path):
            await ctx.approval.require(
                ProposedAction(
                    kind=ActionKind.SENSITIVE_READ,
                    agent=ctx.agent_name,
                    target=rel,
                    detail="The file may contain secrets (its content goes to an AI model)",
                    risk=RiskLevel.MEDIUM,
                    reasons=["sensitive file"],
                )
            )
        fs = ctx.tools_config.filesystem
        size = path.stat().st_size
        if size > fs.max_file_bytes:
            return ToolResult.failure(f"File too large ({size} Bytes, limit {fs.max_file_bytes}). Use search_files.")
        text, _encoding, _newline = await asyncio.to_thread(read_text_file, path)
        lines = text.splitlines()
        total = len(lines)
        start = _clamp(args.get("start_line"), 1, max(total, 1), 1)
        end = _clamp(args.get("end_line"), start, max(total, 1), total)
        rendered: list[str] = []
        used = 0
        last = start - 1
        for number in range(start, end + 1):
            line = f"{number:>5}| {lines[number - 1]}"
            if used + len(line) > fs.max_read_chars and rendered:
                break
            rendered.append(line)
            used += len(line) + 1
            last = number
        header = f"{rel} (lines {start}-{last} of {total})"
        if last < end:
            header += f" - truncated, continue with start_line={last + 1}"
        return ToolResult.success(header + "\n" + "\n".join(rendered), total_lines=total)


class SearchFilesTool(Tool):
    name = "search_files"
    group = "filesystem"
    description = "Search file contents with a regular expression. Returns 'path:line: text' matches."
    parameters = object_schema(
        {
            "pattern": {"type": "string", "description": "Regular expression (Python syntax)"},
            "path": {"type": "string", "description": "Folder to search, default '.'"},
            "glob": {"type": "string", "description": "File name filter, e.g. '*.py'"},
            "case_sensitive": {"type": "boolean"},
            "max_results": {"type": "integer"},
        },
        ["pattern"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        flags = 0 if args.get("case_sensitive") else re.IGNORECASE
        try:
            regex = re.compile(args["pattern"], flags)
        except re.error as exc:
            return ToolResult.failure(f"Invalid regular expression: {exc}")
        base = ctx.sandbox.resolve(args.get("path") or ".")
        fs = ctx.tools_config.filesystem
        limit = _clamp(args.get("max_results"), 1, fs.max_search_results, min(100, fs.max_search_results))
        glob = args.get("glob")
        sandbox = ctx.sandbox

        def search() -> tuple[list[str], int]:
            hits: list[str] = []
            scanned = 0
            for file in _walk_files(base, fs.ignore_patterns):
                if glob and not fnmatch.fnmatch(file.name.lower(), glob.lower()):
                    continue
                if sandbox.is_sensitive(file):
                    continue
                try:
                    if file.stat().st_size > fs.max_file_bytes:
                        continue
                    text, _, _ = read_text_file(file)
                except (ToolError, OSError):
                    continue
                scanned += 1
                for number, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{sandbox.relative(file)}:{number}: {line.strip()[:200]}")
                        if len(hits) >= limit:
                            return hits, scanned
            return hits, scanned

        hits, scanned = await asyncio.to_thread(search)
        if not hits:
            return ToolResult.success(f"No matches for /{args['pattern']}/ ({scanned} files searched; sensitive files excluded)")
        suffix = f"\n(limit of {limit} matches reached)" if len(hits) >= limit else ""
        return ToolResult.success("\n".join(hits) + suffix, matches=len(hits))


class FindFilesTool(Tool):
    name = "find_files"
    group = "filesystem"
    description = "Find files by glob pattern on the relative path or file name, e.g. '*.py', 'src/**/test_*.py'."
    parameters = object_schema(
        {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Folder to search, default '.'"},
        },
        ["pattern"],
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        base = ctx.sandbox.resolve(args.get("path") or ".")
        pattern = args["pattern"].replace("\\", "/").lower()
        fs = ctx.tools_config.filesystem

        def find() -> list[str]:
            found = []
            for file in _walk_files(base, fs.ignore_patterns):
                rel = ctx.sandbox.relative(file)
                if fnmatch.fnmatch(rel.lower(), pattern) or fnmatch.fnmatch(file.name.lower(), pattern):
                    found.append(rel)
                    if len(found) >= fs.max_list_entries:
                        break
            return sorted(found)

        found = await asyncio.to_thread(find)
        if not found:
            return ToolResult.success(f"No files found for '{args['pattern']}'")
        return ToolResult.success("\n".join(found), count=len(found))


# ============================================================================ mutating


def _sensitive_reason(ctx: ToolContext, path: Path) -> list[str]:
    return ["sensitive file"] if ctx.sandbox.is_sensitive(path) else []


class WriteFileTool(Tool):
    name = "write_file"
    group = "filesystem"
    description = "Create a new file or completely overwrite an existing one. Prefer edit_file for small changes."
    parameters = object_schema(
        {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Full file content"},
        },
        ["path", "content"],
    )
    required = frozenset({Capability.WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.sandbox.resolve(args["path"])
        ctx.sandbox.check_writable(path)
        rel = ctx.sandbox.relative(path)
        content: str = args["content"]
        if len(content.encode("utf-8")) > ctx.tools_config.filesystem.max_file_bytes:
            return ToolResult.failure("Content exceeds the maximum file size")
        exists = path.exists()
        if exists and path.is_dir():
            return ToolResult.failure(f"{rel} is a directory")
        reasons = _sensitive_reason(ctx, path)
        await ctx.approval.require(
            ProposedAction(
                kind=ActionKind.OVERWRITE if exists else ActionKind.CREATE,
                agent=ctx.agent_name,
                target=rel,
                detail=f"{len(content)} characters",
                risk=RiskLevel.HIGH if reasons else (RiskLevel.MEDIUM if exists else RiskLevel.LOW),
                reasons=reasons,
            )
        )
        encoding, newline = "utf-8", "\n"
        if exists:
            _text, encoding, newline = await asyncio.to_thread(read_text_file, path)
        ctx.changes.before_write(path, ctx.agent_name)
        await asyncio.to_thread(write_text_atomic, path, content, encoding=encoding, newline=newline)
        verb = "overwritten" if exists else "created"
        return ToolResult.success(f"File {verb}: {rel} ({content.count(chr(10)) + 1} lines)", path=rel, created=not exists)


class EditFileTool(Tool):
    name = "edit_file"
    group = "filesystem"
    description = (
        "Replace an exact text snippet in a file. old_text must match exactly once (include enough context) "
        "unless replace_all is true. Do not include the line-number prefixes shown by read_file."
    )
    parameters = object_schema(
        {
            "path": {"type": "string"},
            "old_text": {"type": "string", "description": "Exact existing text"},
            "new_text": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean"},
        },
        ["path", "old_text", "new_text"],
    )
    required = frozenset({Capability.WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.sandbox.resolve(args["path"])
        ctx.sandbox.check_writable(path)
        rel = ctx.sandbox.relative(path)
        if not path.is_file():
            return ToolResult.failure(f"File does not exist: {rel} (use write_file to create it)")
        text, encoding, newline = await asyncio.to_thread(read_text_file, path)
        normalized = text.replace("\r\n", "\n")
        old = args["old_text"].replace("\r\n", "\n")
        new = args["new_text"].replace("\r\n", "\n")
        if not old:
            return ToolResult.failure("old_text must not be empty")
        count = normalized.count(old)
        if count == 0:
            return ToolResult.failure(f"old_text was found in {rel} not found. Check the file with read_file (copy whitespace and indentation exactly).")
        if count > 1 and not args.get("replace_all"):
            return ToolResult.failure(f"old_text occurs {count} times in {rel} - give more context or set replace_all=true.")
        await ctx.approval.require(
            ProposedAction(
                kind=ActionKind.EDIT,
                agent=ctx.agent_name,
                target=rel,
                detail=f"{count} replacement(s)",
                risk=RiskLevel.HIGH if ctx.sandbox.is_sensitive(path) else RiskLevel.LOW,
                reasons=_sensitive_reason(ctx, path),
            )
        )
        updated = normalized.replace(old, new) if args.get("replace_all") else normalized.replace(old, new, 1)
        ctx.changes.before_write(path, ctx.agent_name)
        await asyncio.to_thread(write_text_atomic, path, updated, encoding=encoding, newline=newline)
        diff = "".join(
            list(
                difflib.unified_diff(
                    normalized.splitlines(keepends=True), updated.splitlines(keepends=True), fromfile=rel, tofile=rel, n=2
                )
            )[:60]
        )
        return ToolResult.success(f"{rel} changed ({count} replacement(s)).\n{diff}", path=rel)


class CreateDirectoryTool(Tool):
    name = "create_directory"
    group = "filesystem"
    description = "Create a folder (including parents)."
    parameters = object_schema({"path": {"type": "string"}}, ["path"])
    required = frozenset({Capability.WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.sandbox.resolve(args["path"])
        ctx.sandbox.check_writable(path)
        rel = ctx.sandbox.relative(path)
        if path.exists():
            return ToolResult.success(f"Directory already exists: {rel}") if path.is_dir() else ToolResult.failure(f"{rel} is a file")
        await ctx.approval.require(ProposedAction(kind=ActionKind.MKDIR, agent=ctx.agent_name, target=rel, risk=RiskLevel.LOW))
        ctx.changes.mark_directory_created(path, ctx.agent_name)
        path.mkdir(parents=True, exist_ok=True)
        return ToolResult.success(f"Directory created: {rel}")


class MovePathTool(Tool):
    name = "move_path"
    group = "filesystem"
    description = "Move or rename a file or folder inside the workspace. The destination must not exist."
    parameters = object_schema({"source": {"type": "string"}, "destination": {"type": "string"}}, ["source", "destination"])
    required = frozenset({Capability.WRITE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        source = ctx.sandbox.resolve(args["source"])
        destination = ctx.sandbox.resolve(args["destination"])
        ctx.sandbox.check_writable(source)
        ctx.sandbox.check_writable(destination)
        src_rel, dst_rel = ctx.sandbox.relative(source), ctx.sandbox.relative(destination)
        if not source.exists():
            return ToolResult.failure(f"Source does not exist: {src_rel}")
        if destination.exists():
            return ToolResult.failure(f"Target already exists: {dst_rel}")
        await ctx.approval.require(
            ProposedAction(kind=ActionKind.MOVE, agent=ctx.agent_name, target=f"{src_rel} -> {dst_rel}", risk=RiskLevel.MEDIUM)
        )
        ctx.changes.before_delete(source, ctx.agent_name)
        ctx.changes.before_write(destination, ctx.agent_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.move, str(source), str(destination))
        return ToolResult.success(f"Moved: {src_rel} -> {dst_rel}")


class DeletePathTool(Tool):
    name = "delete_path"
    group = "filesystem"
    description = "Delete a file, or a folder with recursive=true. A backup is kept for rollback."
    parameters = object_schema({"path": {"type": "string"}, "recursive": {"type": "boolean"}}, ["path"])
    required = frozenset({Capability.DELETE})
    mutating = True
    plan_allowed = False
    parallel_safe = False

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.sandbox.resolve(args["path"])
        ctx.sandbox.check_writable(path)
        rel = ctx.sandbox.relative(path)
        if not path.exists():
            return ToolResult.failure(f"Path does not exist: {rel}")
        if path.is_dir() and any(path.iterdir()) and not args.get("recursive"):
            return ToolResult.failure(f"Directory {rel} is not empty - recursive=true required")
        await ctx.approval.require(
            ProposedAction(
                kind=ActionKind.DELETE,
                agent=ctx.agent_name,
                target=str(path),
                detail="directory (recursively)" if path.is_dir() else "File",
                risk=RiskLevel.HIGH,
                reasons=["Deletion"],
            )
        )
        try:
            ctx.changes.before_delete(path, ctx.agent_name)
        except OSError as exc:
            return ToolResult.failure(f"Backup not possible, deletion aborted: {exc}")
        if path.is_dir():
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            path.unlink()
        return ToolResult.success(f"Deleted (backup kept): {rel}")


TOOLS = [
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    FindFilesTool,
    WriteFileTool,
    EditFileTool,
    CreateDirectoryTool,
    MovePathTool,
    DeletePathTool,
]
