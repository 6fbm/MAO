"""Risk classification for agent actions and shell commands.

This is a defence-in-depth heuristic, not a complete shell parser. Anything
classified MEDIUM or higher is routed through the approval gateway (depending
on the configured rules); denylisted commands are always blocked.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path

from pydantic import BaseModel, Field


class RiskLevel(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return {0: "safe", 1: "low", 2: "medium", 3: "high", 4: "CRITICAL"}[int(self)]


class ActionKind(str, Enum):
    READ = "read"
    SENSITIVE_READ = "sensitive_read"
    CREATE = "create"
    OVERWRITE = "overwrite"
    EDIT = "edit"
    MKDIR = "mkdir"
    MOVE = "move"
    DELETE = "delete"
    EXECUTE = "execute"
    INTERNET = "internet"
    GIT_WRITE = "git_write"
    GIT_DESTRUCTIVE = "git_destructive"
    OUTSIDE_WORKSPACE = "outside_workspace"

    @property
    def label(self) -> str:
        return self.value.upper().replace("_", " ")


class ProposedAction(BaseModel):
    kind: ActionKind
    agent: str
    target: str
    detail: str = ""
    risk: RiskLevel = RiskLevel.LOW
    reasons: list[str] = Field(default_factory=list)


@dataclass
class CommandAssessment:
    risk: RiskLevel
    reasons: list[str] = field(default_factory=list)
    denied: bool = False
    allowlisted: bool = False
    chained: bool = False


_CHAIN_RE = re.compile(r"&&|\|\||[;&|]|\r|\n")
_REDIRECT_RE = re.compile(r"(^|[^<>\d])>{1,2}\s*\S|\d>\s*[^&\s]")

_RULES: list[tuple[RiskLevel, re.Pattern[str], str]] = [
    # --- critical
    (RiskLevel.CRITICAL, re.compile(r"^\s*(format|diskpart|bcdedit|vssadmin|mkfs\S*|fdisk)\b", re.I), "System or storage-device command"),
    (RiskLevel.CRITICAL, re.compile(r"^\s*(shutdown|restart-computer|stop-computer|reboot|halt|poweroff)\b", re.I), "Shutdown or restart"),
    (RiskLevel.CRITICAL, re.compile(r"\breg(\.exe)?\s+delete\b", re.I), "Delete registry entries"),
    (RiskLevel.CRITICAL, re.compile(r"\bcipher(\.exe)?\s+/w", re.I), "Overwrite a storage device"),
    (RiskLevel.CRITICAL, re.compile(r"\brm\s+-[a-z]*r[a-z]*\s+(-[a-z]+\s+)*(/|~|\$home|/\*)(\s|$)", re.I), "Recursive deletion of the root or home directory"),
    (RiskLevel.CRITICAL, re.compile(r"\b(rd|rmdir|del|erase|remove-item|ri)\b.*\s['\"]?[a-z]:\\?['\"]?(\s|$)", re.I), "Deletion at drive level"),
    (RiskLevel.CRITICAL, re.compile(r"(invoke-webrequest|iwr|curl|wget|irm|invoke-restmethod)\b[^|]*\|\s*(iex|invoke-expression|sh|bash|powershell|pwsh|python)\b", re.I), "Download piped straight into execution"),
    (RiskLevel.CRITICAL, re.compile(r"\s-(e|ec|enc|encodedcommand)\s+[A-Za-z0-9+/=]{16,}", re.I), "Encoded PowerShell command"),
    (RiskLevel.CRITICAL, re.compile(r"\bgit\s+push\b.*(\s--force(-with-lease)?\b|\s-f\b)", re.I), "Force-Push"),
    (RiskLevel.CRITICAL, re.compile(r"\bdd\s+if=", re.I), "Raw write access to a device"),
    # --- high
    (RiskLevel.HIGH, re.compile(r"^\s*(del|erase|rd|rmdir|rm|remove-item|ri|unlink|shred)\b", re.I), "Delete command"),
    (RiskLevel.HIGH, re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|checkout\s+(--\s|\.\s*$)|restore\b|stash\s+(drop|clear)|branch\s+-D|rebase\b|filter-branch|reflog\s+expire|gc\s+--prune)", re.I), "Destructive git command"),
    (RiskLevel.HIGH, re.compile(r"\bgit\s+push\b", re.I), "Push to a remote repository"),
    (RiskLevel.HIGH, re.compile(r"\b(invoke-webrequest|iwr|irm|invoke-restmethod|curl|wget|bitsadmin|start-bitstransfer)\b|\bcertutil\b.*-urlcache", re.I), "Network access or download"),
    (RiskLevel.HIGH, re.compile(r"\b(npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b|\bnuget\s+push\b", re.I), "Publishing a package"),
    (RiskLevel.HIGH, re.compile(r"\b(takeown|icacls|cacls|attrib|chmod|chown|set-acl)\b", re.I), "Change of file permissions or attributes"),
    (RiskLevel.HIGH, re.compile(r"\b(setx|reg(\.exe)?\s+add|set-itemproperty|new-itemproperty)\b", re.I), "Change of system or environment settings"),
    (RiskLevel.HIGH, re.compile(r"\b(schtasks|new-service|sc(\.exe)?\s+(create|delete|config|stop))\b", re.I), "Services or scheduled tasks"),
    (RiskLevel.HIGH, re.compile(r"\b(start-process|runas|stop-process|taskkill|kill|pkill)\b", re.I), "Process control"),
    (RiskLevel.HIGH, re.compile(r"\b(ssh|scp|sftp|ftp|telnet|nc|ncat|netcat)\b", re.I), "Network connection"),
    # --- medium
    (RiskLevel.MEDIUM, re.compile(r"\b(pip3?|uv\s+pip|poetry|npm|pnpm|yarn|choco|winget|scoop|cargo|go|dotnet|gem|composer)\s+(install|i|add|remove|uninstall|update|upgrade|get)\b", re.I), "Package install or change"),
    (RiskLevel.MEDIUM, re.compile(r"\bgit\s+(commit|merge|cherry-pick|am|apply|switch|checkout|tag|stash)\b", re.I), "Git change"),
    (RiskLevel.MEDIUM, re.compile(r"\b(python3?|py|node|deno|bun|ruby|perl|php)\s+(-c|-e|--eval)\b", re.I), "Inline code execution"),
    (RiskLevel.MEDIUM, re.compile(r"\b(move|mv|move-item|ren|rename|rename-item|copy|xcopy|robocopy|cp|copy-item|mklink|new-item|set-content|out-file|add-content|clear-content)\b", re.I), "Filesystem change"),
    (RiskLevel.MEDIUM, re.compile(r"%[A-Za-z_]+%|\$env:|\$HOME\b", re.I), "Use of environment variables or paths outside"),
]

_WIN_ABS_PATH_RE = re.compile(r"(?<![\w/\\])([A-Za-z]:[\\/][^\s\"'|&;<>]*)")
_POSIX_ABS_PATH_RE = re.compile(r"(?<![\w.:/-])(/(?:[\w.\-]+/?)+)")
_PARENT_TRAVERSAL_RE = re.compile(r"(^|[\s\"'=])\.\.[\\/]")


def _normalize(command: str) -> str:
    return " ".join(command.strip().split())


def matches_patterns(command: str, patterns: list[str]) -> bool:
    normalized = _normalize(command).lower()
    return any(fnmatch.fnmatchcase(normalized, p.lower()) for p in patterns)


def is_chained(command: str) -> bool:
    return bool(_CHAIN_RE.search(command) or _REDIRECT_RE.search(command))


def split_segments(command: str) -> list[str]:
    return [seg.strip() for seg in re.split(r"&&|\|\||[;&|\r\n]", command) if seg.strip()]


def split_program(segment: str) -> tuple[str, str]:
    """Split a command segment into (program, arguments), honouring a quoted program path."""
    text = segment.strip()
    if text.startswith('"'):
        end = text.find('"', 1)
        if end > 0:
            return text[1:end], text[end + 1 :].strip()
        return text.strip('"'), ""
    parts = text.split(None, 1)
    if not parts:
        return "", ""
    return parts[0], parts[1] if len(parts) > 1 else ""


def canonical_segment(segment: str) -> str:
    """'C:\\Windows\\System32\\format.com c:' -> 'format c:' so rules cannot be bypassed via paths."""
    program, arguments = split_program(segment)
    name = re.split(r"[\\/]", program)[-1]
    name = re.sub(r"\.(exe|com|bat|cmd|ps1)$", "", name, flags=re.I)
    return f"{name} {arguments}".strip()


def assess_command(
    command: str,
    *,
    workspace_root: Path | None = None,
    denylist: list[str] | None = None,
    allowlist: list[str] | None = None,
) -> CommandAssessment:
    normalized = _normalize(command)
    if not normalized:
        return CommandAssessment(risk=RiskLevel.SAFE, reasons=["empty command"])
    chained = is_chained(normalized)
    segments = split_segments(normalized)
    canonical = [canonical_segment(s) for s in segments]
    result = CommandAssessment(risk=RiskLevel.LOW, chained=chained)
    # the full command is checked too (some patterns span a pipe: download | shell), and every
    # segment additionally with the bare program name (an absolute path must not bypass rules)
    candidates = list(dict.fromkeys([normalized, *segments, *canonical]))

    denylist = denylist or []
    for segment in candidates:
        if matches_patterns(segment, denylist):
            result.denied = True
            result.risk = RiskLevel.CRITICAL
            result.reasons.append(f"Command is on the denylist: {segment}")
            break

    for segment in candidates:
        for level, pattern, reason in _RULES:
            if pattern.search(segment):
                if level > result.risk:
                    result.risk = level
                if reason not in result.reasons:
                    result.reasons.append(reason)

    if workspace_root is not None:
        root_norm = os.path.normcase(str(workspace_root))
        # the program may live anywhere (e.g. "C:\Python312\python.exe"); its arguments must not point outside
        arguments = " ".join(split_program(segment)[1] for segment in segments)
        found_paths = list(_WIN_ABS_PATH_RE.findall(arguments))
        if sys.platform != "win32":
            found_paths += _POSIX_ABS_PATH_RE.findall(arguments)
        for raw_path in found_paths:
            path_norm = os.path.normcase(os.path.normpath(raw_path))
            try:
                inside = os.path.commonpath([root_norm, path_norm]) == root_norm
            except ValueError:
                inside = False
            if not inside:
                result.risk = max(result.risk, RiskLevel.HIGH)
                result.reasons.append(f"Path outside the workspace: {raw_path}")
        if _PARENT_TRAVERSAL_RE.search(normalized):
            result.risk = max(result.risk, RiskLevel.MEDIUM)
            result.reasons.append("Relative path with '..' (possibly outside the workspace)")

    if chained and result.risk < RiskLevel.MEDIUM:
        result.risk = RiskLevel.MEDIUM
        result.reasons.append("Command chaining or redirection")

    if allowlist and not chained and not result.denied and (
        matches_patterns(normalized, allowlist) or any(matches_patterns(c, allowlist) for c in canonical)
    ):
        result.allowlisted = True
        if result.risk <= RiskLevel.MEDIUM:
            result.risk = RiskLevel.LOW

    return result
