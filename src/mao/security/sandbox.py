"""Confines all file-system access of agents to the workspace root."""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from pathlib import Path

from mao.core.errors import ConfigError, SandboxViolationError

_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _matches_any(rel_posix: str, patterns: list[str], *, check_segments: bool) -> bool:
    rel = rel_posix.lower()
    segments = rel.split("/")
    for raw_pattern in patterns:
        pattern = raw_pattern.lower().replace("\\", "/")
        if fnmatch.fnmatchcase(rel, pattern) or fnmatch.fnmatchcase(segments[-1], pattern):
            return True
        if check_segments and "/" not in pattern and any(fnmatch.fnmatchcase(seg, pattern) for seg in segments):
            return True
    return False


class WorkspaceSandbox:
    def __init__(
        self,
        root: Path,
        *,
        protected_patterns: list[str] | None = None,
        sensitive_patterns: list[str] | None = None,
    ) -> None:
        resolved = Path(os.path.realpath(root))
        if not resolved.is_dir():
            raise ConfigError(f"The workspace does not exist or is not a directory: {root}")
        self.root = resolved
        self._root_norm = os.path.normcase(str(resolved))
        self.protected_patterns = protected_patterns or []
        self.sensitive_patterns = sensitive_patterns or []

    # ------------------------------------------------------------------ validation

    def _check_raw(self, raw: str) -> str:
        if raw is None or not str(raw).strip():
            raise SandboxViolationError("Empty path")
        text = str(raw).strip().strip('"').strip("'")
        if "\x00" in text:
            raise SandboxViolationError("The path contains a null byte")
        if text.startswith(("\\\\", "//")):
            raise SandboxViolationError(f"UNC and device paths are not allowed: {text}")
        if text.startswith("~"):
            raise SandboxViolationError(f"Home directory paths are not allowed: {text}")
        if sys.platform == "win32":
            body = text[2:] if _DRIVE_RE.match(text) else text
            if ":" in body:
                raise SandboxViolationError(f"Invalid colon in the path (alternate data stream?): {text}")
            for part in re.split(r"[\\/]", text):
                stem = part.split(".")[0].strip().upper()
                if stem in _RESERVED_NAMES:
                    raise SandboxViolationError(f"Reserved Windows device name in the path: {part}")
        return text

    def contains(self, path: Path) -> bool:
        norm = os.path.normcase(str(path))
        try:
            return os.path.commonpath([self._root_norm, norm]) == self._root_norm
        except ValueError:  # different drives
            return False

    def resolve(self, raw: str) -> Path:
        """Resolve a user/agent supplied path (relative to the root) and enforce containment."""
        text = self._check_raw(raw)
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # realpath resolves symlinks and junctions of the existing part of the path
        resolved = Path(os.path.realpath(candidate))
        if not self.contains(resolved):
            raise SandboxViolationError(f"Path lies outside the workspace: {raw}")
        return resolved

    def relative(self, path: Path) -> str:
        rel = os.path.relpath(path, self.root)
        return "." if rel == "." else Path(rel).as_posix()

    def is_protected(self, path: Path) -> bool:
        rel = self.relative(path)
        return rel != "." and _matches_any(rel, self.protected_patterns, check_segments=True)

    def is_sensitive(self, path: Path) -> bool:
        rel = self.relative(path)
        return rel != "." and _matches_any(rel, self.sensitive_patterns, check_segments=False)

    def check_writable(self, path: Path) -> None:
        if path == self.root:
            raise SandboxViolationError("The workspace root directory itself must not be changed")
        if self.is_protected(path):
            raise SandboxViolationError(f"Protected path (not writable): {self.relative(path)}")
