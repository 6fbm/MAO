"""Cheap, LLM-free workspace reconnaissance used as shared starting context."""

from __future__ import annotations

import fnmatch
import json
import os
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin", ".cs": "C#", ".go": "Go", ".rs": "Rust",
    ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++", ".hpp": "C++", ".rb": "Ruby", ".php": "PHP",
    ".swift": "Swift", ".dart": "Dart", ".lua": "Lua", ".scala": "Scala", ".vue": "Vue", ".svelte": "Svelte",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".sql": "SQL", ".sh": "Shell", ".ps1": "PowerShell",
    ".bat": "Batch", ".cmd": "Batch", ".md": "Markdown", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML", ".xml": "XML",
}
KEY_FILE_PATTERNS = [
    "README*", "pyproject.toml", "setup.py", "setup.cfg", "requirements*.txt", "Pipfile", "package.json",
    "tsconfig.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle*", "*.sln", "*.csproj", "Makefile",
    "CMakeLists.txt", "Dockerfile", "docker-compose*.yml", "pytest.ini", "tox.ini", ".gitignore",
]


class ProjectProfile(BaseModel):
    root: str
    file_count: int = 0
    dir_count: int = 0
    total_bytes: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    key_files: list[str] = Field(default_factory=list)
    tree: str = ""
    test_command: str | None = None
    truncated: bool = False
    git_branch: str | None = None
    git_status: str | None = None

    def render(self) -> str:
        languages = ", ".join(f"{name} ({count})" for name, count in self.languages.items()) or "–"
        lines = [
            f"Workspace: {self.root}",
            f"Files: {self.file_count}{'+' if self.truncated else ''}, directories: {self.dir_count}, size: {self.total_bytes / 1024:.0f} KB",
            f"Languages: {languages}",
            f"Key files: {', '.join(self.key_files) or '-'}",
            f"Test command (detected): {self.test_command or 'none'}",
        ]
        if self.git_branch:
            lines.append(f"Git branch: {self.git_branch}")
        if self.git_status:
            lines.append("Git-Status:\n" + self.git_status.strip())
        lines.append("Structure:\n" + self.tree)
        return "\n".join(lines)


def is_ignored(name: str, patterns: list[str]) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatchcase(lower, pattern.lower()) for pattern in patterns)


def detect_test_command(root: Path) -> str | None:
    venv_python = None
    for candidate in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if candidate.exists():
            venv_python = str(candidate.relative_to(root))
            break
    python = venv_python or "python"
    pyproject = root / "pyproject.toml"
    has_pytest_config = (root / "pytest.ini").exists() or (
        pyproject.exists() and "pytest" in pyproject.read_text(encoding="utf-8", errors="replace")
    )
    tests_dir = root / "tests"
    has_python_tests = tests_dir.is_dir() and any(tests_dir.rglob("test_*.py"))
    if has_pytest_config or has_python_tests or any(root.glob("test_*.py")):
        return f"{python} -m pytest -q"
    package_json = root / "package.json"
    if package_json.exists():
        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts") or {}
        except (json.JSONDecodeError, OSError):
            scripts = {}
        test_script = str(scripts.get("test") or "")
        if test_script and "no test specified" not in test_script:
            return "npm test"
    if (root / "Cargo.toml").exists():
        return "cargo test"
    if (root / "go.mod").exists():
        return "go test ./..."
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
        return "dotnet test"
    return None


def scan_workspace(root: Path, ignore_patterns: list[str], *, max_files: int = 20_000, tree_entries: int = 120) -> ProjectProfile:
    profile = ProjectProfile(root=str(root))
    extensions: Counter[str] = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_ignored(d, ignore_patterns) and not os.path.islink(os.path.join(dirpath, d))]
        profile.dir_count += len(dirnames)
        for filename in filenames:
            if is_ignored(filename, ignore_patterns):
                continue
            profile.file_count += 1
            try:
                profile.total_bytes += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
            extensions[Path(filename).suffix.lower()] += 1
            if profile.file_count >= max_files:
                profile.truncated = True
                break
        if profile.truncated:
            break
    languages: Counter[str] = Counter()
    for ext, count in extensions.items():
        if ext in LANGUAGES:
            languages[LANGUAGES[ext]] += count
    profile.languages = dict(languages.most_common(10))
    profile.key_files = sorted(
        {entry.name for pattern in KEY_FILE_PATTERNS for entry in root.glob(pattern) if entry.is_file()}
    )
    profile.test_command = detect_test_command(root)
    profile.tree = render_tree(root, ignore_patterns, depth=2, max_entries=tree_entries)
    return profile


def render_tree(root: Path, ignore_patterns: list[str], *, depth: int = 2, max_entries: int = 200) -> str:
    lines: list[str] = []
    count = 0

    def walk(directory: Path, prefix: str, level: int) -> bool:
        nonlocal count
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return True
        for entry in entries:
            if is_ignored(entry.name, ignore_patterns):
                continue
            count += 1
            if count > max_entries:
                lines.append(f"{prefix}… (further entries omitted)")
                return False
            is_link = entry.is_symlink() or getattr(entry, "is_junction", lambda: False)()
            if entry.is_dir() and not is_link:
                lines.append(f"{prefix}{entry.name}/")
                if level < depth and not walk(entry, prefix + "  ", level + 1):
                    return False
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{prefix}{entry.name} ({size} B)")
        return True

    walk(root, "", 1)
    return "\n".join(lines) or "(empty)"
