"""The models/ folder: local weight files and how they reach a runtime.

A .gguf file is only a blob of weights - something has to serve it. mao hands it
to Ollama, which is the runtime most people already have: `ollama create` copies
the file into Ollama's own store and from then on it is an ordinary local model
that the ollama provider discovers.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mao.core.errors import ConfigError, MaoError

WEIGHT_SUFFIXES = (".gguf",)
_NAME_CLEAN = re.compile(r"[^a-z0-9._-]+")
# Ollama draws a progress display; its escape codes must not reach our output.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
DEFAULT_IMPORT_TIMEOUT_S = 1_800.0

README = """\
Put local model files here.

Drop a .gguf file into this folder, then run inside mao:

    models local              lists the files found here
    models import <file>      hands the file to Ollama

Ollama copies the weights into its own store during the import, so the file is
held twice until you delete it here. After the import the model shows up like
any other local model - check it with `models discover ollama`.

This folder is not versioned: model files are far too large for a repository.
"""


@dataclass(frozen=True)
class LocalModelFile:
    path: Path
    size_bytes: int

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1_000_000_000

    @property
    def size_label(self) -> str:
        if self.size_bytes >= 1_000_000_000:
            return f"{self.size_gb:.1f} GB"
        return f"{self.size_bytes / 1_000_000:.0f} MB"

    @property
    def suggested_model_name(self) -> str:
        """An Ollama-safe name derived from the file name."""
        stem = self.path.stem.lower()
        cleaned = _NAME_CLEAN.sub("-", stem).strip("-._")
        return cleaned or "local-model"


def ensure_dir(models_dir: Path) -> Path:
    """Create the folder on first use and leave a note explaining what it is for."""
    models_dir.mkdir(parents=True, exist_ok=True)
    readme = models_dir / "README.md"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    return models_dir


def scan(models_dir: Path) -> list[LocalModelFile]:
    if not models_dir.is_dir():
        return []
    found = [
        LocalModelFile(path=p, size_bytes=p.stat().st_size)
        for p in sorted(models_dir.iterdir())
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
    ]
    return found


def resolve_file(models_dir: Path, name: str) -> LocalModelFile:
    """Look a file up by name, refusing anything outside the folder."""
    candidate = (models_dir / name).resolve()
    root = models_dir.resolve()
    if root != candidate.parent:
        raise MaoError(f"{name} is not in {models_dir} - only files directly in that folder can be imported.")
    if not candidate.is_file():
        available = ", ".join(f.name for f in scan(models_dir)) or "nothing yet"
        raise MaoError(f"No file {name} in {models_dir}. Found: {available}")
    if candidate.suffix.lower() not in WEIGHT_SUFFIXES:
        raise MaoError(f"{name} is not a {' or '.join(WEIGHT_SUFFIXES)} file.")
    return LocalModelFile(path=candidate, size_bytes=candidate.stat().st_size)


async def import_into_ollama(
    model_file: LocalModelFile,
    model_name: str,
    *,
    timeout_s: float = DEFAULT_IMPORT_TIMEOUT_S,
) -> str:
    """Register the file with Ollama under `model_name`. Returns the runtime output."""
    binary = shutil.which("ollama")
    if binary is None:
        raise ConfigError(
            "ollama was not found. Install it from https://ollama.com and start it, "
            "then run the import again."
        )
    with tempfile.TemporaryDirectory(prefix="mao-import-") as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text(f"FROM {model_file.path}\n", encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            binary,
            "create",
            model_name,
            "-f",
            str(modelfile),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise MaoError(f"The import took longer than {timeout_s:.0f}s and was stopped.") from None
    output = _ANSI.sub("", (stdout or b"").decode("utf-8", errors="replace")).strip()
    output = "\n".join(line.strip() for line in output.splitlines() if line.strip())
    if process.returncode != 0:
        raise MaoError(f"ollama create failed (exit {process.returncode}): {output or 'no output'}")
    return output
