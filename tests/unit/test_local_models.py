"""The models/ folder: scanning, name derivation and the guard rails around imports."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mao.core.errors import ConfigError, MaoError
from mao.models import local


def test_ensure_dir_creates_folder_with_a_note(tmp_path: Path) -> None:
    models = local.ensure_dir(tmp_path / "models")
    assert models.is_dir()
    readme = (models / "README.md").read_text(encoding="utf-8")
    assert "models import" in readme
    # running it again keeps an edited note
    (models / "README.md").write_text("mine", encoding="utf-8")
    local.ensure_dir(models)
    assert (models / "README.md").read_text(encoding="utf-8") == "mine"


def test_scan_finds_weight_files_only(tmp_path: Path) -> None:
    models = local.ensure_dir(tmp_path / "models")
    (models / "a.gguf").write_bytes(b"x" * 10)
    (models / "b.GGUF").write_bytes(b"x" * 20)
    (models / "notes.txt").write_text("ignore me", encoding="utf-8")
    (models / "sub").mkdir()
    found = local.scan(models)
    assert [f.name for f in found] == ["a.gguf", "b.GGUF"]
    assert found[0].size_bytes == 10


def test_scan_tolerates_a_missing_folder(tmp_path: Path) -> None:
    assert local.scan(tmp_path / "nope") == []


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("Qwen3-4B-Instruct-Q4_K_M.gguf", "qwen3-4b-instruct-q4_k_m"),
        ("Meta Llama 3 8B.gguf", "meta-llama-3-8b"),
        ("---.gguf", "local-model"),
    ],
)
def test_model_name_is_derived_from_the_file_name(tmp_path: Path, file_name: str, expected: str) -> None:
    path = tmp_path / file_name
    path.write_bytes(b"x")
    assert local.LocalModelFile(path=path, size_bytes=1).suggested_model_name == expected


@pytest.mark.parametrize(
    ("size", "label"),
    [(3_500_000_000, "3.5 GB"), (1_000_000_000, "1.0 GB"), (250_000_000, "250 MB")],
)
def test_size_label(tmp_path: Path, size: int, label: str) -> None:
    assert local.LocalModelFile(path=tmp_path / "m.gguf", size_bytes=size).size_label == label


def test_resolve_file_refuses_paths_outside_the_folder(tmp_path: Path) -> None:
    models = local.ensure_dir(tmp_path / "models")
    (tmp_path / "secret.gguf").write_bytes(b"x")
    with pytest.raises(MaoError, match="not in"):
        local.resolve_file(models, "../secret.gguf")


def test_resolve_file_rejects_other_suffixes_and_missing_files(tmp_path: Path) -> None:
    models = local.ensure_dir(tmp_path / "models")
    (models / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(MaoError, match="not a .gguf"):
        local.resolve_file(models, "notes.txt")
    with pytest.raises(MaoError, match="No file"):
        local.resolve_file(models, "ghost.gguf")


def test_resolve_file_returns_the_entry(tmp_path: Path) -> None:
    models = local.ensure_dir(tmp_path / "models")
    (models / "m.gguf").write_bytes(b"x" * 7)
    entry = local.resolve_file(models, "m.gguf")
    assert entry.name == "m.gguf" and entry.size_bytes == 7


async def test_import_explains_a_missing_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = local.ensure_dir(tmp_path / "models")
    (models / "m.gguf").write_bytes(b"x")
    entry = local.resolve_file(models, "m.gguf")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ConfigError, match="ollama was not found"):
        await local.import_into_ollama(entry, "m")
