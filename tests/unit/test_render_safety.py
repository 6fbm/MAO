"""Untrusted text (agent output, file content, git output) must never be parsed as rich markup."""

from __future__ import annotations

import io
from types import SimpleNamespace

from rich.console import Console
from rich.table import Table

from mao.cli.console import ConsoleUI
from mao.core.events import EventBus

UNTRUSTED = "[/broken] [ollama] model not found :smile: [bold]x"


def test_console_without_markup_renders_untrusted_text() -> None:
    console = Console(file=io.StringIO(), markup=False, emoji=False, width=160)
    table = Table()
    table.add_column("Details")
    table.add_row(UNTRUSTED)
    console.print(table)
    console.print(UNTRUSTED)
    output = console.file.getvalue()  # type: ignore[attr-defined]
    assert output.count(UNTRUSTED) == 2


def test_console_ui_disables_markup() -> None:
    ui = ConsoleUI(SimpleNamespace(bus=EventBus()), interactive=False)  # type: ignore[arg-type]
    buffer = io.StringIO()
    ui.console = Console(file=buffer, markup=ui.console._markup, emoji=ui.console._emoji, width=160)
    ui.print(UNTRUSTED)
    ui.error(UNTRUSTED)
    assert ui.console._markup is False
    assert buffer.getvalue().count("[/broken] [ollama]") == 2
