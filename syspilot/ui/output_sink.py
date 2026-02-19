from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence
from typing import Protocol

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class TuneUIMode(str, Enum):
    PLAIN = "plain"
    TUI = "tui"


class OutputSink(Protocol):
    def print(self, message: str = "") -> None: ...

    def panel(self, body: str, *, title: str) -> None: ...

    def table(
        self,
        *,
        title: str,
        columns: Sequence[tuple[str, str | None]],
        rows: Sequence[Sequence[str]],
    ) -> None: ...


@dataclass(frozen=True)
class PlainOutputSink:
    console: Console

    def print(self, message: str = "") -> None:
        self.console.print(message)

    def panel(self, body: str, *, title: str) -> None:
        self.console.print(Panel(body, title=title))

    def table(
        self,
        *,
        title: str,
        columns: Sequence[tuple[str, str | None]],
        rows: Sequence[Sequence[str]],
    ) -> None:
        table = Table(title=title)
        for name, style in columns:
            if style is None:
                table.add_column(name)
            else:
                table.add_column(name, style=style)
        for row in rows:
            table.add_row(*[str(value) for value in row])
        self.console.print(table)


@dataclass(frozen=True)
class MutedOutputSink:
    def print(self, message: str = "") -> None:
        _ = message

    def panel(self, body: str, *, title: str) -> None:
        _ = body
        _ = title

    def table(
        self,
        *,
        title: str,
        columns: Sequence[tuple[str, str | None]],
        rows: Sequence[Sequence[str]],
    ) -> None:
        _ = title
        _ = columns
        _ = rows


class SwitchableOutputSink:
    def __init__(self, initial: OutputSink) -> None:
        self._sink: OutputSink = initial

    def switch(self, sink: OutputSink) -> None:
        self._sink = sink

    def print(self, message: str = "") -> None:
        self._sink.print(message)

    def panel(self, body: str, *, title: str) -> None:
        self._sink.panel(body, title=title)

    def table(
        self,
        *,
        title: str,
        columns: Sequence[tuple[str, str | None]],
        rows: Sequence[Sequence[str]],
    ) -> None:
        self._sink.table(title=title, columns=columns, rows=rows)


def make_output_sink(*, ui_mode: TuneUIMode, console: Console) -> OutputSink:
    if ui_mode == TuneUIMode.PLAIN:
        return PlainOutputSink(console=console)
    return MutedOutputSink()
