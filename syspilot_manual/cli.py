from __future__ import annotations

from pathlib import Path


def normalize_manual_inputs(values: list[str]) -> list[str]:
    return sorted({value.strip() for value in values if value and value.strip()})


def resolve_manual_paths(values: list[str]) -> list[Path]:
    return sorted(Path(value).expanduser() for value in normalize_manual_inputs(values))
