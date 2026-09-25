from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def to_native(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_native(item) for item in value]

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        return to_native(scalar_item())
    return value


def normalize_config(config: Any) -> dict[str, Any]:
    return {str(key): to_native(value) for key, value in dict(config).items()}


def compact_json(payload: Any) -> str:
    return json.dumps(
        to_native(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def config_fingerprint(config: Any) -> str:
    canonical = compact_json(normalize_config(config)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, compact_json(payload) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    text = "".join(f"{compact_json(row)}\n" for row in rows)
    _atomic_write_text(path, text)
