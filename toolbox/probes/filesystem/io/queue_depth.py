from __future__ import annotations

from pathlib import Path
from typing import Any

from toolbox.shared.probe_payload import emit_result, load_payload


def _runtime_output(payload: dict[str, Any]) -> dict[str, Any]:
    runtime_output = payload.get("runtime_output", {})
    return runtime_output if isinstance(runtime_output, dict) else {}


def _count_entries(path: Path) -> int | None:
    try:
        if not path.exists():
            return None
        count = 0
        for child in path.rglob("*"):
            if child.is_file():
                count += 1
        return count
    except OSError:
        return None


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = _runtime_output(payload)
    for key in ("duckdb_temp_directory", "dask_local_directory"):
        raw_path = runtime_output.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        count = _count_entries(Path(raw_path))
        if count is not None:
            return {
                "observed": True,
                "source": f"runtime_output.{key}",
                "value": float(count),
                "flat": {"filesystem.io.queue_depth": float(count)},
                "structured": {
                    "path": raw_path,
                    "entry_count": count,
                    "approximation": "filesystem_entry_count_proxy",
                },
            }
    return {
        "observed": False,
        "source": "os_missing",
        "value": None,
        "flat": {},
        "structured": {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
