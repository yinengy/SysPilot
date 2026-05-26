from __future__ import annotations

from pathlib import Path
from typing import Any

from toolbox.shared.probe_payload import combined_output

_MAX_TAIL_BYTES = 256 * 1024
_RUN_DIR_ENV_VARS = ("RUN_DIR", "STATE_DIR", "BENCHMARK_STATE_DIR", "BENCHMARK_RUN_DIR")
_LOG_GLOBS = ("*.out", "*.log")


def candidate_log_paths(payload: dict[str, Any]) -> list[Path]:
    candidate_dirs: list[Path] = []
    workdir = payload.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        candidate_dirs.append(Path(workdir))

    env_map = payload.get("launch_env", {})
    if isinstance(env_map, dict):
        for env_name in _RUN_DIR_ENV_VARS:
            value = env_map.get(env_name)
            if isinstance(value, str) and value.strip():
                candidate_dirs.append(Path(value))

    paths: list[Path] = []
    seen: set[str] = set()
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            continue
        for pattern in _LOG_GLOBS:
            for path in sorted(candidate_dir.glob(pattern)):
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
    return paths


def read_text_tail(path: Path, *, max_bytes: int = _MAX_TAIL_BYTES) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def payload_text_sources(payload: dict[str, Any], *, max_bytes: int = _MAX_TAIL_BYTES) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    runtime_text = combined_output(payload).strip()
    if runtime_text:
        sources.append(("runtime_output", runtime_text))

    for path in candidate_log_paths(payload):
        text = read_text_tail(path, max_bytes=max_bytes).strip()
        if text:
            sources.append((f"file:{path}", text))

    return sources
