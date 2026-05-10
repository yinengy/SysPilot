from __future__ import annotations

from pathlib import Path

from toolbox.shared.probe_payload import emit_result, load_payload, path_size_bytes


def collect(payload: dict[str, object]) -> dict[str, object]:
    candidate_paths: list[Path] = []
    workdir = payload.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        candidate_paths.append(Path(workdir))

    for env_name in ("RUN_DIR", "STATE_DIR", "BENCHMARK_STATE_DIR", "BENCHMARK_RUN_DIR"):
        env_map = payload.get("launch_env", {})
        if isinstance(env_map, dict):
            value = env_map.get(env_name)
            if isinstance(value, str) and value.strip():
                candidate_paths.append(Path(value))

    for candidate in candidate_paths:
        size_bytes = path_size_bytes(candidate)
        if size_bytes is not None:
            return {
                "observed": True,
                "source": f"filesystem:{candidate}",
                "value": size_bytes,
                "flat": {"filesystem.directory_size": size_bytes},
                "structured": {"path": str(candidate), "size_bytes": size_bytes},
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
