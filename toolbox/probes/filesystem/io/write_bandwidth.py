from __future__ import annotations

from pathlib import Path
from typing import Any

from toolbox.probes.process.io.byte_rate import begin as process_begin
from toolbox.probes.process.io.byte_rate import finish as process_finish
from toolbox.probes.process.io.byte_rate import sample as process_sample
from toolbox.shared.probe_payload import emit_result, load_payload, path_size_bytes


def _runtime_output(payload: dict[str, Any]) -> dict[str, Any]:
    runtime_output = payload.get("runtime_output", {})
    return runtime_output if isinstance(runtime_output, dict) else {}


def _rewrite_process_rate(result: dict[str, object]) -> dict[str, object]:
    value = result.get("value")
    if not isinstance(value, (int, float)):
        return result
    structured = result.get("structured", {})
    structured_map = dict(structured) if isinstance(structured, dict) else {}
    structured_map["approximation"] = "process_write_rate_fallback"
    return {
        "observed": bool(result.get("observed")),
        "source": str(result.get("source") or "procfs:process.io.byte_rate"),
        "value": float(value),
        "flat": {"filesystem.io.write_bandwidth": float(value)},
        "structured": structured_map,
    }


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = _runtime_output(payload)
    wall_time = runtime_output.get("wall_time_sec")
    byte_fields = [
        runtime_output.get("dask_local_bytes"),
        runtime_output.get("duckdb_temp_bytes"),
        runtime_output.get("spill_bytes"),
        runtime_output.get("temp_bytes"),
    ]
    total_bytes = sum(float(value) for value in byte_fields if isinstance(value, (int, float)))
    if isinstance(wall_time, (int, float)) and wall_time > 0 and total_bytes > 0:
        bandwidth = total_bytes / float(wall_time)
        return {
            "observed": True,
            "source": "runtime_output.temp_bytes_per_wall_time",
            "value": bandwidth,
            "flat": {"filesystem.io.write_bandwidth": bandwidth},
            "structured": {
                "bandwidth_bytes_per_sec": bandwidth,
                "wall_time_sec": float(wall_time),
                "total_bytes": total_bytes,
            },
        }

    for key in ("duckdb_temp_directory", "dask_local_directory"):
        raw_path = runtime_output.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        size_bytes = path_size_bytes(Path(raw_path))
        if size_bytes is not None and isinstance(wall_time, (int, float)) and wall_time > 0:
            bandwidth = float(size_bytes) / float(wall_time)
            return {
                "observed": True,
                "source": f"runtime_output.{key}",
                "value": bandwidth,
                "flat": {"filesystem.io.write_bandwidth": bandwidth},
                "structured": {
                    "bandwidth_bytes_per_sec": bandwidth,
                    "wall_time_sec": float(wall_time),
                    "size_bytes": float(size_bytes),
                    "path": raw_path,
                },
            }

    from toolbox.probes.process.io.byte_rate import collect as process_collect

    return _rewrite_process_rate(process_collect(payload))


async def begin(payload: dict[str, Any]) -> dict[str, object]:
    return await process_begin(payload)


async def sample(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    return _rewrite_process_rate(await process_sample(payload, state))


async def finish(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    return _rewrite_process_rate(await process_finish(payload, state))


if __name__ == "__main__":
    emit_result(collect(load_payload()))
