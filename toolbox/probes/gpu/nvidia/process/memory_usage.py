from __future__ import annotations

from toolbox.probes.gpu.nvidia._nvidia_smi import (
    MIB,
    numeric_int,
    query_compute_apps,
    target_pids,
    unavailable_result,
)
from toolbox.shared.probe_payload import emit_result, load_payload

VIRTUAL_PROBE = "gpu.nvidia.process.memory_usage"


def collect(payload: dict[str, object]) -> dict[str, object]:
    fields = ["pid", "process_name", "gpu_uuid", "used_memory"]
    output, rows = query_compute_apps(fields)
    if not output.observed:
        return unavailable_result(VIRTUAL_PROBE, output)

    wanted_pids = target_pids(payload)
    processes: list[dict[str, object]] = []
    total_bytes = 0
    for row in rows:
        pid = numeric_int(row.get("pid"))
        used_mib = numeric_int(row.get("used_memory"))
        if pid is None or used_mib is None:
            continue
        is_target = not wanted_pids or pid in wanted_pids
        if not is_target:
            continue
        used_bytes = used_mib * MIB
        total_bytes += used_bytes
        processes.append(
            {
                "pid": pid,
                "process_name": row.get("process_name", ""),
                "device_uuid": row.get("gpu_uuid", ""),
                "memory_used_bytes": used_bytes,
                "is_target_descendant": bool(wanted_pids and pid in wanted_pids),
                "attribution_quality": "process_exact",
            }
        )

    return {
        "observed": bool(processes),
        "source": output.source if processes else "nvidia_smi_no_target_process",
        "unit": "bytes",
        "value": total_bytes if processes else None,
        "flat": {"gpu.nvidia.process.memory_used_bytes": total_bytes} if processes else {},
        "structured": {
            "schema_version": "gpu_probe/v1",
            "metric": "memory_used_bytes",
            "unit": "bytes",
            "scope": "process",
            "target_pids": sorted(wanted_pids),
            "attribution_quality": "process_exact" if processes else "process_absent",
            "processes": processes,
        },
    }


def sample(payload: dict[str, object], state: object | None = None) -> dict[str, object]:
    del state
    return collect(payload)


if __name__ == "__main__":
    emit_result(collect(load_payload()))
