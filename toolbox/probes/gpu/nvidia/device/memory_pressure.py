from __future__ import annotations

from toolbox.probes.gpu.nvidia._nvidia_smi import MIB, query_gpu, unavailable_result
from toolbox.shared.probe_payload import emit_result, load_payload

VIRTUAL_PROBE = "gpu.nvidia.device.memory_pressure"


def collect(payload: dict[str, object]) -> dict[str, object]:
    del payload
    fields = ["uuid", "index", "name", "memory.used", "memory.total", "memory.free"]
    output, rows = query_gpu(fields)
    if not output.observed:
        return unavailable_result(VIRTUAL_PROBE, output)

    devices: list[dict[str, object]] = []
    pressure_values: list[float] = []
    total_used_bytes = 0
    total_capacity_bytes = 0
    for row in rows:
        try:
            used_mib = float(row["memory.used"])
            total_mib = float(row["memory.total"])
        except (KeyError, ValueError):
            continue
        if total_mib <= 0:
            continue
        free_mib = float(row.get("memory.free") or max(total_mib - used_mib, 0.0))
        pressure_pct = (used_mib / total_mib) * 100.0
        pressure_values.append(pressure_pct)
        used_bytes = int(used_mib * MIB)
        total_bytes = int(total_mib * MIB)
        total_used_bytes += used_bytes
        total_capacity_bytes += total_bytes
        devices.append(
            {
                "device_uuid": row.get("uuid", ""),
                "visible_device": row.get("index", ""),
                "name": row.get("name", ""),
                "memory_used_bytes": used_bytes,
                "memory_total_bytes": total_bytes,
                "memory_free_bytes": int(free_mib * MIB),
                "memory_pressure_pct": pressure_pct,
                "attribution_quality": "device_exact",
            }
        )

    if not pressure_values:
        return {
            "observed": False,
            "source": "nvidia_smi_empty",
            "value": None,
            "flat": {},
            "structured": {"schema_version": "gpu_probe/v1", "devices": []},
        }

    max_pressure = max(pressure_values)
    aggregate_pressure = (
        (total_used_bytes / total_capacity_bytes) * 100.0
        if total_capacity_bytes
        else max_pressure
    )
    return {
        "observed": True,
        "source": output.source,
        "unit": "percent",
        "value": max_pressure,
        "flat": {
            "gpu.nvidia.device.memory_pressure_pct": max_pressure,
            "gpu.nvidia.device.memory_pressure_aggregate_pct": aggregate_pressure,
            "gpu.nvidia.device.memory_used_bytes": total_used_bytes,
            "gpu.nvidia.device.memory_total_bytes": total_capacity_bytes,
        },
        "structured": {
            "schema_version": "gpu_probe/v1",
            "metric": "memory_pressure_pct",
            "unit": "percent",
            "scope": "device",
            "aggregation": "max_visible_device",
            "attribution_quality": "device_exact",
            "devices": devices,
        },
    }


def sample(payload: dict[str, object], state: object | None = None) -> dict[str, object]:
    del state
    return collect(payload)


if __name__ == "__main__":
    emit_result(collect(load_payload()))
