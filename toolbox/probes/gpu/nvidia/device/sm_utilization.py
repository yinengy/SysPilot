from __future__ import annotations

from toolbox.probes.gpu.nvidia._nvidia_smi import query_gpu, unavailable_result
from toolbox.shared.probe_payload import emit_result, load_payload

VIRTUAL_PROBE = "gpu.nvidia.device.sm_utilization"


def collect(payload: dict[str, object]) -> dict[str, object]:
    del payload
    fields = ["uuid", "index", "name", "utilization.gpu", "utilization.memory"]
    output, rows = query_gpu(fields)
    if not output.observed:
        return unavailable_result(VIRTUAL_PROBE, output)

    devices: list[dict[str, object]] = []
    values: list[float] = []
    for row in rows:
        try:
            sm_util = float(row["utilization.gpu"])
        except (KeyError, ValueError):
            continue
        values.append(sm_util)
        devices.append(
            {
                "device_uuid": row.get("uuid", ""),
                "visible_device": row.get("index", ""),
                "name": row.get("name", ""),
                "sm_utilization_pct": sm_util,
                "memory_utilization_pct": float(row.get("utilization.memory") or 0.0),
                "attribution_quality": "device_exact",
            }
        )

    if not values:
        return {
            "observed": False,
            "source": "nvidia_smi_empty",
            "value": None,
            "flat": {},
            "structured": {"schema_version": "gpu_probe/v1", "devices": []},
        }

    max_value = max(values)
    avg_value = sum(values) / len(values)
    return {
        "observed": True,
        "source": output.source,
        "unit": "percent",
        "value": max_value,
        "flat": {
            "gpu.nvidia.device.sm_utilization_pct": max_value,
            "gpu.nvidia.device.sm_utilization_avg_pct": avg_value,
        },
        "structured": {
            "schema_version": "gpu_probe/v1",
            "metric": "sm_utilization_pct",
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
