from __future__ import annotations

from toolbox.probes.gpu.nvidia._nvidia_smi import query_pmon, target_pids, unavailable_result
from toolbox.shared.probe_payload import emit_result, load_payload

VIRTUAL_PROBE = "gpu.nvidia.process.sm_utilization"


def collect(payload: dict[str, object]) -> dict[str, object]:
    output, rows = query_pmon()
    if not output.observed:
        return unavailable_result(VIRTUAL_PROBE, output)

    wanted_pids = target_pids(payload)
    processes: list[dict[str, object]] = []
    values: list[float] = []
    for row in rows:
        pid = row.get("pid")
        sm_util = row.get("sm_utilization_pct")
        if not isinstance(pid, int) or not isinstance(sm_util, (int, float)):
            continue
        is_target = not wanted_pids or pid in wanted_pids
        if not is_target:
            continue
        value = float(sm_util)
        values.append(value)
        processes.append(
            {
                **row,
                "sm_utilization_pct": value,
                "is_target_descendant": bool(wanted_pids and pid in wanted_pids),
                "attribution_quality": "process_best_effort",
            }
        )

    total_sm = sum(values)
    return {
        "observed": bool(processes),
        "source": output.source if processes else "nvidia_smi_pmon_no_target_process",
        "unit": "percent",
        "value": total_sm if processes else None,
        "flat": {"gpu.nvidia.process.sm_utilization_pct": total_sm} if processes else {},
        "structured": {
            "schema_version": "gpu_probe/v1",
            "metric": "sm_utilization_pct",
            "unit": "percent",
            "scope": "process",
            "aggregation": "sum_target_processes",
            "target_pids": sorted(wanted_pids),
            "attribution_quality": "process_best_effort" if processes else "process_absent",
            "processes": processes,
        },
    }


def sample(payload: dict[str, object], state: object | None = None) -> dict[str, object]:
    del state
    return collect(payload)


if __name__ == "__main__":
    emit_result(collect(load_payload()))
