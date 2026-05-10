from __future__ import annotations

from toolbox.shared.probe_payload import emit_result, load_payload, parse_cpu_list_count, primary_runtime_map


def collect(payload: dict[str, object]) -> dict[str, object]:
    runtime_map = primary_runtime_map(payload)
    cpus_allowed = runtime_map.get("Cpus_allowed_list")
    value = parse_cpu_list_count(cpus_allowed)
    flat = {"Cpus_allowed_list": cpus_allowed} if cpus_allowed is not None else {}
    return {
        "observed": value is not None,
        "source": "proc_status.Cpus_allowed_list" if value is not None else "os_missing",
        "value": value,
        "flat": flat,
        "structured": {"Cpus_allowed_list": cpus_allowed, "allowed_cpu_count": value} if cpus_allowed is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
