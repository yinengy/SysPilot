from __future__ import annotations

from toolbox.shared.probe_payload import emit_result, first_non_none, load_payload, primary_collect_profile, primary_runtime_map


def collect(payload: dict[str, object]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    runtime_map = primary_runtime_map(payload)
    collect_profile = primary_collect_profile(payload)
    cpu_collector = collect_profile.get("cpu", {}) if isinstance(collect_profile, dict) else {}
    cpu_data = cpu_collector.get("data", {}) if isinstance(cpu_collector, dict) else {}
    value = first_non_none(
        runtime_output.get("threads_counted") if isinstance(runtime_output, dict) else None,
        cpu_data.get("num_threads"),
        runtime_map.get("Threads"),
    )
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return {
        "observed": value is not None,
        "source": "runtime_output.threads_counted"
        if isinstance(runtime_output, dict) and runtime_output.get("threads_counted") is not None
        else ("collect_profile.cpu.num_threads" if cpu_data.get("num_threads") is not None else ("proc_status.Threads" if runtime_map.get("Threads") is not None else "os_missing")),
        "value": value,
        "flat": {"process.thread_count": value} if value is not None else {},
        "structured": {"thread_count": value} if value is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
