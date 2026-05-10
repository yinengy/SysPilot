from __future__ import annotations

from toolbox.shared.probe_payload import first_non_none, load_payload, emit_result, primary_collect_profile, primary_runtime_map, parse_rss_kb


def collect(payload: dict[str, object]) -> dict[str, object]:
    runtime_map = primary_runtime_map(payload)
    collect_profile = primary_collect_profile(payload)
    memory_collector = collect_profile.get("memory", {}) if isinstance(collect_profile, dict) else {}
    memory_data = memory_collector.get("data", {}) if isinstance(memory_collector, dict) else {}

    rss_kb = parse_rss_kb(runtime_map.get("VmRSS"))
    value = first_non_none(rss_kb, memory_data.get("peak_memory_mb"), memory_data.get("rss_mb"))
    flat = {"process_rss": value} if value is not None else {}
    return {
        "observed": value is not None,
        "source": "proc_status" if rss_kb is not None else ("collect_profile.memory" if value is not None else "os_missing"),
        "value": value,
        "flat": flat,
        "structured": {"process_rss": value} if value is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
