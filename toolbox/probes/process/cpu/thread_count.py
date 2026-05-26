from __future__ import annotations

from toolbox.shared.probe_payload import emit_result, first_non_none, json_rows_from_output, libcontentbench_metric_timeseries, load_payload, primary_collect_profile, primary_runtime_map


def _numeric_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _thread_count_from_json_rows(payload: dict[str, object]) -> int | None:
    values: list[int] = []
    for row in json_rows_from_output(payload):
        sampler_summary = row.get("sampler_summary")
        if isinstance(sampler_summary, dict):
            peak_threads = _numeric_int(sampler_summary.get("peak_threads"))
            if peak_threads is not None:
                values.append(peak_threads)
        secondary = _numeric_int(row.get("resource_signal_secondary"))
        if secondary is not None and "thread" in str(row.get("config_summary", row)).lower():
            values.append(secondary)
    return max(values) if values else None


def _thread_timeseries_from_libcontentbench(payload: dict[str, object]) -> list[dict[str, float]]:
    raw_rows = libcontentbench_metric_timeseries(payload)
    parsed: list[tuple[float, float]] = []
    for row in raw_rows:
        raw_ts = row.get("ts")
        raw_threads = row.get("threads")
        try:
            ts = float(raw_ts) if raw_ts is not None else None
            threads = float(raw_threads) if raw_threads is not None else None
        except (TypeError, ValueError):
            continue
        if ts is None or threads is None:
            continue
        parsed.append((ts, threads))
    if not parsed:
        return []
    first_ts = parsed[0][0]
    return [
        {"time_offset_s": max(ts - first_ts, 0.0), "value": threads}
        for ts, threads in parsed
    ]


def collect(payload: dict[str, object]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    runtime_map = primary_runtime_map(payload)
    collect_profile = primary_collect_profile(payload)
    cpu_collector = collect_profile.get("cpu", {}) if isinstance(collect_profile, dict) else {}
    cpu_data = cpu_collector.get("data", {}) if isinstance(cpu_collector, dict) else {}
    json_thread_count = _thread_count_from_json_rows(payload)
    timeseries = _thread_timeseries_from_libcontentbench(payload)
    timeseries_peak = int(max(point["value"] for point in timeseries)) if timeseries else None
    value = first_non_none(
        runtime_output.get("threads_counted") if isinstance(runtime_output, dict) else None,
        cpu_data.get("num_threads"),
        runtime_map.get("Threads"),
        timeseries_peak,
        json_thread_count,
    )
    value = _numeric_int(value)
    return {
        "observed": value is not None,
        "source": "runtime_output.threads_counted"
        if isinstance(runtime_output, dict) and runtime_output.get("threads_counted") is not None
        else (
            "collect_profile.cpu.num_threads"
            if cpu_data.get("num_threads") is not None
            else (
                "proc_status.Threads"
                if runtime_map.get("Threads") is not None
                else (
                    "runtime_output.json.metrics_timeseries.threads"
                    if timeseries_peak is not None
                    else ("runtime_output.json.sampler_summary.peak_threads" if json_thread_count is not None else "os_missing")
                )
            )
        ),
        "value": value,
        "flat": {"process.thread_count": value} if value is not None else {},
        "structured": {"thread_count": value, "timeseries": timeseries} if value is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
