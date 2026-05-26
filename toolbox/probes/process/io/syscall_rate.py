from __future__ import annotations

from typing import Any

from toolbox.probes.process.io._common import build_rate_result, read_proc_io_totals, target_pids
from toolbox.shared.probe_payload import emit_result, json_rows_from_output, libcontentbench_metric_timeseries, load_payload


def _numeric_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _runtime_output_syscall_rate(payload: dict[str, Any]) -> dict[str, object] | None:
    rates: list[float] = []
    for row in json_rows_from_output(payload):
        sampler_summary = row.get("sampler_summary")
        if not isinstance(sampler_summary, dict):
            continue
        explicit_rate = _numeric_float(sampler_summary.get("write_count_per_sec"))
        if explicit_rate is not None:
            rates.append(explicit_rate)
            continue
        write_count_delta = _numeric_float(sampler_summary.get("write_count_delta"))
        wall_time = _numeric_float(row.get("wall_time_sec"))
        if write_count_delta is not None and wall_time is not None and wall_time > 0.0:
            rates.append(max(0.0, write_count_delta / wall_time))
    if not rates:
        return None
    value = max(rates)
    return {
        "observed": True,
        "source": "runtime_output.json.sampler_summary.write_count_delta",
        "value": value,
        "flat": {"process.io.syscall_rate": value},
        "structured": {
            "probe": "process.io.syscall_rate",
            "metric": "syscw",
            "rate_per_sec": value,
            "runtime_rows": len(rates),
        },
    }


def _runtime_output_syscall_timeseries(payload: dict[str, Any]) -> list[dict[str, float]]:
    raw_rows = libcontentbench_metric_timeseries(payload)
    parsed: list[tuple[float, float]] = []
    for row in raw_rows:
        raw_ts = row.get("ts")
        raw_write_count = row.get("write_count")
        try:
            ts = float(raw_ts) if raw_ts is not None else None
            write_count = float(raw_write_count) if raw_write_count is not None else None
        except (TypeError, ValueError):
            continue
        if ts is None or write_count is None:
            continue
        parsed.append((ts, write_count))
    if len(parsed) < 2:
        return []
    first_ts = parsed[0][0]
    points: list[dict[str, float]] = []
    previous_ts, previous_count = parsed[0]
    for ts, write_count in parsed[1:]:
        interval = ts - previous_ts
        if interval > 0:
            value = max(0.0, (write_count - previous_count) / interval)
            points.append({"time_offset_s": max(ts - first_ts, 0.0), "value": value})
        previous_ts = ts
        previous_count = write_count
    return points


def _runtime_output_syscall_timeseries_result(payload: dict[str, Any]) -> dict[str, object] | None:
    timeseries = _runtime_output_syscall_timeseries(payload)
    if not timeseries:
        return None
    value = max(point["value"] for point in timeseries)
    return {
        "observed": True,
        "source": "runtime_output.json.metrics_timeseries.write_count",
        "value": value,
        "flat": {"process.io.syscall_rate": value},
        "structured": {
            "probe": "process.io.syscall_rate",
            "metric": "syscw",
            "rate_per_sec": value,
            "timeseries": timeseries,
        },
    }


def collect(payload: dict[str, Any]) -> dict[str, object]:
    snapshot = read_proc_io_totals(target_pids(payload))
    result = build_rate_result(
        probe_name="process.io.syscall_rate",
        metric_name="syscw",
        current_snapshot=snapshot,
        previous_snapshot=None,
    )
    if result.get("observed"):
        return result
    return _runtime_output_syscall_timeseries_result(payload) or _runtime_output_syscall_rate(payload) or result


async def begin(payload: dict[str, Any]) -> dict[str, object]:
    snapshot = read_proc_io_totals(target_pids(payload))
    return {"previous_snapshot": snapshot}


async def sample(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    current = read_proc_io_totals(target_pids(payload))
    previous = state.get("previous_snapshot") if isinstance(state, dict) else None
    if isinstance(state, dict):
        state["previous_snapshot"] = current
    result = build_rate_result(
        probe_name="process.io.syscall_rate",
        metric_name="syscw",
        current_snapshot=current,
        previous_snapshot=previous if isinstance(previous, dict) else None,
    )
    if result.get("observed"):
        return result
    return _runtime_output_syscall_timeseries_result(payload) or _runtime_output_syscall_rate(payload) or result


async def finish(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    return await sample(payload, state)


if __name__ == "__main__":
    emit_result(collect(load_payload()))
