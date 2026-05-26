from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any


def target_pids(payload: dict[str, Any]) -> list[int]:
    seen: set[int] = set()
    results: list[int] = []

    primary_pid = payload.get("primary_pid")
    try:
        if primary_pid is not None:
            pid = int(primary_pid)
            if pid > 0 and pid not in seen:
                seen.add(pid)
                results.append(pid)
    except (TypeError, ValueError):
        pass

    raw_target_pids = payload.get("target_pids", [])
    if isinstance(raw_target_pids, list):
        for item in raw_target_pids:
            try:
                pid = int(item)
            except (TypeError, ValueError):
                continue
            if pid > 0 and pid not in seen:
                seen.add(pid)
                results.append(pid)
    return results


def read_proc_io_totals(pids: list[int]) -> dict[str, object]:
    totals = {"write_bytes": 0, "syscw": 0}
    observed_pids: list[int] = []
    missing_pids: list[int] = []
    for pid in pids:
        path = Path("/proc") / str(pid) / "io"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            missing_pids.append(pid)
            continue
        fields: dict[str, int] = {}
        for line in text.splitlines():
            key, _sep, raw_value = line.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()
            if key not in {"write_bytes", "syscw"}:
                continue
            try:
                fields[key] = int(raw_value)
            except ValueError:
                continue
        observed_pids.append(pid)
        totals["write_bytes"] += fields.get("write_bytes", 0)
        totals["syscw"] += fields.get("syscw", 0)
    return {
        "observed_pids": observed_pids,
        "missing_pids": missing_pids,
        "totals": totals,
        "timestamp": monotonic(),
    }


def build_rate_result(
    *,
    probe_name: str,
    metric_name: str,
    current_snapshot: dict[str, object],
    previous_snapshot: dict[str, object] | None,
) -> dict[str, object]:
    observed_pids = current_snapshot.get("observed_pids", [])
    if not isinstance(observed_pids, list) or not observed_pids:
        return {
            "observed": False,
            "source": "os_missing",
            "value": None,
            "flat": {},
            "structured": {},
        }

    totals = current_snapshot.get("totals", {})
    current_total = totals.get(metric_name) if isinstance(totals, dict) else None
    current_time = current_snapshot.get("timestamp")
    previous_total = None
    previous_time = None
    if isinstance(previous_snapshot, dict):
        prior_totals = previous_snapshot.get("totals", {})
        if isinstance(prior_totals, dict):
            previous_total = prior_totals.get(metric_name)
        previous_time = previous_snapshot.get("timestamp")

    rate_value = None
    interval_seconds = None
    if isinstance(current_total, int) and isinstance(previous_total, int):
        delta = current_total - previous_total
        if isinstance(current_time, (int, float)) and isinstance(previous_time, (int, float)):
            interval_seconds = float(current_time) - float(previous_time)
            if interval_seconds > 0:
                rate_value = max(0.0, float(delta) / interval_seconds)

    return {
        "observed": True,
        "source": f"procfs:{probe_name}",
        "value": rate_value,
        "flat": {probe_name: rate_value} if rate_value is not None else {},
        "structured": {
            "probe": probe_name,
            "metric": metric_name,
            "rate_per_sec": rate_value,
            "interval_seconds": interval_seconds,
            "cumulative_total": current_total,
            "observed_pids": observed_pids,
            "missing_pids": current_snapshot.get("missing_pids", []),
        },
    }
