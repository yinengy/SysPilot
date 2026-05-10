from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


def parse_proc_cgroup_v2_path(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None

    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            path = parts[2].strip()
            return path or "/"
    return None


def resolve_cgroup_v2_context(
    target_pids: list[int],
    primary_pid: int | None,
) -> dict[str, Any]:
    per_pid_paths: dict[int, str] = {}
    for pid in target_pids:
        path = parse_proc_cgroup_v2_path(int(pid))
        if path:
            per_pid_paths[int(pid)] = path

    selected_path: str | None = None
    selected_pid: int | None = None
    if primary_pid is not None and int(primary_pid) in per_pid_paths:
        selected_pid = int(primary_pid)
        selected_path = per_pid_paths[selected_pid]
    elif per_pid_paths:
        counts: dict[str, int] = {}
        for path in per_pid_paths.values():
            counts[path] = counts.get(path, 0) + 1
        selected_path = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        for pid, path in per_pid_paths.items():
            if path == selected_path:
                selected_pid = pid
                break

    selected_target_pids = [pid for pid, path in per_pid_paths.items() if path == selected_path]
    return {
        "selected_path": selected_path,
        "selected_pid": selected_pid,
        "selected_target_pids": selected_target_pids,
        "per_pid_paths": {str(pid): path for pid, path in per_pid_paths.items()},
    }


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_cgroup_int_value(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_cgroup_kv_ints(raw: str | None) -> dict[str, int]:
    if raw is None:
        return {}
    parsed: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            parsed[key] = int(value)
        except ValueError:
            continue
    return parsed


def _parse_pressure_file(raw: str | None) -> dict[str, dict[str, float | int]]:
    if raw is None:
        return {}
    parsed: dict[str, dict[str, float | int]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        scope = parts[0]
        values: dict[str, float | int] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            try:
                values[key] = int(raw_value) if key == "total" else float(raw_value)
            except ValueError:
                continue
        if values:
            parsed[scope] = values
    return parsed


def read_cgroup_memory_snapshot(control_group: str) -> dict[str, Any]:
    cgroup_root = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    snapshot: dict[str, Any] = {
        "control_group": control_group,
        "cgroup_root": str(cgroup_root),
        "exists": cgroup_root.exists(),
    }
    if not cgroup_root.exists():
        return snapshot

    snapshot["memory_high_bytes"] = _parse_cgroup_int_value(_read_text_file(cgroup_root / "memory.high"))
    snapshot["memory_max_bytes"] = _parse_cgroup_int_value(_read_text_file(cgroup_root / "memory.max"))
    snapshot["memory_current_bytes"] = _parse_cgroup_int_value(_read_text_file(cgroup_root / "memory.current"))
    snapshot["memory_peak_bytes"] = _parse_cgroup_int_value(_read_text_file(cgroup_root / "memory.peak"))
    snapshot["memory_swap_current_bytes"] = _parse_cgroup_int_value(
        _read_text_file(cgroup_root / "memory.swap.current")
    )
    snapshot["memory_events"] = _parse_cgroup_kv_ints(_read_text_file(cgroup_root / "memory.events"))
    snapshot["memory_swap_events"] = _parse_cgroup_kv_ints(
        _read_text_file(cgroup_root / "memory.swap.events")
    )
    snapshot["memory_pressure"] = _parse_pressure_file(
        _read_text_file(cgroup_root / "memory.pressure")
    )
    snapshot["memory_stat"] = _parse_cgroup_kv_ints(_read_text_file(cgroup_root / "memory.stat"))
    return snapshot


def _has_cgroup_memory_snapshot_data(snapshot: dict[str, Any]) -> bool:
    return any(
        snapshot.get(key) not in (None, {}, "")
        for key in (
            "memory_current_bytes",
            "memory_peak_bytes",
            "memory_events",
            "memory_pressure",
            "memory_stat",
            "memory_swap_current_bytes",
            "memory_swap_events",
        )
    )


def _pressure_has_activity(pressure: dict[str, Any]) -> bool:
    for values in pressure.values():
        if not isinstance(values, dict):
            continue
        for value in values.values():
            if isinstance(value, (int, float)) and value > 0:
                return True
    return False


def effective_cgroup_limit_bytes(snapshot: dict[str, Any]) -> int | None:
    soft = snapshot.get("memory_high_bytes")
    hard = snapshot.get("memory_max_bytes")
    if isinstance(soft, int) and soft > 0:
        return soft
    if isinstance(hard, int) and hard > 0:
        return hard
    return None


async def sample_cgroup_memory_evidence(
    control_group: str,
    stop_event: asyncio.Event,
    *,
    sample_interval_s: float = 0.25,
) -> dict[str, Any]:
    latest_snapshot: dict[str, Any] = {}
    sampled_peak_bytes: int | None = None
    min_headroom_bytes: int | None = None
    sample_count = 0

    while True:
        snapshot = read_cgroup_memory_snapshot(control_group)
        if _has_cgroup_memory_snapshot_data(snapshot):
            latest_snapshot = snapshot
            sample_count += 1
            current_bytes = snapshot.get("memory_current_bytes")
            if isinstance(current_bytes, int):
                if sampled_peak_bytes is None or current_bytes > sampled_peak_bytes:
                    sampled_peak_bytes = current_bytes
                effective_limit = effective_cgroup_limit_bytes(snapshot)
                if isinstance(effective_limit, int) and effective_limit > 0:
                    headroom = effective_limit - current_bytes
                    if min_headroom_bytes is None or headroom < min_headroom_bytes:
                        min_headroom_bytes = headroom

        if stop_event.is_set():
            break

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sample_interval_s)
        except asyncio.TimeoutError:
            continue

    final_snapshot = read_cgroup_memory_snapshot(control_group)
    if _has_cgroup_memory_snapshot_data(final_snapshot):
        latest_snapshot = final_snapshot
        current_bytes = final_snapshot.get("memory_current_bytes")
        if isinstance(current_bytes, int):
            if sampled_peak_bytes is None or current_bytes > sampled_peak_bytes:
                sampled_peak_bytes = current_bytes
            effective_limit = effective_cgroup_limit_bytes(final_snapshot)
            if isinstance(effective_limit, int) and effective_limit > 0:
                headroom = effective_limit - current_bytes
                if min_headroom_bytes is None or headroom < min_headroom_bytes:
                    min_headroom_bytes = headroom

    return {
        "latest_snapshot": latest_snapshot,
        "sampled_peak_bytes": sampled_peak_bytes,
        "min_headroom_bytes": min_headroom_bytes,
        "sample_count": sample_count,
    }


def normalize_cgroup_memory_evidence(
    *,
    requested_memory_high: str | None,
    cgroup_context: dict[str, Any],
    sample_result: dict[str, Any],
) -> dict[str, Any]:
    latest_snapshot = sample_result.get("latest_snapshot", {})
    sampled_peak_bytes = sample_result.get("sampled_peak_bytes")
    current_bytes = latest_snapshot.get("memory_current_bytes")
    file_peak_bytes = latest_snapshot.get("memory_peak_bytes")
    peak_candidates = [
        value
        for value in (current_bytes, file_peak_bytes, sampled_peak_bytes)
        if isinstance(value, int)
    ]
    peak_bytes = max(peak_candidates) if peak_candidates else None
    effective_limit = effective_cgroup_limit_bytes(latest_snapshot)
    peak_usage_ratio = (
        (peak_bytes / effective_limit)
        if isinstance(peak_bytes, int) and isinstance(effective_limit, int) and effective_limit > 0
        else None
    )
    memory_events = latest_snapshot.get("memory_events", {})
    memory_pressure = latest_snapshot.get("memory_pressure", {})
    status = "healthy"
    if any(int(memory_events.get(key, 0)) > 0 for key in ("oom", "oom_kill", "max")):
        status = "hard_failure"
    elif int(memory_events.get("high", 0)) > 0 or _pressure_has_activity(memory_pressure):
        status = "pressured"

    return {
        "requested_memory_high": requested_memory_high,
        "control_group": cgroup_context.get("selected_path"),
        "cgroup_root": latest_snapshot.get("cgroup_root"),
        "selected_pid": cgroup_context.get("selected_pid"),
        "selected_target_pids": cgroup_context.get("selected_target_pids", []),
        "per_pid_paths": cgroup_context.get("per_pid_paths", {}),
        "soft_limit_bytes": latest_snapshot.get("memory_high_bytes"),
        "hard_limit_bytes": latest_snapshot.get("memory_max_bytes"),
        "effective_limit_bytes": effective_limit,
        "current_bytes": current_bytes,
        "peak_bytes": peak_bytes,
        "sampled_peak_bytes": sampled_peak_bytes,
        "min_headroom_bytes": sample_result.get("min_headroom_bytes"),
        "peak_usage_ratio": peak_usage_ratio,
        "sample_count": sample_result.get("sample_count", 0),
        "memory_events": memory_events,
        "memory_pressure": memory_pressure,
        "memory_stat": latest_snapshot.get("memory_stat", {}),
        "swap_current_bytes": latest_snapshot.get("memory_swap_current_bytes"),
        "swap_events": latest_snapshot.get("memory_swap_events", {}),
        "status": status,
        "source": "live_cgroup_v2",
    }


def flatten_cgroup_memory_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    simple_fields = {
        "memory.high": evidence.get("soft_limit_bytes"),
        "memory.max": evidence.get("hard_limit_bytes"),
        "memory.current": evidence.get("current_bytes"),
        "memory.peak": evidence.get("peak_bytes"),
        "memory.swap.current": evidence.get("swap_current_bytes"),
    }
    for key, value in simple_fields.items():
        if value is not None:
            flat[key] = value
            flat[f"cgroup.{key}"] = value

    for key, value in evidence.get("memory_events", {}).items():
        flat[f"memory.events.{key}"] = value
        flat[f"cgroup.memory.events.{key}"] = value

    for scope, values in evidence.get("memory_pressure", {}).items():
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            flat[f"memory.pressure.{scope}.{metric}"] = value
            flat[f"cgroup.memory.pressure.{scope}.{metric}"] = value

    for key, value in evidence.get("memory_stat", {}).items():
        flat[f"memory.stat.{key}"] = value
        flat[f"cgroup.memory.stat.{key}"] = value

    for key, value in evidence.get("swap_events", {}).items():
        flat[f"memory.swap.events.{key}"] = value
        flat[f"cgroup.memory.swap.events.{key}"] = value

    return flat
