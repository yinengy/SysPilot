from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from toolbox.shared.cgroup_v2 import (
    effective_cgroup_limit_bytes,
    flatten_cgroup_memory_evidence,
    normalize_cgroup_memory_evidence,
    read_cgroup_memory_snapshot,
    resolve_cgroup_v2_context,
    sample_cgroup_memory_evidence,
)
from toolbox.shared.probe_payload import load_payload


def _target_pids(payload: dict[str, Any]) -> list[int]:
    target_pids = payload.get("target_pids", [])
    if not isinstance(target_pids, list):
        return []
    results: list[int] = []
    for item in target_pids:
        try:
            results.append(int(item))
        except (TypeError, ValueError):
            continue
    return results


async def begin(payload: dict[str, Any]) -> dict[str, Any]:
    context = resolve_cgroup_v2_context(_target_pids(payload), payload.get("primary_pid"))
    stop_event = asyncio.Event()
    sample_task: asyncio.Task[dict[str, Any]] | None = None
    selected_path = context.get("selected_path")
    if isinstance(selected_path, str) and selected_path:
        sample_task = asyncio.create_task(sample_cgroup_memory_evidence(selected_path, stop_event))
    return {
        "context": context,
        "stop_event": stop_event,
        "sample_task": sample_task,
    }


def _build_probe_result(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    sample_result: dict[str, Any],
) -> dict[str, Any]:
    structured = normalize_cgroup_memory_evidence(
        requested_memory_high=str(payload.get("memory_high") or "") or None,
        cgroup_context=context,
        sample_result=sample_result,
    )
    flat = flatten_cgroup_memory_evidence(structured)
    memory_events = structured.get("memory_events", {})
    high_value = memory_events.get("high") if isinstance(memory_events, dict) else None
    latest_snapshot = sample_result.get("latest_snapshot", {})
    snapshot_exists = True
    if isinstance(latest_snapshot, dict):
        snapshot_exists = bool(latest_snapshot.get("exists", True))
    return {
        "observed": bool(structured.get("control_group")) and snapshot_exists,
        "source": str(structured.get("source", "live_cgroup_v2")),
        "value": high_value,
        "flat": flat,
        "structured": structured,
    }


def _single_snapshot_sample_result(snapshot: dict[str, Any]) -> dict[str, Any]:
    current_bytes = snapshot.get("memory_current_bytes")
    file_peak_bytes = snapshot.get("memory_peak_bytes")
    peak_candidates = [value for value in (current_bytes, file_peak_bytes) if isinstance(value, int)]
    sampled_peak_bytes = max(peak_candidates) if peak_candidates else None
    effective_limit = effective_cgroup_limit_bytes(snapshot)
    min_headroom_bytes = None
    if isinstance(effective_limit, int) and effective_limit > 0 and isinstance(current_bytes, int):
        min_headroom_bytes = effective_limit - current_bytes
    return {
        "latest_snapshot": snapshot,
        "sampled_peak_bytes": sampled_peak_bytes,
        "min_headroom_bytes": min_headroom_bytes,
        "sample_count": 1 if snapshot.get("exists") else 0,
    }


async def sample(payload: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"observed": False, "source": "os_missing", "flat": {}, "structured": {}}
    context = state.get("context", {}) if isinstance(state.get("context"), dict) else {}
    control_group = context.get("selected_path")
    if not isinstance(control_group, str) or not control_group:
        return {"observed": False, "source": "os_missing", "flat": {}, "structured": {}}
    snapshot = read_cgroup_memory_snapshot(control_group)
    return _build_probe_result(
        payload=payload,
        context=context,
        sample_result=_single_snapshot_sample_result(snapshot),
    )


async def finish(payload: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"observed": False, "source": "os_missing", "flat": {}, "structured": {}}
    sample_task = state.get("sample_task")
    stop_event = state.get("stop_event")
    context = state.get("context", {}) if isinstance(state.get("context"), dict) else {}
    if not isinstance(stop_event, asyncio.Event) or sample_task is None:
        return {"observed": False, "source": "os_missing", "flat": {}, "structured": {}}

    stop_event.set()
    sample_result = await sample_task
    return _build_probe_result(payload=payload, context=context, sample_result=sample_result)


async def _main() -> None:
    payload = load_payload()
    state = await begin(payload)
    result = await finish(payload, state)
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(_main())
