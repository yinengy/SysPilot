from __future__ import annotations

from typing import Any

from toolbox.probes.process.io._common import read_proc_io_totals, target_pids
from toolbox.shared.probe_payload import emit_result, load_payload


def _build_result(snapshot: dict[str, object], previous_snapshot: dict[str, object] | None) -> dict[str, object]:
    observed_pids = snapshot.get("observed_pids", [])
    if not isinstance(observed_pids, list) or not observed_pids:
        return {
            "observed": False,
            "source": "os_missing",
            "value": None,
            "flat": {},
            "structured": {},
        }

    totals = snapshot.get("totals", {})
    current_total = totals.get("write_bytes") if isinstance(totals, dict) else None
    previous_total = None
    if isinstance(previous_snapshot, dict):
        prior_totals = previous_snapshot.get("totals", {})
        if isinstance(prior_totals, dict):
            previous_total = prior_totals.get("write_bytes")
    delta = None
    if isinstance(current_total, int) and isinstance(previous_total, int):
        delta = current_total - previous_total

    return {
        "observed": True,
        "source": "procfs:process.io.write_bytes",
        "value": current_total,
        "flat": {"process.io.write_bytes": current_total} if current_total is not None else {},
        "structured": {
            "write_bytes": current_total,
            "delta_since_previous": delta,
            "observed_pids": observed_pids,
            "missing_pids": snapshot.get("missing_pids", []),
        },
    }


def collect(payload: dict[str, Any]) -> dict[str, object]:
    return _build_result(read_proc_io_totals(target_pids(payload)), None)


async def begin(payload: dict[str, Any]) -> dict[str, object]:
    snapshot = read_proc_io_totals(target_pids(payload))
    return {"previous_snapshot": snapshot}


async def sample(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    current = read_proc_io_totals(target_pids(payload))
    previous = state.get("previous_snapshot") if isinstance(state, dict) else None
    if isinstance(state, dict):
        state["previous_snapshot"] = current
    return _build_result(current, previous if isinstance(previous, dict) else None)


async def finish(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    return await sample(payload, state)


if __name__ == "__main__":
    emit_result(collect(load_payload()))
