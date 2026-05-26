from __future__ import annotations

from typing import Any

from toolbox.probes.process.io._common import build_rate_result, read_proc_io_totals, target_pids
from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, object]:
    snapshot = read_proc_io_totals(target_pids(payload))
    return build_rate_result(
        probe_name="process.io.byte_rate",
        metric_name="write_bytes",
        current_snapshot=snapshot,
        previous_snapshot=None,
    )


async def begin(payload: dict[str, Any]) -> dict[str, object]:
    snapshot = read_proc_io_totals(target_pids(payload))
    return {"previous_snapshot": snapshot}


async def sample(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    current = read_proc_io_totals(target_pids(payload))
    previous = state.get("previous_snapshot") if isinstance(state, dict) else None
    if isinstance(state, dict):
        state["previous_snapshot"] = current
    return build_rate_result(
        probe_name="process.io.byte_rate",
        metric_name="write_bytes",
        current_snapshot=current,
        previous_snapshot=previous if isinstance(previous, dict) else None,
    )


async def finish(payload: dict[str, Any], state: dict[str, object] | None) -> dict[str, object]:
    return await sample(payload, state)


if __name__ == "__main__":
    emit_result(collect(load_payload()))
