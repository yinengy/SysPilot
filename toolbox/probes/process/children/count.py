from __future__ import annotations

from typing import Any

from toolbox.probes.process.io._common import target_pids
from toolbox.shared.probe_payload import emit_result, load_payload


def _count_children(payload: dict[str, Any]) -> int | None:
    raw_primary_pid = payload.get("primary_pid")
    try:
        pid = int(raw_primary_pid) if raw_primary_pid is not None else 0
    except (TypeError, ValueError):
        pid = 0
    if pid > 0:
        try:
            import psutil

            process = psutil.Process(pid)
            return len(process.children(recursive=True))
        except Exception:
            pass

    pids = target_pids(payload)
    if not pids:
        return None
    if pid > 0 and pid in pids:
        return max(0, len(pids) - 1)
    return len(pids)


def collect(payload: dict[str, Any]) -> dict[str, object]:
    child_count = _count_children(payload)
    return {
        "observed": child_count is not None,
        "source": "psutil" if child_count is not None else "os_missing",
        "value": child_count,
        "flat": {"process.children.count": child_count} if child_count is not None else {},
        "structured": {"child_count": child_count} if child_count is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
