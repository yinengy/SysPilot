from __future__ import annotations

from typing import cast

from toolbox.shared.probe_payload import emit_result, load_payload


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    raw_values = cast(list[object], value)
    result: list[int] = []
    for item in raw_values:
        if isinstance(item, int) and not isinstance(item, bool):
            result.append(item)
            continue
        if isinstance(item, str):
            try:
                result.append(int(item))
            except ValueError:
                continue
    return result


def collect(payload: dict[str, object]) -> dict[str, object]:
    target_pids = _int_list(payload.get("target_pids"))
    process_tree_targets = payload.get("process_tree_targets")
    tree_entries = cast(list[object], process_tree_targets) if isinstance(process_tree_targets, list) else []
    tree_count = len(tree_entries)
    target_count = len(target_pids)
    primary_pid = payload.get("primary_pid")
    primary: int | None = None
    if isinstance(primary_pid, int) and not isinstance(primary_pid, bool):
        primary = primary_pid
    elif isinstance(primary_pid, str):
        try:
            primary = int(primary_pid)
        except ValueError:
            primary = None
    child_targets = [pid for pid in target_pids if primary is None or pid != primary]
    observed = bool(target_pids or tree_count)
    return {
        "observed": observed,
        "source": "process_tree_targets" if observed else "process_tree_missing",
        "value": len(child_targets) if observed else None,
        "flat": {
            "vlcs.process.target_count": target_count,
            "vlcs.process.child_target_count": len(child_targets),
            "vlcs.process.tree_target_count": tree_count,
        }
        if observed
        else {},
        "structured": {
            "primary_pid": primary,
            "target_pids": target_pids,
            "child_target_pids": child_targets,
            "process_tree_target_count": tree_count,
        }
        if observed
        else {},
    }


if __name__ == "__main__":
    emit_result(collect(cast(dict[str, object], load_payload())))
