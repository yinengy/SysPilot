from __future__ import annotations

from typing import cast

from toolbox.shared.probe_payload import emit_result, load_payload


CORE_COUNT = 24
WORKFLOWS = (0, 1)


def command_parts(payload: dict[str, object]) -> list[str]:
    raw = payload.get("command_parts")
    if isinstance(raw, list):
        raw_parts = cast(list[object], raw)
        parts: list[str] = []
        for part in raw_parts:
            parts.append(str(part))
        return parts
    raw_command = payload.get("command")
    if isinstance(raw_command, str):
        return raw_command.split()
    return []


def parse_assignments(parts: list[str]) -> dict[int, list[int]]:
    assignments: dict[int, list[int]] = {workflow_id: [] for workflow_id in WORKFLOWS}
    index = 0
    while index < len(parts):
        token = parts[index]
        if not token.startswith("--core-"):
            index += 1
            continue
        raw_core = token.removeprefix("--core-")
        if not raw_core.isdigit() or index + 1 >= len(parts):
            index += 1
            continue
        core_id = int(raw_core)
        try:
            workflow_id = int(parts[index + 1])
        except ValueError:
            index += 2
            continue
        if 0 <= core_id < CORE_COUNT and workflow_id in assignments:
            assignments[workflow_id].append(core_id)
        index += 2
    return {workflow_id: sorted(set(cores)) for workflow_id, cores in assignments.items()}


def collect(payload: dict[str, object]) -> dict[str, object]:
    assignments = parse_assignments(command_parts(payload))
    assigned = sorted(core_id for cores in assignments.values() for core_id in cores)
    unassigned = [core_id for core_id in range(CORE_COUNT) if core_id not in assigned]
    counts = {str(workflow_id): len(assignments[workflow_id]) for workflow_id in WORKFLOWS}
    both_workflows_assigned = all(counts[str(workflow_id)] > 0 for workflow_id in WORKFLOWS)
    observed = bool(assigned)
    count_gap = abs(counts["0"] - counts["1"])
    return {
        "observed": observed,
        "source": "command_parts" if observed else "command_parts_missing",
        "value": count_gap if observed else None,
        "flat": {
            "vlcs.core_assignment.workflow_0_count": counts["0"],
            "vlcs.core_assignment.workflow_1_count": counts["1"],
            "vlcs.core_assignment.total_assigned": len(assigned),
            "vlcs.core_assignment.count_gap": count_gap,
        }
        if observed
        else {},
        "structured": {
            "assignments": {str(workflow_id): assignments[workflow_id] for workflow_id in WORKFLOWS},
            "counts": counts,
            "unassigned": unassigned,
            "both_workflows_assigned": both_workflows_assigned,
        }
        if observed
        else {},
    }


if __name__ == "__main__":
    emit_result(collect(cast(dict[str, object], load_payload())))
