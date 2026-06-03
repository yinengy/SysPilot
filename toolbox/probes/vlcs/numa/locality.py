from __future__ import annotations

from pathlib import Path
from typing import cast

from toolbox.probes.vlcs.core.assignment import command_parts, parse_assignments
from toolbox.shared.probe_payload import emit_result, load_payload


def _node_for_core(core_id: int) -> int | None:
    cpu_dir = Path(f"/sys/devices/system/cpu/cpu{core_id}")
    try:
        for node_path in cpu_dir.glob("node*"):
            suffix = node_path.name.removeprefix("node")
            if suffix.isdigit():
                return int(suffix)
    except OSError:
        return None
    return None


def collect(payload: dict[str, object]) -> dict[str, object]:
    assignments = parse_assignments(command_parts(payload))
    workflow_nodes: dict[str, list[int]] = {}
    missing_topology: list[int] = []
    for workflow_id, cores in assignments.items():
        nodes: set[int] = set()
        for core_id in cores:
            node_id = _node_for_core(core_id)
            if node_id is None:
                missing_topology.append(core_id)
                continue
            nodes.add(node_id)
        workflow_nodes[str(workflow_id)] = sorted(nodes)
    observed = any(assignments.values()) and bool(workflow_nodes)
    cross_numa_workflows = [workflow_id for workflow_id, nodes in workflow_nodes.items() if len(nodes) > 1]
    return {
        "observed": observed,
        "source": "sysfs_cpu_nodes+command_parts" if observed else "command_parts_or_sysfs_missing",
        "value": len(cross_numa_workflows) if observed else None,
        "flat": {
            "vlcs.numa.cross_numa_workflow_count": len(cross_numa_workflows),
            "vlcs.numa.workflow_0_node_count": len(workflow_nodes.get("0", [])),
            "vlcs.numa.workflow_1_node_count": len(workflow_nodes.get("1", [])),
        }
        if observed
        else {},
        "structured": {
            "workflow_nodes": workflow_nodes,
            "cross_numa_workflows": cross_numa_workflows,
            "missing_topology_cores": sorted(set(missing_topology)),
        }
        if observed
        else {},
    }


if __name__ == "__main__":
    emit_result(collect(cast(dict[str, object], load_payload())))
