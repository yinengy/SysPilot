from __future__ import annotations

import re
from typing import cast

from toolbox.shared.probe_payload import combined_output, emit_result, json_objects_from_output, load_payload


VLC_END_TIME_PATTERN = re.compile(
    r"VLC\s+(?P<vlc_id>[12])\(transformer\)\s+end time:\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)s"
)


def _numeric(value: object) -> float | None:
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


def _from_json(payload: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in json_objects_from_output(payload):
        for key in ("walltime", "workflow_0_walltime", "workflow_1_walltime"):
            raw_value: object = item.get(key)
            parsed = _numeric(raw_value)
            if parsed is not None:
                values[key] = parsed
    return values


def _from_vlc_logs(payload: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in VLC_END_TIME_PATTERN.finditer(combined_output(payload)):
        key = f"workflow_{int(match.group('vlc_id')) - 1}_walltime"
        values[key] = float(match.group("seconds"))
    return values


def collect(payload: dict[str, object]) -> dict[str, object]:
    values = _from_json(payload)
    values.update({key: value for key, value in _from_vlc_logs(payload).items() if key not in values})
    workflow_0 = values.get("workflow_0_walltime")
    workflow_1 = values.get("workflow_1_walltime")
    gap = abs(workflow_0 - workflow_1) if workflow_0 is not None and workflow_1 is not None else None
    ratio = None
    if workflow_0 is not None and workflow_1 is not None and min(workflow_0, workflow_1) > 0:
        ratio = max(workflow_0, workflow_1) / min(workflow_0, workflow_1)
    observed = bool(values)
    flat: dict[str, float] = {}
    if "walltime" in values:
        flat["vlcs.walltime"] = values["walltime"]
    if workflow_0 is not None:
        flat["vlcs.workflow_0_walltime"] = workflow_0
    if workflow_1 is not None:
        flat["vlcs.workflow_1_walltime"] = workflow_1
    if gap is not None:
        flat["vlcs.workflow_walltime_gap"] = gap
    if ratio is not None:
        flat["vlcs.workflow_walltime_ratio"] = ratio
    return {
        "observed": observed,
        "source": "runtime_output.json_or_vlc_logs" if observed else "runtime_output_missing",
        "value": ratio if ratio is not None else values.get("walltime"),
        "flat": flat,
        "structured": {"walltimes": values, "gap_seconds": gap, "ratio": ratio} if observed else {},
    }


if __name__ == "__main__":
    emit_result(collect(cast(dict[str, object], load_payload())))
