from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .ledger import (
    BASELINE_FINISHED_EVENT,
    BASELINE_STARTED_EVENT,
    EXECUTION_FINISHED_EVENT,
    EXECUTION_STARTED_EVENT,
    PROPOSAL_EVENT,
    PROPOSAL_VALIDATED_EVENT,
    TERMINAL_EVENT,
    validate_event_accounting,
)
from .manifest import SearchSpaceManifest, load_manifest


def execution_records(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    event_list = list(events)
    validate_event_accounting(event_list)
    finishes = {
        int(event["execution_sequence"]): event
        for event in event_list
        if event["event_type"] == EXECUTION_FINISHED_EVENT
    }
    records: list[dict[str, Any]] = []
    for start in event_list:
        if start["event_type"] != EXECUTION_STARTED_EVENT:
            continue
        finish = finishes.get(int(start["sequence"]))
        records.append(
            {
                "unique_execution_index": start["unique_execution_index"],
                "execution_sequence": start["sequence"],
                "proposal_sequence": start["proposal_sequence"],
                "config": start["config"],
                "config_fingerprint": start["config_fingerprint"],
                "status": None if finish is None else finish["status"],
                "metrics": {} if finish is None else dict(finish["metrics"]),
                "failure": None if finish is None else finish.get("failure"),
            }
        )
    return records


def select_best_execution(
    records: Iterable[Mapping[str, Any]],
    *,
    metric_name: str,
    mode: str,
) -> dict[str, Any] | None:
    if mode not in {"maximize", "minimize"}:
        raise ValueError("mode must be 'maximize' or 'minimize'")
    candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        value = record.get("metrics", {}).get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        candidates.append((numeric_value, int(record["unique_execution_index"]), record))
    if not candidates:
        return None
    if mode == "maximize":
        _, _, best = max(candidates, key=lambda item: (item[0], -item[1]))
    else:
        _, _, best = min(candidates, key=lambda item: (item[0], item[1]))
    return dict(best)


def summarize_events(
    events: Iterable[Mapping[str, Any]],
    *,
    manifest: SearchSpaceManifest | None = None,
    metric_name: str | None = None,
    metric_mode: str = "maximize",
) -> dict[str, Any]:
    active_manifest = manifest or load_manifest()
    event_list = list(events)
    validate_event_accounting(event_list)
    proposals = [event for event in event_list if event["event_type"] == PROPOSAL_EVENT]
    validations = {
        int(event["proposal_sequence"]): event
        for event in event_list
        if event["event_type"] == PROPOSAL_VALIDATED_EVENT
    }
    effective_proposals = [
        validations.get(int(event["sequence"]), event)
        for event in proposals
        if event.get("disposition") != "pending" or int(event["sequence"]) in validations
    ]
    records = execution_records(event_list)
    disposition_counts = Counter(str(event["disposition"]) for event in effective_proposals)
    status_counts = Counter(
        str(record["status"]) for record in records if record["status"] is not None
    )
    best = (
        None
        if metric_name is None
        else select_best_execution(records, metric_name=metric_name, mode=metric_mode)
    )
    baseline_start = next(
        (event for event in event_list if event["event_type"] == BASELINE_STARTED_EVENT),
        None,
    )
    baseline_finish = next(
        (event for event in event_list if event["event_type"] == BASELINE_FINISHED_EVENT),
        None,
    )
    terminal = next(
        (event for event in event_list if event["event_type"] == TERMINAL_EVENT),
        None,
    )
    return {
        "schema_version": 1,
        "study_id": active_manifest.study_id,
        "event_count": len(event_list),
        "proposal_count": len(proposals),
        "proposal_cap": active_manifest.proposal_cap,
        "proposal_cap_reached": len(proposals) >= active_manifest.proposal_cap,
        "pending_proposal_count": sum(
            event.get("disposition") == "pending" and int(event["sequence"]) not in validations
            for event in proposals
        ),
        "candidate_proposal_count": disposition_counts["candidate"],
        "duplicate_proposal_count": disposition_counts["duplicate"],
        "rejected_proposal_count": disposition_counts["rejected"],
        "unique_execution_count": len(records),
        "unique_execution_target": active_manifest.unique_execution_target,
        "unique_execution_target_reached": len(records) >= active_manifest.unique_execution_target,
        "completed_execution_count": sum(record["status"] is not None for record in records),
        "incomplete_execution_count": sum(record["status"] is None for record in records),
        "execution_status_counts": dict(sorted(status_counts.items())),
        "metric_name": metric_name,
        "metric_mode": metric_mode if metric_name is not None else None,
        "best_execution_index": None if best is None else best["unique_execution_index"],
        "best_config_fingerprint": None if best is None else best["config_fingerprint"],
        "best_metric_value": None if best is None else best["metrics"][metric_name],
        "baseline_recorded": baseline_start is not None,
        "baseline_status": None if baseline_finish is None else baseline_finish["status"],
        "terminal_reason": None if terminal is None else terminal.get("reason"),
    }
