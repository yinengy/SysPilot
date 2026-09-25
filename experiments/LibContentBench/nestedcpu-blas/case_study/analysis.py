from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.LibContentBench.libcontentbench.tuner_trace import (
    compact_json,
    config_fingerprint,
)

from .configuration import canonical_config_fingerprint, canonicalize_config
from .ledger import (
    BASELINE_FINISHED_EVENT,
    BASELINE_STARTED_EVENT,
    EXECUTION_FINISHED_EVENT,
    EXECUTION_STARTED_EVENT,
    PROPOSAL_EVENT,
    PROPOSAL_VALIDATED_EVENT,
    TERMINAL_EVENT,
    LedgerFormatError,
    read_events,
    validate_event_accounting,
)
from .manifest import SearchSpaceManifest, load_manifest

ANALYSIS_SCHEMA_VERSION = 1
OPTIMIZER_ORDER = ("concord", "smac3", "opentuner")
OPTIMIZER_NAMES = {
    "concord": {"concord", "concord-syspilot", "concord-syspilot-runtime-retry"},
    "smac3": {"smac3"},
    "opentuner": {"opentuner"},
}
CSV_FIELDS = (
    "optimizer",
    "row_type",
    "proposal_index",
    "request_order",
    "unique_execution_index",
    "config_fingerprint",
    "canonical_config",
    "baseline_delta",
    "disposition",
    "reason",
    "origin",
    "origin_raw",
    "requestor",
    "technique",
    "selection_provenance",
    "execution_status",
    "metric",
    "wall_time_s",
    "incumbent_metric",
    "is_incumbent",
    "best_so_far_metric",
    "best_so_far_fingerprint",
    "duplicate_of_proposal_index",
    "duplicate_of_baseline",
    "reference_member",
    "reference_distance",
    "terminal_reason",
)


class AnalysisValidationError(ValueError):
    pass


def reference_line_distance(
    config: Mapping[str, Any], manifest: SearchSpaceManifest | None = None
) -> tuple[bool, int]:
    active_manifest = manifest or load_manifest()
    canonical = canonicalize_config(config, active_manifest)
    fixed = dict(active_manifest.reference_fixed)
    distance = 0
    for control in active_manifest.controls:
        if control.name == active_manifest.reference_vary:
            continue
        actual = canonical[control.name]
        expected = fixed[control.name]
        if control.kind == "integer":
            distance += abs(int(actual) - int(expected))
        else:
            distance += int(actual != expected)
    return distance == 0, distance


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnalysisValidationError(f"Cannot read native trace {path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisValidationError(
                f"Invalid JSON in native trace {path} at line {index}"
            ) from exc
        if not isinstance(row, dict):
            raise AnalysisValidationError(
                f"Native trace {path} line {index} is not a JSON object"
            )
        rows.append(row)
    return rows


def _load_native(path: Path | None, optimizer: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {}
    trace_path = path / "optimizer_trace.jsonl" if path.is_dir() else path
    manifest_path = path / "trace_manifest.json" if path.is_dir() else path.with_name(
        "trace_manifest.json"
    )
    rows = _load_jsonl(trace_path)
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise AnalysisValidationError(f"Native manifest {manifest_path} is not an object")
        manifest = decoded
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != 1 or row.get("optimizer") != optimizer:
            raise AnalysisValidationError(
                f"Native {optimizer} trace row {index} has an incompatible schema or optimizer"
            )
    return rows, manifest


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _is_better(candidate: float, incumbent: float, mode: str) -> bool:
    return candidate > incumbent if mode == "maximize" else candidate < incumbent


def _baseline_delta(metric: float | None, baseline: float | None, mode: str) -> float | None:
    if metric is None or baseline is None or baseline == 0:
        return None
    if mode == "maximize":
        return (metric - baseline) / abs(baseline)
    return (baseline - metric) / abs(baseline)


def _effective_proposals(
    events: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    validations = {
        int(event["proposal_sequence"]): event
        for event in events
        if event["event_type"] == PROPOSAL_VALIDATED_EVENT
    }
    return [
        (proposal, validations.get(int(proposal["sequence"]), proposal))
        for proposal in events
        if proposal["event_type"] == PROPOSAL_EVENT
    ]


def _validate_optimizer(events: list[dict[str, Any]], optimizer: str, path: Path) -> None:
    names = {
        str(event.get("optimizer"))
        for event in events
        if event["event_type"] == PROPOSAL_EVENT
    }
    unexpected = sorted(names - OPTIMIZER_NAMES[optimizer])
    if unexpected:
        raise AnalysisValidationError(
            f"Ledger {path} is incompatible with {optimizer}: proposal optimizer(s) "
            f"{', '.join(unexpected)}"
        )


def _validate_native(
    optimizer: str,
    proposals: list[tuple[dict[str, Any], dict[str, Any]]],
    native_rows: list[dict[str, Any]],
    manifest: SearchSpaceManifest,
) -> None:
    if len(native_rows) != len(proposals):
        raise AnalysisValidationError(
            f"{optimizer} native/shared proposal count mismatch: "
            f"native={len(native_rows)} shared={len(proposals)}"
        )
    order_field = "ask_order" if optimizer == "smac3" else "request_index"
    for proposal_index, ((proposal, effective), native) in enumerate(
        zip(proposals, native_rows, strict=True), start=1
    ):
        if native.get(order_field) != proposal_index:
            raise AnalysisValidationError(
                f"{optimizer} native order mismatch at proposal {proposal_index}: "
                f"{order_field}={native.get(order_field)!r}"
            )
        config = effective.get("config")
        if config is None and isinstance(proposal.get("raw_config"), Mapping):
            try:
                config = canonicalize_config(proposal["raw_config"], manifest)
            except ValueError:
                config = None
        expected_fingerprint = (
            None if config is None else canonical_config_fingerprint(config, manifest)
        )
        if optimizer == "opentuner":
            native_config = native.get("config")
            if not isinstance(native_config, Mapping):
                raise AnalysisValidationError(
                    f"opentuner native config is not a mapping at proposal {proposal_index}"
                )
            if config_fingerprint(native_config) != native.get("config_fingerprint"):
                message = "opentuner native fingerprint self-integrity failure"
                raise AnalysisValidationError(
                    f"{message} at proposal {proposal_index}"
                )
            try:
                native_canonical_fingerprint = canonical_config_fingerprint(
                    native_config, manifest
                )
            except ValueError as exc:
                raise AnalysisValidationError(
                    f"opentuner native config is invalid at proposal {proposal_index}: {exc}"
                ) from exc
            if expected_fingerprint != native_canonical_fingerprint:
                message = "opentuner native/shared canonical fingerprint mismatch"
                raise AnalysisValidationError(
                    f"{message} at proposal {proposal_index}"
                )
        elif expected_fingerprint != native.get("config_fingerprint"):
            raise AnalysisValidationError(
                f"{optimizer} native/shared fingerprint mismatch at proposal {proposal_index}"
            )


def _native_provenance(
    optimizer: str,
    proposal: Mapping[str, Any],
    effective: Mapping[str, Any],
    native: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if optimizer == "concord":
        raw = proposal.get("raw_config")
        raw_reason = raw.get("reason") if isinstance(raw, Mapping) else None
        reason = raw_reason or effective.get("reason")
        return {
            "origin": None,
            "origin_raw": None,
            "requestor": None,
            "technique": None,
            "selection_provenance": reason,
        }
    if native is None:
        return {
            "origin": None,
            "origin_raw": None,
            "requestor": None,
            "technique": None,
            "selection_provenance": None,
        }
    if optimizer == "smac3":
        return {
            "origin": native.get("origin"),
            "origin_raw": native.get("origin_raw"),
            "requestor": None,
            "technique": None,
            "selection_provenance": native.get("origin"),
        }
    return {
        "origin": None,
        "origin_raw": None,
        "requestor": native.get("requestor"),
        "technique": native.get("technique"),
        "selection_provenance": native.get("requestor") or native.get("technique"),
    }


def analyze_ledger(
    optimizer: str,
    ledger_path: Path,
    *,
    metric_name: str,
    metric_mode: str,
    native_path: Path | None = None,
    manifest: SearchSpaceManifest | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if optimizer not in OPTIMIZER_ORDER:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if metric_mode not in {"maximize", "minimize"}:
        raise ValueError("metric_mode must be 'maximize' or 'minimize'")
    active_manifest = manifest or load_manifest()
    if not ledger_path.exists():
        raise AnalysisValidationError(f"Shared ledger does not exist: {ledger_path}")
    try:
        events = read_events(ledger_path)
        validate_event_accounting(events)
    except LedgerFormatError as exc:
        raise AnalysisValidationError(f"Invalid shared ledger {ledger_path}: {exc}") from exc
    _validate_optimizer(events, optimizer, ledger_path)
    proposals = _effective_proposals(events)
    native_rows, native_manifest = _load_native(native_path, optimizer)
    if native_path is not None:
        _validate_native(optimizer, proposals, native_rows, active_manifest)

    baseline_start = next(
        (event for event in events if event["event_type"] == BASELINE_STARTED_EVENT), None
    )
    baseline_finish = next(
        (event for event in events if event["event_type"] == BASELINE_FINISHED_EVENT), None
    )
    terminal = next((event for event in events if event["event_type"] == TERMINAL_EVENT), None)
    starts = {
        int(event["proposal_sequence"]): event
        for event in events
        if event["event_type"] == EXECUTION_STARTED_EVENT
    }
    finishes = {
        int(event["execution_sequence"]): event
        for event in events
        if event["event_type"] == EXECUTION_FINISHED_EVENT
    }
    proposal_index_by_sequence = {
        int(proposal["sequence"]): int(proposal["proposal_index"])
        for proposal, _ in proposals
    }

    baseline_status = None if baseline_finish is None else baseline_finish.get("status")
    baseline_metrics = {} if baseline_finish is None else baseline_finish.get("metrics", {})
    baseline_metric = (
        _finite_metric(baseline_metrics.get(metric_name)) if baseline_status == "ok" else None
    )
    baseline_wall_time = _finite_metric(baseline_metrics.get("wall_time_s"))
    incumbent_metric = baseline_metric
    incumbent_fingerprint = (
        None if baseline_start is None else baseline_start.get("config_fingerprint")
    )
    cumulative_wall_time = baseline_wall_time or 0.0
    time_available = baseline_wall_time is not None
    best_proposal_index = 0 if baseline_metric is not None else None
    best_execution_index = 0 if baseline_metric is not None else None
    time_to_best = baseline_wall_time if baseline_metric is not None else None
    rows: list[dict[str, Any]] = []

    if baseline_start is not None:
        baseline_config = canonicalize_config(baseline_start["config"], active_manifest)
        reference_member, reference_distance = reference_line_distance(
            baseline_config, active_manifest
        )
        rows.append(
            {
                "optimizer": optimizer,
                "row_type": "baseline",
                "proposal_index": 0,
                "request_order": 0,
                "unique_execution_index": 0,
                "config_fingerprint": baseline_start["config_fingerprint"],
                "canonical_config": compact_json(baseline_config),
                "baseline_delta": 0.0 if baseline_metric is not None else None,
                "disposition": "baseline",
                "reason": None,
                "origin": None,
                "origin_raw": None,
                "requestor": None,
                "technique": None,
                "selection_provenance": "shared_ledger_baseline",
                "execution_status": baseline_status or "incomplete",
                "metric": baseline_metric,
                "wall_time_s": baseline_wall_time,
                "incumbent_metric": incumbent_metric,
                "is_incumbent": baseline_metric is not None,
                "best_so_far_metric": incumbent_metric,
                "best_so_far_fingerprint": incumbent_fingerprint,
                "duplicate_of_proposal_index": None,
                "duplicate_of_baseline": False,
                "reference_member": reference_member,
                "reference_distance": reference_distance,
                "terminal_reason": None if terminal is None else terminal.get("reason"),
            }
        )

    for offset, (proposal, effective) in enumerate(proposals):
        proposal_index = int(proposal["proposal_index"])
        native = native_rows[offset] if native_rows else None
        provenance = _native_provenance(optimizer, proposal, effective, native)
        config = effective.get("config")
        canonical = None if config is None else canonicalize_config(config, active_manifest)
        start = starts.get(int(proposal["sequence"]))
        finish = None if start is None else finishes.get(int(start["sequence"]))
        status = None if finish is None else str(finish["status"])
        metrics = {} if finish is None else finish.get("metrics", {})
        metric = _finite_metric(metrics.get(metric_name)) if status == "ok" else None
        wall_time = _finite_metric(metrics.get("wall_time_s"))
        if finish is not None:
            if wall_time is None:
                time_available = False
            else:
                cumulative_wall_time += wall_time
        became_incumbent = False
        if metric is not None and (
            incumbent_metric is None or _is_better(metric, incumbent_metric, metric_mode)
        ):
            incumbent_metric = metric
            incumbent_fingerprint = effective.get("config_fingerprint")
            became_incumbent = True
            best_proposal_index = proposal_index
            best_execution_index = None if start is None else int(start["unique_execution_index"])
            time_to_best = cumulative_wall_time if time_available else None
        duplicate_sequence = effective.get("duplicate_of_proposal_sequence")
        duplicate_of_index = (
            None
            if duplicate_sequence is None
            else proposal_index_by_sequence.get(int(duplicate_sequence))
        )
        reference_member, reference_distance = (
            (None, None)
            if canonical is None
            else reference_line_distance(canonical, active_manifest)
        )
        rows.append(
            {
                "optimizer": optimizer,
                "row_type": "proposal",
                "proposal_index": proposal_index,
                "request_order": proposal_index,
                "unique_execution_index": (
                    None if start is None else int(start["unique_execution_index"])
                ),
                "config_fingerprint": effective.get("config_fingerprint"),
                "canonical_config": None if canonical is None else compact_json(canonical),
                "baseline_delta": _baseline_delta(metric, baseline_metric, metric_mode),
                "disposition": effective.get("disposition", "pending"),
                "reason": effective.get("reason"),
                **provenance,
                "execution_status": status or ("launched" if start is not None else "not_launched"),
                "metric": metric,
                "wall_time_s": wall_time,
                "incumbent_metric": incumbent_metric,
                "is_incumbent": became_incumbent,
                "best_so_far_metric": incumbent_metric,
                "best_so_far_fingerprint": incumbent_fingerprint,
                "duplicate_of_proposal_index": duplicate_of_index,
                "duplicate_of_baseline": effective.get("duplicate_of_baseline_sequence")
                is not None,
                "reference_member": reference_member,
                "reference_distance": reference_distance,
                "terminal_reason": None if terminal is None else terminal.get("reason"),
            }
        )

    proposal_rows = [row for row in rows if row["row_type"] == "proposal"]
    dispositions = Counter(str(row["disposition"]) for row in proposal_rows)
    execution_rows = [row for row in proposal_rows if row["unique_execution_index"] is not None]
    completed_rows = [
        row for row in execution_rows if row["execution_status"] in {"ok", "error", "timeout"}
    ]
    statuses = Counter(str(row["execution_status"]) for row in completed_rows)
    unique_launches = len(execution_rows)
    coverage_fraction = unique_launches / active_manifest.domain_cardinality
    best_row = next(
        (
            row
            for row in reversed(rows)
            if row["config_fingerprint"] == incumbent_fingerprint
            and row["metric"] == incumbent_metric
        ),
        None,
    )
    reference_rows = [row for row in execution_rows if row["reference_member"]]
    summary = {
        "optimizer": optimizer,
        "ledger_path": str(ledger_path),
        "native_trace_path": None if native_path is None else str(native_path),
        "native_trace_supplied": native_path is not None,
        "native_trace_row_count": len(native_rows),
        "native_manifest": {
            key: native_manifest[key]
            for key in (
                "optimizer_version",
                "configured_techniques",
                "default_technique",
                "acquisition_score",
            )
            if key in native_manifest
        },
        "proposal_count": len(proposal_rows),
        "validated_proposal_count": sum(row["disposition"] != "pending" for row in proposal_rows),
        "pending_proposal_count": dispositions["pending"],
        "candidate_proposal_count": dispositions["candidate"],
        "duplicate_proposal_count": dispositions["duplicate"],
        "baseline_duplicate_proposal_count": sum(
            bool(row["duplicate_of_baseline"]) for row in proposal_rows
        ),
        "rejected_proposal_count": dispositions["rejected"],
        "unique_launch_count": unique_launches,
        "completed_launch_count": len(completed_rows),
        "incomplete_launch_count": unique_launches - len(completed_rows),
        "execution_status_counts": dict(sorted(statuses.items())),
        "ok_launch_count": statuses["ok"],
        "error_launch_count": statuses["error"],
        "timeout_launch_count": statuses["timeout"],
        "proposal_loss_before_unique_launch": len(proposal_rows) - unique_launches,
        "duplicate_loss_count": dispositions["duplicate"],
        "rejection_loss_count": dispositions["rejected"],
        "pending_or_unlaunched_candidate_loss_count": sum(
            row["unique_execution_index"] is None
            and row["disposition"] not in {"duplicate", "rejected"}
            for row in proposal_rows
        ),
        "failed_or_timed_out_launch_count": statuses["error"] + statuses["timeout"],
        "baseline_recorded": baseline_start is not None,
        "baseline_status": baseline_status,
        "baseline_metric": baseline_metric,
        "baseline_wall_time_s": baseline_wall_time,
        "best_metric": incumbent_metric,
        "best_config_fingerprint": incumbent_fingerprint,
        "best_config": (
            None
            if best_row is None or best_row["canonical_config"] is None
            else json.loads(best_row["canonical_config"])
        ),
        "best_baseline_delta": (
            None if best_row is None else best_row["baseline_delta"]
        ),
        "proposals_to_best": best_proposal_index,
        "unique_executions_to_best": best_execution_index,
        "wall_time_to_best_s": time_to_best,
        "total_measured_wall_time_s": cumulative_wall_time if time_available else None,
        "reference_member_launch_count": len(reference_rows),
        "best_reference_distance": min(
            (int(row["reference_distance"]) for row in execution_rows), default=None
        ),
        "best_config_reference_member": (
            None if best_row is None else best_row["reference_member"]
        ),
        "best_config_reference_distance": (
            None if best_row is None else best_row["reference_distance"]
        ),
        "terminal_reason": None if terminal is None else terminal.get("reason"),
        "terminal_metadata": {} if terminal is None else terminal.get("metadata", {}),
        "shortfall": max(active_manifest.unique_execution_target - unique_launches, 0),
        "coverage": {
            "unique_launches": unique_launches,
            "domain_cardinality": active_manifest.domain_cardinality,
            "exact_ratio": f"{unique_launches}/{active_manifest.domain_cardinality}",
            "percentage": f"{coverage_fraction * 100:.12f}%",
            "scientific_fraction": f"{coverage_fraction:.6e}",
        },
    }
    return summary, rows


def analyze_studies(
    inputs: Mapping[str, tuple[Path, Path | None]],
    *,
    metric_name: str,
    metric_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest()
    optimizer_summaries: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in inputs:
            continue
        ledger_path, native_path = inputs[optimizer]
        summary, rows = analyze_ledger(
            optimizer,
            ledger_path,
            metric_name=metric_name,
            metric_mode=metric_mode,
            native_path=native_path,
            manifest=manifest,
        )
        optimizer_summaries.append(summary)
        trajectory.extend(rows)
    if not optimizer_summaries:
        raise AnalysisValidationError("At least one named optimizer ledger is required")
    aggregate = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_id": manifest.study_id,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "domain": {
            "cardinality": manifest.domain_cardinality,
            "factorization": "2*48^6",
            "control_order": list(manifest.control_order),
            "baseline": manifest.baseline,
            "reference_subset_label": "evidenced_reference_line",
            "reference_subset_cardinality": len(manifest.reference_configurations()),
            "reference_subset_pattern": "(outer_jobs,1,1,1,1,1,threads)",
            "reference_distance": (
                "Manhattan distance to the nearest reference-line point; outer_jobs is free, "
                "integer fixed coordinates use absolute difference, and backend mismatch costs 1."
            ),
            "exhaustive_search_claimed": False,
        },
        "protocol": {
            "proposal_cap": manifest.proposal_cap,
            "unique_execution_target": manifest.unique_execution_target,
            "baseline_excluded_from_unique_execution_count": True,
        },
        "available_optimizers": [item["optimizer"] for item in optimizer_summaries],
        "optimizers": optimizer_summaries,
    }
    return aggregate, trajectory


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _latex_escape(value: Any) -> str:
    text = _format_cell(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _render_markdown(aggregate: Mapping[str, Any]) -> str:
    domain = aggregate["domain"]
    lines = [
        "# NestedCPU-BLAS Deep-Trace Analysis",
        "",
        f"Metric: `{aggregate['metric_name']}` ({aggregate['metric_mode']}).",
        "",
        f"The exact domain is **{domain['cardinality']:,}** points (`{domain['factorization']}`). ",
        "All coverage values below are bounded trace coverage, not exhaustive-search evidence.",
        "The 48-point reference line is an evidenced subset, not an optimum claim.",
        "",
        "| Optimizer | Proposals | Validated | Duplicates | Rejected | Unique launches | "
        "OK/Error/Timeout | Coverage | Baseline | Best | Proposals to best | "
        "Executions to best | Time to best (s) | Terminal reason |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in aggregate["optimizers"]:
        statuses = item["execution_status_counts"]
        lines.append(
            "| {optimizer} | {proposal_count} | {validated_proposal_count} | "
            "{duplicate_proposal_count} | {rejected_proposal_count} | {unique_launch_count} | "
            "{ok}/{error}/{timeout} | {coverage} | {baseline} | {best} | {proposals} | "
            "{executions} | {time} | {terminal} |".format(
                optimizer=item["optimizer"],
                proposal_count=item["proposal_count"],
                validated_proposal_count=item["validated_proposal_count"],
                duplicate_proposal_count=item["duplicate_proposal_count"],
                rejected_proposal_count=item["rejected_proposal_count"],
                unique_launch_count=item["unique_launch_count"],
                ok=statuses.get("ok", 0),
                error=statuses.get("error", 0),
                timeout=statuses.get("timeout", 0),
                coverage=item["coverage"]["percentage"],
                baseline=_format_cell(item["baseline_metric"]),
                best=_format_cell(item["best_metric"]),
                proposals=_format_cell(item["proposals_to_best"]),
                executions=_format_cell(item["unique_executions_to_best"]),
                time=_format_cell(item["wall_time_to_best_s"]),
                terminal=_format_cell(item["terminal_reason"]),
            )
        )
    lines.extend(
        [
            "",
            "## Accounting Notes",
            "",
            "- `trajectory.csv` contains the baseline row and every proposal in stable "
            "optimizer/proposal order.",
            "- Duplicate, rejection, pending/unlaunched, error, and timeout losses are "
            "reported separately.",
            "- Failed and timed-out executions never become incumbents.",
            "- Native provenance is emitted only when present: CONCORD reason, SMAC origin, "
            "and OpenTuner requestor/technique.",
            "- Reference distance is the documented coordinate distance in "
            "`aggregate_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_latex(aggregate: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Optimizer & Proposals & Unique & OK & Dup. & Reject. & "
        r"Coverage (\%) & Baseline & Best \\",
        r"\midrule",
    ]
    for item in aggregate["optimizers"]:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(item["optimizer"]),
                item["proposal_count"],
                item["unique_launch_count"],
                item["ok_launch_count"],
                item["duplicate_proposal_count"],
                item["rejected_proposal_count"],
                _latex_escape(item["coverage"]["percentage"].removesuffix("%")),
                _latex_escape(item["baseline_metric"]),
                _latex_escape(item["best_metric"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_analysis_artifacts(
    output_dir: Path,
    aggregate: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "aggregate_summary.json"
    trajectory_path = output_dir / "trajectory.csv"
    report_path = output_dir / "report.md"
    latex_path = output_dir / "table.tex"
    _atomic_write(summary_path, compact_json(aggregate) + "\n")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in trajectory:
        writer.writerow({field: _format_cell(row.get(field)) for field in CSV_FIELDS})
    _atomic_write(trajectory_path, buffer.getvalue())
    _atomic_write(report_path, _render_markdown(aggregate))
    _atomic_write(latex_path, _render_latex(aggregate))
    return {
        "summary": summary_path,
        "trajectory": trajectory_path,
        "report": report_path,
        "latex": latex_path,
    }
