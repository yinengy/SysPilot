from __future__ import annotations

import importlib.metadata
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.LibContentBench.libcontentbench.common import load_json_object
from experiments.LibContentBench.libcontentbench.tuner_trace import (
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    normalize_config,
)

EXPECTED_OPENTUNER_VERSION = "0.8.8"
OPTIMIZER = "opentuner"
TRACE_SCHEMA_VERSION = 1


class TraceAvailabilityError(RuntimeError):
    pass


class OpenTunerStudyTrace:
    def __init__(
        self,
        path: Path,
        *,
        proposal_cap: int | None = None,
        unique_execution_target: int | None = None,
    ) -> None:
        import importlib

        case_study = importlib.import_module(
            "experiments.LibContentBench.nestedcpu-blas.case_study"
        )
        manifest = case_study.load_manifest()
        if proposal_cap is not None or unique_execution_target is not None:
            manifest = replace(
                manifest,
                proposal_cap=(
                    manifest.proposal_cap if proposal_cap is None else int(proposal_cap)
                ),
                unique_execution_target=(
                    manifest.unique_execution_target
                    if unique_execution_target is None
                    else int(unique_execution_target)
                ),
            )
        if manifest.proposal_cap < manifest.unique_execution_target:
            raise ValueError("study proposal cap cannot be below the unique execution target")

        self.path = Path(path)
        self.case_study = case_study
        self.manifest = manifest
        self.baseline_config = case_study.canonicalize_config(manifest.baseline, manifest)
        self.baseline_fingerprint = case_study.canonical_config_fingerprint(
            self.baseline_config, manifest
        )

    @property
    def proposal_cap(self) -> int:
        return int(self.manifest.proposal_cap)

    @property
    def unique_execution_target(self) -> int:
        return int(self.manifest.unique_execution_target)

    def canonicalize(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.case_study.canonicalize_config(config, self.manifest)

    def fingerprint(self, config: dict[str, Any]) -> str:
        return self.case_study.canonical_config_fingerprint(config, self.manifest)

    def write_from_native_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        metric_name: str,
        metric_mode: str,
        optimizer_stop_reason: str,
    ) -> dict[str, Any]:
        ledger = self.case_study.EventLedger(self.path, manifest=self.manifest)
        if ledger.events:
            raise ValueError("study ledger path must be new or empty")

        first_proposal_by_fingerprint: dict[str, int] = {}
        baseline_sequence: int | None = None
        baseline_finished = False

        for row in rows:
            raw_config = dict(row["config"])
            proposal = ledger.record_proposal_received(
                raw_config,
                optimizer="opentuner",
                metadata={
                    "request_index": row["request_index"],
                    "desired_result_id": row["desired_result_id"],
                    "generation": row["generation"],
                    "requestor": row.get("requestor"),
                    "requestor_unavailable_reason": row.get(
                        "requestor_unavailable_reason"
                    ),
                },
            )
            canonical = self.canonicalize(raw_config)
            fingerprint = self.fingerprint(canonical)
            is_baseline = fingerprint == self.baseline_fingerprint
            duplicate_of_proposal = first_proposal_by_fingerprint.get(fingerprint)

            if is_baseline:
                if baseline_sequence is None:
                    baseline = ledger.record_baseline_started(
                        canonical,
                        metadata={"request_index": row["request_index"]},
                    )
                    baseline_sequence = int(baseline["sequence"])
                disposition = "duplicate"
                reason = "baseline_configuration"
            elif duplicate_of_proposal is not None:
                disposition = "duplicate"
                reason = "duplicate_configuration"
            elif _study_execution_count(ledger.events) >= self.unique_execution_target:
                disposition = "rejected"
                reason = "unique_execution_target_reached_before_launch"
            else:
                disposition = "candidate"
                reason = "first_unique_configuration"

            ledger.record_proposal_validation(
                int(proposal["sequence"]),
                config=canonical if disposition != "rejected" else None,
                disposition=disposition,
                reason=reason,
                duplicate_of_proposal_sequence=(
                    duplicate_of_proposal
                    if disposition == "duplicate" and not is_baseline
                    else None
                ),
                duplicate_of_baseline_sequence=(
                    baseline_sequence if disposition == "duplicate" and is_baseline else None
                ),
                metadata={
                    "request_index": row["request_index"],
                    "executed": bool(row["executed"]),
                    "reused_result": bool(row["reused_result"]),
                },
            )
            first_proposal_by_fingerprint.setdefault(fingerprint, int(proposal["sequence"]))

            if is_baseline and not baseline_finished and row["executed"]:
                assert baseline_sequence is not None
                ledger.record_baseline_finished(
                    baseline_sequence,
                    status=_study_ledger_status(row.get("result_state")),
                    metrics=_study_metrics(row, metric_name),
                    failure=_study_failure(row),
                )
                baseline_finished = True
            elif disposition == "candidate" and row["executed"]:
                execution = ledger.record_execution_started(
                    int(proposal["sequence"]),
                    metadata={
                        "request_index": row["request_index"],
                        "execution_index": row.get("execution_index"),
                        "trial_artifact": row.get("trial_artifact"),
                    },
                )
                ledger.record_execution_finished(
                    int(execution["sequence"]),
                    status=_study_ledger_status(row.get("result_state")),
                    metrics=_study_metrics(row, metric_name),
                    failure=_study_failure(row),
                )

        unique_execution_count = _study_execution_count(ledger.events)
        proposal_count = _study_proposal_count(ledger.events)
        if baseline_finished and unique_execution_count >= self.unique_execution_target:
            terminal_reason = "unique_execution_target_reached"
        elif proposal_count >= self.proposal_cap:
            terminal_reason = "proposal_cap_reached_with_shortfall"
        else:
            terminal_reason = "terminated_with_shortfall"
        ledger.record_terminal(
            reason=terminal_reason,
            metadata={
                "optimizer_stop_reason": optimizer_stop_reason,
                "proposal_count": proposal_count,
                "proposal_cap": self.proposal_cap,
                "unique_execution_count": unique_execution_count,
                "unique_execution_target": self.unique_execution_target,
                "baseline_excluded_from_unique_execution_count": True,
                "shortfall": max(self.unique_execution_target - unique_execution_count, 0),
            },
        )
        return self.case_study.summarize_events(
            ledger.events,
            manifest=self.manifest,
            metric_name=metric_name,
            metric_mode=metric_mode,
        )


def _study_proposal_count(events: tuple[dict[str, Any], ...]) -> int:
    return sum(event.get("event_type") == "proposal.recorded" for event in events)


def _study_execution_count(events: tuple[dict[str, Any], ...]) -> int:
    return sum(event.get("event_type") == "execution.started" for event in events)


def _study_ledger_status(result_state: Any) -> str:
    if result_state == "OK":
        return "ok"
    if result_state == "TIMEOUT":
        return "timeout"
    return "error"


def _study_metrics(row: dict[str, Any], metric_name: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if row.get("trial_wall_time_sec") is not None:
        metrics["wall_time_s"] = row["trial_wall_time_sec"]
    if row.get("metric_value") is not None:
        metrics[metric_name] = row["metric_value"]
    return metrics


def _study_failure(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("result_state") == "OK":
        return None
    return {
        "result_state": row.get("result_state"),
        "desired_result_id": row.get("desired_result_id"),
        "trial_artifact": row.get("trial_artifact"),
    }


def installed_opentuner_version() -> str:
    try:
        return importlib.metadata.version("opentuner")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "LibContentBench OpenTuner requires opentuner==0.8.8, but the "
            "opentuner distribution is not installed."
        ) from exc


def require_opentuner_0_8_8(installed_version: str | None = None) -> str:
    actual_version = installed_version or installed_opentuner_version()
    if actual_version != EXPECTED_OPENTUNER_VERSION:
        raise RuntimeError(
            "LibContentBench OpenTuner requires opentuner==0.8.8; installed "
            f"opentuner version is {actual_version}."
        )
    return actual_version


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_trial_artifacts(output_dir: Path) -> dict[int, dict[str, Any]]:
    artifacts: dict[int, dict[str, Any]] = {}
    for artifact_path in sorted((output_dir / "trials").glob("trial_*/trial_artifact.json")):
        artifact = load_json_object(artifact_path)
        desired_result_id = artifact.get("desired_result_id")
        if isinstance(desired_result_id, int):
            artifact["artifact_path"] = str(artifact_path.relative_to(output_dir))
            artifacts[desired_result_id] = artifact
    return artifacts


def _model_columns(model: Any) -> set[str]:
    table = getattr(model, "__table__", None)
    if table is None:
        return set()
    return {str(column.name) for column in table.columns}


def _validate_orm_schema(models: Any) -> None:
    required = {
        "DesiredResult": {
            "id",
            "configuration_id",
            "tuning_run_id",
            "generation",
            "requestor",
            "state",
            "result_id",
        },
        "Configuration": {"id", "hash", "data"},
        "Result": {"id", "state", "time", "accuracy", "was_new_best"},
        "TuningRun": {"id", "state"},
    }
    incompatible: list[str] = []
    for model_name, required_columns in required.items():
        model = getattr(models, model_name, None)
        actual_columns = _model_columns(model)
        missing = sorted(required_columns - actual_columns)
        if missing:
            incompatible.append(f"{model_name} missing {', '.join(missing)}")
    if incompatible:
        details = "; ".join(incompatible)
        raise TraceAvailabilityError(
            "OpenTuner 0.8.8 trace unavailable because the installed ORM schema "
            f"is incompatible: {details}."
        )


def build_trace_rows(
    desired_results: list[Any],
    *,
    trial_artifacts: dict[int, dict[str, Any]],
    optimizer_version: str,
) -> list[dict[str, Any]]:
    ordered_requests = sorted(desired_results, key=lambda request: int(request.id))
    first_request_by_configuration: dict[int, int] = {}
    fallback_execution_index_by_result: dict[int, int] = {}
    next_fallback_execution_index = 1
    rows: list[dict[str, Any]] = []

    for request_index, request in enumerate(ordered_requests, start=1):
        configuration_id = int(request.configuration_id)
        duplicate_of_request_index = first_request_by_configuration.get(configuration_id)
        first_request_by_configuration.setdefault(configuration_id, request_index)

        result = request.result
        result_id = None if result is None else int(result.id)
        artifact = trial_artifacts.get(int(request.id))
        executed = artifact is not None
        execution_index = None
        if artifact is not None:
            execution_index = int(artifact["trial_index"])
        elif result_id is not None and duplicate_of_request_index is None:
            execution_index = fallback_execution_index_by_result.get(result_id)
            if execution_index is None:
                execution_index = next_fallback_execution_index
                fallback_execution_index_by_result[result_id] = execution_index
                next_fallback_execution_index += 1
            executed = True

        config = normalize_config(request.configuration.data)
        artifact_state = None if artifact is None else artifact.get("result_state")
        result_state = (
            artifact_state if artifact_state is not None else getattr(result, "state", None)
        )
        reused_result = bool(duplicate_of_request_index is not None and result_id is not None)
        raw_was_new_best = None if result is None else result.was_new_best

        raw_requestor = getattr(request, "requestor", None)
        requestor = None if raw_requestor in {None, ""} else str(raw_requestor)
        rows.append(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "optimizer": OPTIMIZER,
                "optimizer_version": optimizer_version,
                "request_index": request_index,
                "desired_result_id": int(request.id),
                "generation": int(request.generation),
                "requestor": requestor,
                "requestor_unavailable_reason": (
                    None if requestor is not None else "not_recorded_by_opentuner_orm"
                ),
                "request_state": str(request.state),
                "configuration_id": configuration_id,
                "configuration_hash": str(request.configuration.hash),
                "config": config,
                "config_fingerprint": config_fingerprint(config),
                "duplicate_of_request_index": duplicate_of_request_index,
                "executed": executed,
                "execution_index": execution_index,
                "trial_artifact": None if artifact is None else artifact.get("artifact_path"),
                "trial_wall_time_sec": (
                    None
                    if artifact is None
                    else _finite_number(artifact.get("trial_wall_time_sec"))
                ),
                "metric_value": (
                    None if artifact is None else _finite_number(artifact.get("metric_value"))
                ),
                "result_id": result_id,
                "reused_result": reused_result,
                "result_accuracy": (
                    _finite_number(artifact.get("result_accuracy"))
                    if artifact is not None
                    else _finite_number(getattr(result, "accuracy", None))
                ),
                "result_time": (
                    _finite_number(artifact.get("result_time"))
                    if artifact is not None
                    else _finite_number(getattr(result, "time", None))
                ),
                "result_state": None if result_state is None else str(result_state),
                "was_new_best": (
                    bool(raw_was_new_best)
                    if executed and raw_was_new_best is not None
                    else False
                ),
            }
        )
    return rows


def _best_ok_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["executed"]
        and row["result_state"] == "OK"
        and row["result_accuracy"] is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["result_accuracy"], -row["request_index"]))


def write_trace_artifacts(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    optimizer_version: str,
    tuning_run_id: int,
    tuning_run_state: str,
    baseline_fingerprint: str,
    configured_techniques: list[str],
) -> None:
    best_row = _best_ok_row(rows)
    result_state_counts = Counter(str(row["result_state"]) for row in rows)
    requestor_counts = Counter(
        str(row["requestor"])
        if row["requestor"] is not None
        else str(row["requestor_unavailable_reason"])
        for row in rows
    )
    manifest = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "optimizer": OPTIMIZER,
        "optimizer_version": optimizer_version,
        "expected_opentuner_version": EXPECTED_OPENTUNER_VERSION,
        "installed_opentuner_version": optimizer_version,
        "opentuner_version_matches_expected": optimizer_version == EXPECTED_OPENTUNER_VERSION,
        "tuning_run_id": tuning_run_id,
        "tuning_run_state": tuning_run_state,
        "default_technique": "AUCBanditMetaTechniqueA",
        "configured_techniques": configured_techniques,
        "baseline_config_fingerprint": baseline_fingerprint,
        "request_order": "desired_result.id ascending",
        "artifacts": ["optimizer_trace.jsonl", "trace_manifest.json", "trace_summary.json"],
    }
    summary = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "optimizer": OPTIMIZER,
        "optimizer_version": optimizer_version,
        "tuning_run_id": tuning_run_id,
        "tuning_run_state": tuning_run_state,
        "requested_results": len(rows),
        "executed_results": sum(1 for row in rows if row["executed"]),
        "reused_results": sum(1 for row in rows if row["reused_result"]),
        "duplicate_requests": sum(
            1 for row in rows if row["duplicate_of_request_index"] is not None
        ),
        "incomplete_requests": sum(1 for row in rows if row["result_id"] is None),
        "result_state_counts": dict(sorted(result_state_counts.items())),
        "requestor_counts": dict(sorted(requestor_counts.items())),
        "new_best_transition_count": sum(1 for row in rows if row["was_new_best"]),
        "baseline_config_fingerprint": baseline_fingerprint,
        "best_ok_request_index": None if best_row is None else best_row["request_index"],
        "best_ok_config_fingerprint": (
            None if best_row is None else best_row["config_fingerprint"]
        ),
        "best_ok_accuracy": None if best_row is None else best_row["result_accuracy"],
    }
    atomic_write_jsonl(output_dir / "optimizer_trace.jsonl", rows)
    atomic_write_json(output_dir / "trace_manifest.json", manifest)
    atomic_write_json(output_dir / "trace_summary.json", summary)


def finalize_opentuner_trace(
    *,
    database_url: str,
    tuning_run_id: int,
    output_dir: Path,
    baseline_fingerprint: str,
    configured_techniques: list[str] | None,
    study_trace: Any | None = None,
    metric_name: str | None = None,
    metric_mode: str | None = None,
    optimizer_stop_reason: str = "opentuner_run_finished",
) -> dict[str, Any] | None:
    optimizer_version = require_opentuner_0_8_8()
    if not database_url.startswith("sqlite:///"):
        raise TraceAvailabilityError(
            "OpenTuner 0.8.8 trace extraction requires a SQLite database URL; "
            f"received {database_url!r}."
        )

    from opentuner import resultsdb
    from opentuner.resultsdb import models

    session = None
    try:
        _validate_orm_schema(models)
        _, session_factory = resultsdb.connect(database_url)
        session = session_factory()
        tuning_run = session.query(models.TuningRun).filter_by(id=tuning_run_id).one_or_none()
        if tuning_run is None:
            raise TraceAvailabilityError(
                f"OpenTuner trace unavailable: tuning run id {tuning_run_id} was not found."
            )
        desired_results = (
            session.query(models.DesiredResult)
            .filter_by(tuning_run_id=tuning_run_id)
            .order_by(models.DesiredResult.id)
            .all()
        )
        rows = build_trace_rows(
            desired_results,
            trial_artifacts=_load_trial_artifacts(output_dir),
            optimizer_version=optimizer_version,
        )
        write_trace_artifacts(
            output_dir=output_dir,
            rows=rows,
            optimizer_version=optimizer_version,
            tuning_run_id=tuning_run_id,
            tuning_run_state=str(tuning_run.state),
            baseline_fingerprint=baseline_fingerprint,
            configured_techniques=list(configured_techniques or ["AUCBanditMetaTechniqueA"]),
        )
        if study_trace is not None:
            if metric_name is None or metric_mode is None:
                raise ValueError("study trace finalization requires metric name and mode")
            return study_trace.write_from_native_rows(
                rows,
                metric_name=metric_name,
                metric_mode=metric_mode,
                optimizer_stop_reason=optimizer_stop_reason,
            )
        return None
    except TraceAvailabilityError:
        raise
    except Exception as exc:
        raise TraceAvailabilityError(
            "OpenTuner 0.8.8 trace extraction failed against the installed ORM schema."
        ) from exc
    finally:
        if session is not None:
            session.close()
