from __future__ import annotations

import importlib.metadata
from collections import Counter
from pathlib import Path
from typing import Any

from smac import Callback

from experiments.LibContentBench.libcontentbench.tuner_trace import (
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    normalize_config,
    to_native,
)

ACQUISITION_UNAVAILABLE_REASON = "not_exposed_by_smac_2_4_public_callbacks"
EXPECTED_SMAC_VERSION = "2.4.0"
OPTIMIZER = "smac3"
TRACE_SCHEMA_VERSION = 1


def installed_smac_version() -> str:
    try:
        return importlib.metadata.version("smac")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "LibContentBench SMAC3 requires smac==2.4.0, but the smac distribution "
            "is not installed."
        ) from exc


def require_smac_2_4_0(installed_version: str | None = None) -> str:
    actual_version = installed_version or installed_smac_version()
    if actual_version != EXPECTED_SMAC_VERSION:
        raise RuntimeError(
            "LibContentBench SMAC3 requires smac==2.4.0; installed smac version is "
            f"{actual_version}."
        )
    return actual_version


def classify_origin(origin_raw: str | None, *, is_baseline: bool = False) -> str:
    if is_baseline:
        return "initial_design.baseline"

    normalized = (origin_raw or "").strip().lower()
    if "initial design" in normalized and "sobol" in normalized:
        return "initial_design.sobol"
    if "acquisition function maximizer" in normalized and "local search" in normalized:
        return "model_based.local_search"
    if "acquisition function maximizer" in normalized and "random search" in normalized:
        return "model_based.random_search"
    if normalized == "random search":
        return "random_design"
    if "max retries" in normalized or "no candidates" in normalized:
        return "random_fallback"
    return "other"


def _status_name(status: Any) -> str:
    name = getattr(status, "name", None)
    return str(name if name is not None else status)


class SmacDecisionTrace(Callback):
    def __init__(
        self,
        *,
        output_dir: Path,
        evaluator_records: list[dict[str, Any]],
        baseline_fingerprint: str,
        budget: int,
        metric_name: str,
        metric_mode: str,
        installed_smac_version: str,
        initial_design_n_configs_requested: int,
        initial_design_max_ratio: float,
        expected_baseline_ask_order: int | None = None,
        study_proposal_cap: int | None = None,
        study_unique_execution_target: int | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.evaluator_records = evaluator_records
        self.baseline_fingerprint = baseline_fingerprint
        self.budget = int(budget)
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.installed_smac_version = installed_smac_version
        self.initial_design_n_configs_requested = int(initial_design_n_configs_requested)
        self.initial_design_max_ratio = float(initial_design_max_ratio)
        self.expected_baseline_ask_order = expected_baseline_ask_order
        self.study_proposal_cap = study_proposal_cap
        self.study_unique_execution_target = study_unique_execution_target
        self.rows: list[dict[str, Any]] = []
        self._tell_order = 0
        self._claimed_execution_orders: set[int] = set()
        self._incumbent_fingerprint: str | None = None
        self._incumbent_cost: Any = None
        self._finalized = False
        self._reused_records: dict[int, dict[str, Any]] = {}

    def on_ask_end(self, smbo: Any, info: Any) -> None:
        config = normalize_config(info.config)
        fingerprint = config_fingerprint(config)
        origin_value = getattr(info.config, "origin", None)
        origin_raw = None if origin_value is None else str(origin_value)
        is_baseline = fingerprint == self.baseline_fingerprint
        self.rows.append(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "optimizer": OPTIMIZER,
                "optimizer_version": self.installed_smac_version,
                "ask_order": len(self.rows) + 1,
                "execution_order": None,
                "tell_order": None,
                "config": config,
                "config_fingerprint": fingerprint,
                "origin": classify_origin(origin_raw, is_baseline=is_baseline),
                "origin_raw": origin_raw,
                "is_baseline": is_baseline,
                "proposal_disposition": "asked",
                "result_reused": False,
                "duplicate_of_ask_order": None,
                "reused_execution_order": None,
                "trial_status": "PENDING",
                "smac_status": None,
                "cost": None,
                "metric_name": self.metric_name,
                "metric_value": None,
                "incumbent_before_fingerprint": self._incumbent_fingerprint,
                "incumbent_before_cost": self._incumbent_cost,
                "incumbent_after_fingerprint": None,
                "incumbent_after_cost": None,
                "incumbent_changed": False,
                "acquisition_score": None,
                "acquisition_score_available": False,
                "acquisition_score_unavailable_reason": ACQUISITION_UNAVAILABLE_REASON,
            }
        )

    def annotate_latest_ask(
        self,
        info: Any,
        *,
        disposition: str,
        reused_record: dict[str, Any] | None = None,
        duplicate_of_ask_order: int | None = None,
    ) -> None:
        fingerprint = config_fingerprint(info.config)
        row = self._next_pending_row(fingerprint)
        row["proposal_disposition"] = disposition
        row["duplicate_of_ask_order"] = duplicate_of_ask_order
        if reused_record is not None:
            ask_order = int(row["ask_order"])
            self._reused_records[ask_order] = reused_record
            row["result_reused"] = True
            row["reused_execution_order"] = int(reused_record["trial_index"])

    def _next_pending_row(self, fingerprint: str) -> dict[str, Any]:
        for row in self.rows:
            if row["config_fingerprint"] == fingerprint and row["tell_order"] is None:
                return row
        raise RuntimeError(f"SMAC tell had no matching ask for fingerprint {fingerprint}")

    def _next_evaluator_record(self, fingerprint: str) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for record in self.evaluator_records:
            execution_order = int(record["trial_index"])
            if execution_order in self._claimed_execution_orders:
                continue
            record_fingerprint = record.get("config_fingerprint")
            if record_fingerprint is None:
                record_fingerprint = config_fingerprint(record["config"])
            if record_fingerprint == fingerprint:
                candidates.append(record)
        if not candidates:
            return None
        record = min(candidates, key=lambda item: int(item["trial_index"]))
        self._claimed_execution_orders.add(int(record["trial_index"]))
        return record

    def _current_incumbent(self, smbo: Any) -> tuple[str | None, Any]:
        incumbent = smbo.intensifier.get_incumbent()
        if incumbent is None:
            return None, None
        return config_fingerprint(incumbent), to_native(smbo.runhistory.get_cost(incumbent))

    def on_tell_end(self, smbo: Any, info: Any, value: Any) -> bool | None:
        fingerprint = config_fingerprint(info.config)
        row = self._next_pending_row(fingerprint)
        evaluator_record = self._next_evaluator_record(fingerprint)
        reused_record = self._reused_records.get(int(row["ask_order"]))
        result_record = evaluator_record if evaluator_record is not None else reused_record
        self._tell_order += 1

        incumbent_before_fingerprint = self._incumbent_fingerprint
        incumbent_before_cost = self._incumbent_cost
        incumbent_after_fingerprint, incumbent_after_cost = self._current_incumbent(smbo)

        row.update(
            {
                "execution_order": (
                    int(evaluator_record["trial_index"]) if evaluator_record is not None else None
                ),
                "tell_order": self._tell_order,
                "trial_status": (
                    str(result_record["state"])
                    if result_record is not None
                    else _status_name(value.status)
                ),
                "smac_status": _status_name(value.status),
                "cost": to_native(
                    result_record["cost"] if result_record is not None else value.cost
                ),
                "metric_value": (
                    to_native(result_record["metric_value"])
                    if result_record is not None
                    else None
                ),
                "incumbent_before_fingerprint": incumbent_before_fingerprint,
                "incumbent_before_cost": incumbent_before_cost,
                "incumbent_after_fingerprint": incumbent_after_fingerprint,
                "incumbent_after_cost": incumbent_after_cost,
                "incumbent_changed": incumbent_after_fingerprint != incumbent_before_fingerprint,
            }
        )
        self._incumbent_fingerprint = incumbent_after_fingerprint
        self._incumbent_cost = incumbent_after_cost
        return None

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        for row in self.rows:
            if row["tell_order"] is None:
                row["trial_status"] = "INCOMPLETE"

        baseline_ask_order = next(
            (row["ask_order"] for row in self.rows if row["is_baseline"]),
            None,
        )
        origin_counts = Counter(str(row["origin"]) for row in self.rows)
        completed = [row for row in self.rows if row["trial_status"] != "INCOMPLETE"]
        expected_baseline_position = self.expected_baseline_ask_order
        if expected_baseline_position is None and self.budget == 30:
            expected_baseline_position = 8
        version_matches = self.installed_smac_version == EXPECTED_SMAC_VERSION

        manifest = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "optimizer": OPTIMIZER,
            "optimizer_version": self.installed_smac_version,
            "expected_smac_version": EXPECTED_SMAC_VERSION,
            "installed_smac_version": self.installed_smac_version,
            "smac_version_matches_expected": version_matches,
            "smac_version_status": "exact_match" if version_matches else "version_mismatch",
            "budget": self.budget,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
            "baseline_config_fingerprint": self.baseline_fingerprint,
            "initial_design": {
                "n_configs_requested": self.initial_design_n_configs_requested,
                "max_ratio": self.initial_design_max_ratio,
                "additional_baseline_configs": 1,
                "healthy_budget_30_baseline_expected_position": expected_baseline_position,
            },
            "acquisition_score": {
                "available": False,
                "unavailable_reason": ACQUISITION_UNAVAILABLE_REASON,
            },
            "artifacts": ["optimizer_trace.jsonl", "trace_manifest.json", "trace_summary.json"],
        }
        summary = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "optimizer": OPTIMIZER,
            "optimizer_version": self.installed_smac_version,
            "asked_trials": len(self.rows),
            "completed_trials": len(completed),
            "incomplete_trials": len(self.rows) - len(completed),
            "baseline_config_fingerprint": self.baseline_fingerprint,
            "baseline_ask_order": baseline_ask_order,
            "healthy_budget_30_baseline_expected_position": expected_baseline_position,
            "healthy_budget_30_baseline_position_matches": (
                baseline_ask_order == expected_baseline_position
                if expected_baseline_position is not None and baseline_ask_order is not None
                else None
            ),
            "origin_counts": dict(sorted(origin_counts.items())),
            "incumbent_transition_count": sum(
                1 for row in self.rows if row["incumbent_changed"]
            ),
            "final_incumbent_fingerprint": self._incumbent_fingerprint,
            "final_incumbent_cost": self._incumbent_cost,
            "acquisition_score_available": False,
            "acquisition_score_unavailable_reason": ACQUISITION_UNAVAILABLE_REASON,
        }
        if self.study_proposal_cap is not None:
            study_metadata = {
                "enabled": True,
                "proposal_cap": self.study_proposal_cap,
                "unique_execution_target": self.study_unique_execution_target,
                "baseline_excluded_from_unique_execution_count": True,
            }
            manifest["study"] = study_metadata
            summary["study"] = study_metadata

        atomic_write_jsonl(self.output_dir / "optimizer_trace.jsonl", self.rows)
        atomic_write_json(self.output_dir / "trace_manifest.json", manifest)
        atomic_write_json(self.output_dir / "trace_summary.json", summary)
