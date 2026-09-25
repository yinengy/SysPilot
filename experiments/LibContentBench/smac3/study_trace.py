from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.LibContentBench.libcontentbench.tuner_trace import normalize_config


@dataclass(frozen=True, slots=True)
class StudyAsk:
    proposal_sequence: int
    config: dict[str, Any]
    config_fingerprint: str
    disposition: str
    duplicate_of_proposal_sequence: int | None
    duplicate_of_baseline_sequence: int | None


class SmacStudyTrace:
    def __init__(
        self,
        path: Path,
        *,
        proposal_cap: int | None = None,
        unique_execution_target: int | None = None,
    ) -> None:
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

        self._case_study = case_study
        self._manifest = manifest
        self._ledger = case_study.EventLedger(Path(path), manifest=manifest)
        if self._ledger.events:
            raise ValueError("study ledger path must be new or empty")
        self._baseline_sequence: int | None = None
        self._baseline_finished = False
        self._terminal_finished = False
        self._first_proposal_by_fingerprint: dict[str, int] = {}

    @property
    def proposal_cap(self) -> int:
        return int(self._manifest.proposal_cap)

    @property
    def unique_execution_target(self) -> int:
        return int(self._manifest.unique_execution_target)

    @property
    def proposal_count(self) -> int:
        return sum(
            event.get("event_type") == "proposal.recorded" for event in self._ledger.events
        )

    @property
    def unique_execution_count(self) -> int:
        return sum(
            event.get("event_type") == "execution.started" for event in self._ledger.events
        )

    @property
    def baseline_recorded(self) -> bool:
        return self._baseline_sequence is not None

    @property
    def baseline_finished(self) -> bool:
        return self._baseline_finished

    @property
    def target_reached(self) -> bool:
        return self.unique_execution_count >= self.unique_execution_target

    @property
    def proposal_cap_reached(self) -> bool:
        return self.proposal_count >= self.proposal_cap

    def record_ask(
        self,
        config: Any,
        *,
        ask_order: int,
        origin: str | None,
        is_baseline: bool,
    ) -> StudyAsk:
        raw_config = normalize_config(config)
        proposal = self._ledger.record_proposal_received(
            raw_config,
            optimizer="smac3",
            metadata={"ask_order": ask_order, "origin": origin},
        )
        canonical = self._case_study.canonicalize_config(raw_config, self._manifest)
        fingerprint = self._case_study.canonical_config_fingerprint(
            canonical,
            self._manifest,
        )

        duplicate_of_proposal = self._first_proposal_by_fingerprint.get(fingerprint)
        duplicate_of_baseline = None
        if is_baseline:
            if self._baseline_sequence is None:
                baseline = self._ledger.record_baseline_started(
                    canonical,
                    metadata={"ask_order": ask_order, "origin": origin},
                )
                self._baseline_sequence = int(baseline["sequence"])
            duplicate_of_baseline = self._baseline_sequence
            disposition = "duplicate"
            reason = "baseline_configuration"
        elif duplicate_of_proposal is not None:
            disposition = "duplicate"
            reason = "duplicate_configuration"
        elif self.target_reached:
            disposition = "rejected"
            reason = "unique_execution_target_reached_before_launch"
        else:
            disposition = "candidate"
            reason = "first_unique_configuration"

        validation = self._ledger.record_proposal_validation(
            int(proposal["sequence"]),
            config=canonical if disposition != "rejected" else None,
            disposition=disposition,
            reason=reason,
            duplicate_of_proposal_sequence=duplicate_of_proposal,
            duplicate_of_baseline_sequence=duplicate_of_baseline,
            metadata={"ask_order": ask_order},
        )
        if disposition != "rejected" and fingerprint not in self._first_proposal_by_fingerprint:
            self._first_proposal_by_fingerprint[fingerprint] = int(proposal["sequence"])

        return StudyAsk(
            proposal_sequence=int(proposal["sequence"]),
            config=canonical,
            config_fingerprint=fingerprint,
            disposition=str(validation["disposition"]),
            duplicate_of_proposal_sequence=duplicate_of_proposal,
            duplicate_of_baseline_sequence=duplicate_of_baseline,
        )

    def record_execution_started(
        self,
        ask: StudyAsk,
        *,
        execution_order: int,
    ) -> int:
        event = self._ledger.record_execution_started(
            ask.proposal_sequence,
            metadata={"execution_order": execution_order},
        )
        return int(event["sequence"])

    def record_execution_finished(
        self,
        execution_sequence: int,
        record: dict[str, Any],
        *,
        metric_name: str,
    ) -> None:
        status = _ledger_status(record)
        metrics: dict[str, Any] = {"wall_time_s": float(record["elapsed_sec"])}
        if record.get("metric_value") is not None:
            metrics[metric_name] = record["metric_value"]
        failure = None
        if status != "ok":
            failure = {
                "result_state": record.get("state"),
                "returncode": record.get("returncode"),
                "timed_out": bool(record.get("timed_out")),
            }
        self._ledger.record_execution_finished(
            execution_sequence,
            status=status,
            metrics=metrics,
            failure=failure,
        )

    def record_baseline_finished(
        self,
        record: dict[str, Any],
        *,
        metric_name: str,
        reused_execution_order: int | None = None,
    ) -> None:
        if self._baseline_sequence is None or self._baseline_finished:
            return
        status = _ledger_status(record)
        metrics: dict[str, Any] = {"wall_time_s": float(record["elapsed_sec"])}
        if record.get("metric_value") is not None:
            metrics[metric_name] = record["metric_value"]
        failure = None
        if status != "ok":
            failure = {
                "result_state": record.get("state"),
                "returncode": record.get("returncode"),
                "timed_out": bool(record.get("timed_out")),
            }
        if reused_execution_order is not None:
            metrics["reused_execution_order"] = reused_execution_order
        self._ledger.record_baseline_finished(
            self._baseline_sequence,
            status=status,
            metrics=metrics,
            failure=failure,
        )
        self._baseline_finished = True

    def finish(
        self,
        *,
        stop_reason: str,
        metric_name: str,
        metric_mode: str,
    ) -> dict[str, Any]:
        if not self._terminal_finished:
            if self.target_reached and self.baseline_finished:
                reason = "unique_execution_target_reached"
            elif self.proposal_cap_reached:
                reason = "proposal_cap_reached_with_shortfall"
            else:
                reason = "terminated_with_shortfall"
            self._ledger.record_terminal(
                reason=reason,
                metadata={
                    "optimizer_stop_reason": stop_reason,
                    "proposal_count": self.proposal_count,
                    "proposal_cap": self.proposal_cap,
                    "unique_execution_count": self.unique_execution_count,
                    "unique_execution_target": self.unique_execution_target,
                    "baseline_excluded_from_unique_execution_count": True,
                    "shortfall": max(
                        self.unique_execution_target - self.unique_execution_count,
                        0,
                    ),
                },
            )
            self._terminal_finished = True
        return self._case_study.summarize_events(
            self._ledger.events,
            manifest=self._manifest,
            metric_name=metric_name,
            metric_mode=metric_mode,
        )


def _ledger_status(record: dict[str, Any]) -> str:
    state = str(record.get("state"))
    if state == "OK":
        return "ok"
    if state == "TIMEOUT":
        return "timeout"
    return "error"
