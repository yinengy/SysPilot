from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class RepositoryStudySourceUnavailableError(ImportError):
    pass


@dataclass(frozen=True, slots=True)
class StudyProposal:
    sequence: int
    config: dict[str, Any] | None
    error: str | None


class NestedCpuBlasStudyTrace:
    def __init__(
        self,
        path: Path,
        *,
        proposal_cap: int | None = None,
        unique_execution_target: int | None = None,
    ) -> None:
        target = "experiments.LibContentBench.nestedcpu-blas.case_study"
        try:
            case_study = importlib.import_module(target)
        except ModuleNotFoundError as exc:
            missing_name = exc.name
            if missing_name == target or (
                missing_name is not None and target.startswith(f"{missing_name}.")
            ):
                wheel_message = " ".join(
                    (
                        "The nestedcpu-blas research study sources are not included",
                        "in production wheels.",
                    )
                )
                message = "\n".join(
                    (
                        "--study-ledger requires a SysPilot repository source checkout.",
                        wheel_message,
                    )
                )
                raise RepositoryStudySourceUnavailableError(
                    message
                ) from exc
            raise
        manifest = case_study.load_manifest()
        if proposal_cap is not None or unique_execution_target is not None:
            manifest = replace(
                manifest,
                proposal_cap=proposal_cap or manifest.proposal_cap,
                unique_execution_target=(
                    unique_execution_target or manifest.unique_execution_target
                ),
            )
        if manifest.proposal_cap < manifest.unique_execution_target:
            raise ValueError("study proposal cap cannot be below the unique execution target")

        self._case_study = case_study
        self._manifest = manifest
        self._ledger = case_study.EventLedger(path, manifest=manifest)
        if self._ledger.events:
            raise ValueError("study ledger path must be new or empty")
        self._baseline_sequence: int | None = None
        self._baseline_finished = False
        self._terminal_finished = False

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
    def proposal_cap_reached(self) -> bool:
        return self.proposal_count >= self.proposal_cap

    @property
    def unique_execution_target_reached(self) -> bool:
        return self.unique_execution_count >= self.unique_execution_target

    @property
    def search_active(self) -> bool:
        return not self.proposal_cap_reached and not self.unique_execution_target_reached

    def _coerce_control_value(self, control: Any, value: Any) -> Any:
        if control.kind == "integer":
            if isinstance(value, bool):
                raise ValueError(f"{control.name} must be an integer")
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{control.name} must be an integer") from exc
        return str(value)

    def configuration_from_runtime(
        self,
        command_parts: Sequence[str],
        env_overrides: Mapping[str, str],
    ) -> dict[str, Any]:
        aliases = {
            alias: control
            for control in self._manifest.controls
            for alias in control.aliases
        }
        values: dict[str, Any] = {}
        for name, value in env_overrides.items():
            control = aliases.get(str(name))
            if control is not None:
                values[control.name] = self._coerce_control_value(control, value)

        index = 0
        tokens = list(command_parts)
        while index < len(tokens):
            token = tokens[index]
            option_name, separator, inline_value = token.partition("=")
            control = aliases.get(option_name)
            if control is None:
                index += 1
                continue
            if separator:
                raw_value: Any = inline_value
            elif index + 1 < len(tokens):
                index += 1
                raw_value = tokens[index]
            else:
                raise ValueError(f"Missing value for study control {option_name}")
            values[control.name] = self._coerce_control_value(control, raw_value)
            index += 1
        return self._case_study.canonicalize_config(values, self._manifest)

    def _configuration_from_changes(
        self,
        current_config: Mapping[str, Any],
        raw_candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        changes = raw_candidate.get("changes")
        if not isinstance(changes, Mapping):
            raise ValueError("candidate changes must be an object")
        proposed = dict(current_config)
        aliases = {
            alias: control
            for control in self._manifest.controls
            for alias in control.aliases
        }
        for raw_name, raw_value in changes.items():
            control = aliases.get(str(raw_name))
            if control is None:
                raise ValueError(f"unknown study control: {raw_name}")
            proposed[control.name] = self._coerce_control_value(control, raw_value)
        return self._case_study.canonicalize_config(proposed, self._manifest)

    def record_baseline_started(
        self,
        command_parts: Sequence[str],
        env_overrides: Mapping[str, str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        config = self.configuration_from_runtime(command_parts, env_overrides)
        event = self._ledger.record_baseline_started(config, metadata=metadata)
        self._baseline_sequence = int(event["sequence"])

    def record_baseline_finished(
        self,
        *,
        status: str,
        wall_time_s: float,
        objective_name: str | None,
        objective_value: float | None,
        returncode: int,
    ) -> None:
        if self._baseline_finished or self._baseline_sequence is None:
            return
        metrics: dict[str, Any] = {"wall_time_s": wall_time_s}
        if objective_name is not None and objective_value is not None:
            metrics[objective_name] = objective_value
        failure = None if status == "ok" else {"returncode": returncode}
        self._ledger.record_baseline_finished(
            self._baseline_sequence,
            status=status,
            metrics=metrics,
            failure=failure,
        )
        self._baseline_finished = True

    def record_proposals(
        self,
        raw_candidates: Sequence[object],
        current_command_parts: Sequence[str],
        current_env_overrides: Mapping[str, str],
        *,
        metadata: Mapping[str, Any],
    ) -> dict[int, StudyProposal]:
        current_config = self.configuration_from_runtime(
            current_command_parts,
            current_env_overrides,
        )
        proposals: dict[int, StudyProposal] = {}
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            if self.proposal_cap_reached:
                break
            raw_mapping = (
                dict(raw_candidate)
                if isinstance(raw_candidate, Mapping)
                else {"invalid_candidate": raw_candidate}
            )
            event = self._ledger.record_proposal_received(
                raw_mapping,
                optimizer="concord-syspilot",
                metadata={**metadata, "model_candidate_index": candidate_index},
            )
            config: dict[str, Any] | None = None
            error: str | None = None
            try:
                config = self._configuration_from_changes(current_config, raw_mapping)
            except ValueError as exc:
                error = str(exc)
            proposals[candidate_index] = StudyProposal(
                sequence=int(event["sequence"]),
                config=config,
                error=error,
            )
        return proposals

    def record_runtime_candidate(
        self,
        command_parts: Sequence[str],
        env_overrides: Mapping[str, str],
        *,
        metadata: Mapping[str, Any],
    ) -> StudyProposal | None:
        if self.proposal_cap_reached:
            return None
        config = self.configuration_from_runtime(command_parts, env_overrides)
        event = self._ledger.record_proposal_received(
            {"runtime_config": config},
            optimizer="concord-syspilot-runtime-retry",
            metadata=metadata,
        )
        return StudyProposal(sequence=int(event["sequence"]), config=config, error=None)

    def validate_proposal(
        self,
        proposal: StudyProposal,
        *,
        accepted: bool,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if proposal.error is not None or proposal.config is None:
            self._ledger.record_proposal_validation(
                proposal.sequence,
                config=None,
                disposition="rejected",
                reason=proposal.error or reason,
                metadata=metadata,
            )
            return "rejected"
        if not accepted:
            self._ledger.record_proposal_validation(
                proposal.sequence,
                config=None,
                disposition="rejected",
                reason=reason,
                metadata=metadata,
            )
            return "rejected"

        fingerprint = self._case_study.canonical_config_fingerprint(
            proposal.config,
            self._manifest,
        )
        duplicate_proposal_sequence: int | None = None
        duplicate_baseline_sequence: int | None = None
        for event in self._ledger.events:
            if event.get("event_type") == "baseline.started" and event.get(
                "config_fingerprint"
            ) == fingerprint:
                duplicate_baseline_sequence = int(event["sequence"])
            if event.get("event_type") == "proposal.validated" and event.get(
                "config_fingerprint"
            ) == fingerprint:
                duplicate_proposal_sequence = int(event["proposal_sequence"])
                break
            if (
                event.get("event_type") == "proposal.recorded"
                and event.get("disposition") == "candidate"
                and event.get("config_fingerprint") == fingerprint
            ):
                duplicate_proposal_sequence = int(event["sequence"])
                break

        if duplicate_proposal_sequence is not None or duplicate_baseline_sequence is not None:
            self._ledger.record_proposal_validation(
                proposal.sequence,
                config=proposal.config,
                disposition="duplicate",
                reason="duplicate_configuration",
                duplicate_of_proposal_sequence=duplicate_proposal_sequence,
                duplicate_of_baseline_sequence=duplicate_baseline_sequence,
                metadata=metadata,
            )
            return "duplicate"

        self._ledger.record_proposal_validation(
            proposal.sequence,
            config=proposal.config,
            disposition="candidate",
            reason=reason,
            metadata=metadata,
        )
        return "candidate"

    def record_execution_started(
        self,
        proposal: StudyProposal,
        *,
        metadata: Mapping[str, Any],
    ) -> int:
        event = self._ledger.record_execution_started(proposal.sequence, metadata=metadata)
        return int(event["sequence"])

    def record_execution_finished(
        self,
        execution_sequence: int,
        *,
        status: str,
        wall_time_s: float,
        objective_name: str | None,
        objective_value: float | None,
        returncode: int,
        timed_out: bool,
        failure_reason: str | None = None,
    ) -> None:
        metrics: dict[str, Any] = {"wall_time_s": wall_time_s}
        if objective_name is not None and objective_value is not None:
            metrics[objective_name] = objective_value
        failure = None
        if status != "ok":
            failure = {
                "returncode": returncode,
                "timed_out": timed_out,
                "reason": failure_reason,
            }
        self._ledger.record_execution_finished(
            execution_sequence,
            status=status,
            metrics=metrics,
            failure=failure,
        )

    def finish(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        if self._terminal_finished:
            return
        if self.unique_execution_target_reached:
            reason = "unique_execution_target_reached"
        elif self.proposal_cap_reached:
            reason = "proposal_cap_reached_with_shortfall"
        else:
            reason = "terminated_with_shortfall"
        self._ledger.record_terminal(
            reason=reason,
            metadata={
                **dict(metadata or {}),
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
