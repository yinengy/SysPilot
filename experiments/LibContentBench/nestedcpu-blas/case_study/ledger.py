from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from experiments.LibContentBench.libcontentbench.tuner_trace import (
    compact_json,
    normalize_config,
    to_native,
)

from .configuration import canonical_config_fingerprint, canonicalize_config
from .manifest import SearchSpaceManifest, load_manifest

LEDGER_SCHEMA_VERSION = 1
PROPOSAL_EVENT = "proposal.recorded"
PROPOSAL_VALIDATED_EVENT = "proposal.validated"
EXECUTION_STARTED_EVENT = "execution.started"
EXECUTION_FINISHED_EVENT = "execution.finished"
BASELINE_STARTED_EVENT = "baseline.started"
BASELINE_FINISHED_EVENT = "baseline.finished"
TERMINAL_EVENT = "study.finished"
PROPOSAL_DISPOSITIONS = {"candidate", "duplicate", "rejected"}
EXECUTION_STATUSES = {"ok", "error", "timeout"}


class LedgerFormatError(ValueError):
    pass


def _has_nonterminated_tail(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        _ = handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


def read_events(path: Path, *, tolerate_partial_tail: bool = True) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    lines = ledger_path.read_text(encoding="utf-8").splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            is_partial_tail = line_index == len(lines) and not line.endswith("\n")
            if tolerate_partial_tail and is_partial_tail:
                break
            raise LedgerFormatError(f"Invalid JSON at ledger line {line_index}") from exc
        if not isinstance(decoded, dict):
            raise LedgerFormatError(f"Ledger line {line_index} is not a JSON object")
        event = cast(dict[str, Any], decoded)
        expected_sequence = len(events) + 1
        if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerFormatError(f"Unsupported ledger schema at line {line_index}")
        if event.get("sequence") != expected_sequence:
            raise LedgerFormatError(
                f"Ledger sequence mismatch at line {line_index}: expected {expected_sequence}"
            )
        events.append(event)
    return events


def validate_event_accounting(events: Iterable[Mapping[str, Any]]) -> None:
    proposals: dict[int, Mapping[str, Any]] = {}
    validations: dict[int, Mapping[str, Any]] = {}
    starts: dict[int, Mapping[str, Any]] = {}
    started_fingerprints: set[str] = set()
    finished_execution_sequences: set[int] = set()
    baseline_start_sequence: int | None = None
    baseline_finished = False
    terminal_seen = False
    proposal_count = 0
    execution_count = 0

    for expected_sequence, event in enumerate(events, start=1):
        if event.get("sequence") != expected_sequence:
            raise LedgerFormatError(f"Expected event sequence {expected_sequence}")
        event_type = event.get("event_type")
        if event_type == PROPOSAL_EVENT:
            proposal_count += 1
            if event.get("proposal_index") != proposal_count:
                raise LedgerFormatError("Proposal indices must be contiguous")
            disposition = event.get("disposition")
            if disposition not in {*PROPOSAL_DISPOSITIONS, "pending"}:
                raise LedgerFormatError(f"Invalid proposal disposition: {disposition!r}")
            proposals[expected_sequence] = event
            continue
        if event_type == PROPOSAL_VALIDATED_EVENT:
            proposal_sequence = event.get("proposal_sequence")
            if not isinstance(proposal_sequence, int) or proposal_sequence not in proposals:
                raise LedgerFormatError("Proposal validation has no matching proposal")
            if proposal_sequence in validations:
                raise LedgerFormatError("Proposal has more than one validation")
            if proposals[proposal_sequence].get("disposition") != "pending":
                raise LedgerFormatError("Only pending proposals may be validated separately")
            disposition = event.get("disposition")
            if disposition not in PROPOSAL_DISPOSITIONS:
                raise LedgerFormatError(f"Invalid proposal disposition: {disposition!r}")
            validations[proposal_sequence] = event
            continue
        if event_type == EXECUTION_STARTED_EVENT:
            execution_count += 1
            if event.get("unique_execution_index") != execution_count:
                raise LedgerFormatError("Unique execution indices must be contiguous")
            proposal_sequence = event.get("proposal_sequence")
            if not isinstance(proposal_sequence, int):
                raise LedgerFormatError("Execution proposal sequence must be an integer")
            proposal = proposals.get(proposal_sequence)
            validation = validations.get(proposal_sequence, proposal)
            if (
                proposal is None
                or validation is None
                or validation.get("disposition") != "candidate"
            ):
                raise LedgerFormatError("Executions must reference a candidate proposal")
            fingerprint = str(event.get("config_fingerprint"))
            if fingerprint != validation.get("config_fingerprint"):
                raise LedgerFormatError("Execution fingerprint does not match its proposal")
            if fingerprint in started_fingerprints:
                raise LedgerFormatError("A configuration can have only one first execution")
            started_fingerprints.add(fingerprint)
            starts[expected_sequence] = event
            continue
        if event_type == EXECUTION_FINISHED_EVENT:
            execution_sequence = event.get("execution_sequence")
            if not isinstance(execution_sequence, int):
                raise LedgerFormatError("Execution finish sequence must be an integer")
            start = starts.get(execution_sequence)
            if start is None:
                raise LedgerFormatError("Execution finish has no matching start")
            if execution_sequence in finished_execution_sequences:
                raise LedgerFormatError("Execution has more than one finish event")
            if event.get("unique_execution_index") != start.get("unique_execution_index"):
                raise LedgerFormatError("Execution finish index does not match its start")
            if event.get("status") not in EXECUTION_STATUSES:
                raise LedgerFormatError(f"Invalid execution status: {event.get('status')!r}")
            finished_execution_sequences.add(execution_sequence)
            continue
        if event_type == BASELINE_STARTED_EVENT:
            if baseline_start_sequence is not None:
                raise LedgerFormatError("Baseline can start only once")
            baseline_start_sequence = expected_sequence
            continue
        if event_type == BASELINE_FINISHED_EVENT:
            if baseline_start_sequence is None:
                raise LedgerFormatError("Baseline finish has no matching start")
            if baseline_finished:
                raise LedgerFormatError("Baseline can finish only once")
            if event.get("baseline_sequence") != baseline_start_sequence:
                raise LedgerFormatError("Baseline finish sequence does not match its start")
            if event.get("status") not in EXECUTION_STATUSES:
                raise LedgerFormatError(f"Invalid baseline status: {event.get('status')!r}")
            baseline_finished = True
            continue
        if event_type == TERMINAL_EVENT:
            if terminal_seen:
                raise LedgerFormatError("Study can have only one terminal event")
            terminal_seen = True
            continue
        raise LedgerFormatError(f"Unknown event type: {event_type!r}")


class EventLedger:
    def __init__(
        self,
        path: Path,
        manifest: SearchSpaceManifest | None = None,
    ) -> None:
        self.path: Path = Path(path)
        self.manifest: SearchSpaceManifest = manifest or load_manifest()
        self._has_partial_tail: bool = _has_nonterminated_tail(self.path)
        self._events: list[dict[str, Any]] = read_events(self.path)
        validate_event_accounting(self._events)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._has_partial_tail:
            raise LedgerFormatError("Cannot append after a non-terminated ledger tail")
        event = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": len(self._events) + 1,
            "event_type": event_type,
            **{str(key): to_native(value) for key, value in payload.items()},
        }
        candidate_events = [*self._events, event]
        validate_event_accounting(candidate_events)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(compact_json(event) + "\n")
            handle.flush()
            _ = os.fsync(handle.fileno())
        self._events.append(event)
        return dict(event)

    def _proposal_count(self) -> int:
        return sum(event["event_type"] == PROPOSAL_EVENT for event in self._events)

    def record_proposal(
        self,
        config: Mapping[str, Any],
        *,
        optimizer: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal_index = self._proposal_count() + 1
        if proposal_index > self.manifest.proposal_cap:
            raise ValueError(f"Proposal cap of {self.manifest.proposal_cap} has been reached")
        canonical = canonicalize_config(config, self.manifest)
        fingerprint = canonical_config_fingerprint(canonical, self.manifest)
        first_proposal = next(
            (
                event
                for event in self._events
                if event["event_type"] == PROPOSAL_EVENT
                and event.get("config_fingerprint") == fingerprint
            ),
            None,
        )
        disposition = "duplicate" if first_proposal is not None else "candidate"
        return self._append(
            PROPOSAL_EVENT,
            {
                "proposal_index": proposal_index,
                "optimizer": str(optimizer),
                "disposition": disposition,
                "config": canonical,
                "config_fingerprint": fingerprint,
                "duplicate_of_proposal_sequence": (
                    None if first_proposal is None else first_proposal["sequence"]
                ),
                "reason": "duplicate_configuration" if first_proposal is not None else None,
                "metadata": normalize_config(metadata or {}),
            },
        )

    def record_proposal_received(
        self,
        raw_config: Mapping[str, Any],
        *,
        optimizer: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal_index = self._proposal_count() + 1
        if proposal_index > self.manifest.proposal_cap:
            raise ValueError(f"Proposal cap of {self.manifest.proposal_cap} has been reached")
        return self._append(
            PROPOSAL_EVENT,
            {
                "proposal_index": proposal_index,
                "optimizer": str(optimizer),
                "disposition": "pending",
                "config": None,
                "raw_config": normalize_config(raw_config),
                "config_fingerprint": None,
                "duplicate_of_proposal_sequence": None,
                "reason": None,
                "metadata": normalize_config(metadata or {}),
            },
        )

    def record_proposal_validation(
        self,
        proposal_sequence: int,
        *,
        config: Mapping[str, Any] | None,
        disposition: str,
        reason: str | None = None,
        duplicate_of_proposal_sequence: int | None = None,
        duplicate_of_baseline_sequence: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if disposition not in PROPOSAL_DISPOSITIONS:
            raise ValueError(f"Unsupported proposal disposition: {disposition!r}")
        proposal = next(
            (event for event in self._events if event["sequence"] == proposal_sequence),
            None,
        )
        if proposal is None or proposal["event_type"] != PROPOSAL_EVENT:
            raise ValueError(f"Unknown proposal sequence: {proposal_sequence}")
        if proposal.get("disposition") != "pending":
            raise ValueError("Only pending proposals may be validated separately")
        if any(
            event["event_type"] == PROPOSAL_VALIDATED_EVENT
            and event.get("proposal_sequence") == proposal_sequence
            for event in self._events
        ):
            raise ValueError("Proposal already has a validation event")

        canonical: dict[str, Any] | None = None
        fingerprint: str | None = None
        if disposition in {"candidate", "duplicate"}:
            if config is None:
                raise ValueError("Candidate and duplicate validations require a configuration")
            canonical = canonicalize_config(config, self.manifest)
            fingerprint = canonical_config_fingerprint(canonical, self.manifest)
        if (
            disposition == "duplicate"
            and duplicate_of_proposal_sequence is None
            and duplicate_of_baseline_sequence is None
        ):
            raise ValueError("Duplicate validations require proposal linkage")

        return self._append(
            PROPOSAL_VALIDATED_EVENT,
            {
                "proposal_sequence": proposal_sequence,
                "disposition": disposition,
                "config": canonical,
                "config_fingerprint": fingerprint,
                "duplicate_of_proposal_sequence": duplicate_of_proposal_sequence,
                "duplicate_of_baseline_sequence": duplicate_of_baseline_sequence,
                "reason": str(reason) if reason else None,
                "metadata": normalize_config(metadata or {}),
            },
        )
    def record_rejection(
        self,
        raw_config: Mapping[str, Any],
        *,
        optimizer: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal_index = self._proposal_count() + 1
        if proposal_index > self.manifest.proposal_cap:
            raise ValueError(f"Proposal cap of {self.manifest.proposal_cap} has been reached")
        return self._append(
            PROPOSAL_EVENT,
            {
                "proposal_index": proposal_index,
                "optimizer": str(optimizer),
                "disposition": "rejected",
                "config": None,
                "raw_config": normalize_config(raw_config),
                "config_fingerprint": None,
                "duplicate_of_proposal_sequence": None,
                "reason": str(reason),
                "metadata": normalize_config(metadata or {}),
            },
        )

    def record_execution_started(
        self,
        proposal_sequence: int,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal = next(
            (event for event in self._events if event["sequence"] == proposal_sequence),
            None,
        )
        if proposal is None or proposal["event_type"] != PROPOSAL_EVENT:
            raise ValueError(f"Unknown proposal sequence: {proposal_sequence}")
        validation = next(
            (
                event
                for event in self._events
                if event["event_type"] == PROPOSAL_VALIDATED_EVENT
                and event.get("proposal_sequence") == proposal_sequence
            ),
            proposal,
        )
        if validation["disposition"] != "candidate":
            raise ValueError("Only first-time candidate proposals may launch a workload")
        fingerprint = validation["config_fingerprint"]
        if any(
            event["event_type"] == EXECUTION_STARTED_EVENT
            and event["config_fingerprint"] == fingerprint
            for event in self._events
        ):
            raise ValueError("Configuration already has a unique workload execution")
        unique_execution_index = 1 + sum(
            event["event_type"] == EXECUTION_STARTED_EVENT for event in self._events
        )
        return self._append(
            EXECUTION_STARTED_EVENT,
            {
                "unique_execution_index": unique_execution_index,
                "proposal_sequence": proposal_sequence,
                "config": validation["config"],
                "config_fingerprint": fingerprint,
                "metadata": normalize_config(metadata or {}),
            },
        )

    def record_execution_finished(
        self,
        execution_sequence: int,
        *,
        status: str,
        metrics: Mapping[str, Any] | None = None,
        failure: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in EXECUTION_STATUSES:
            raise ValueError(f"Unsupported execution status: {status!r}")
        start = next(
            (event for event in self._events if event["sequence"] == execution_sequence),
            None,
        )
        if start is None or start["event_type"] != EXECUTION_STARTED_EVENT:
            raise ValueError(f"Unknown execution sequence: {execution_sequence}")
        if any(
            event["event_type"] == EXECUTION_FINISHED_EVENT
            and event["execution_sequence"] == execution_sequence
            for event in self._events
        ):
            raise ValueError("Execution already has a finish event")
        return self._append(
            EXECUTION_FINISHED_EVENT,
            {
                "unique_execution_index": start["unique_execution_index"],
                "execution_sequence": execution_sequence,
                "status": status,
                "metrics": normalize_config(metrics or {}),
                "failure": normalize_config(failure or {}) if failure else None,
            },
        )

    def record_baseline_started(
        self,
        config: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        canonical = canonicalize_config(config, self.manifest)
        return self._append(
            BASELINE_STARTED_EVENT,
            {
                "config": canonical,
                "config_fingerprint": canonical_config_fingerprint(canonical, self.manifest),
                "metadata": normalize_config(metadata or {}),
            },
        )

    def record_baseline_finished(
        self,
        baseline_sequence: int,
        *,
        status: str,
        metrics: Mapping[str, Any] | None = None,
        failure: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in EXECUTION_STATUSES:
            raise ValueError(f"Unsupported baseline status: {status!r}")
        return self._append(
            BASELINE_FINISHED_EVENT,
            {
                "baseline_sequence": baseline_sequence,
                "status": status,
                "metrics": normalize_config(metrics or {}),
                "failure": normalize_config(failure or {}) if failure else None,
            },
        )

    def record_terminal(
        self,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._append(
            TERMINAL_EVENT,
            {
                "reason": str(reason),
                "metadata": normalize_config(metadata or {}),
            },
        )
