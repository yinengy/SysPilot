from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

case_study = importlib.import_module("case_study")
CanonicalConfig = case_study.CanonicalConfig
EventLedger = case_study.EventLedger
LedgerFormatError = case_study.LedgerFormatError
canonical_config_fingerprint = case_study.canonical_config_fingerprint
canonicalize_config = case_study.canonicalize_config
config_delta = case_study.config_delta
load_manifest = case_study.load_manifest
read_events = case_study.read_events
summarize_events = case_study.summarize_events

EXPECTED_BASELINE: dict[str, object] = {
    "outer_jobs": 6,
    "omp_threads": 6,
    "openblas_threads": 6,
    "mkl_threads": 1,
    "blis_threads": 1,
    "numexpr_threads": 1,
    "backend_prefer": "processes",
}


def _baseline(**changes: object) -> dict[str, object]:
    values = dict(EXPECTED_BASELINE)
    values.update(changes)
    return values


def test_pinned_manifest_has_exact_domain_workload_budgets_and_reference_line() -> None:
    manifest = load_manifest()

    assert manifest.workload.to_dict() == {
        "matrix_size": 320,
        "task_count": 800,
        "repetitions": 6,
    }
    assert manifest.control_order == (
        "outer_jobs",
        "omp_threads",
        "openblas_threads",
        "mkl_threads",
        "blis_threads",
        "numexpr_threads",
        "backend_prefer",
    )
    assert manifest.baseline == EXPECTED_BASELINE
    assert manifest.domain_cardinality == 24_461_180_928
    assert manifest.declared_domain_cardinality == 24_461_180_928
    assert manifest.proposal_cap == 60
    assert manifest.unique_execution_target == 30

    reference = manifest.reference_configurations()
    assert len(reference) == 48
    assert [point["outer_jobs"] for point in reference] == list(range(1, 49))
    assert all(
        point
        == {
            "outer_jobs": point["outer_jobs"],
            "omp_threads": 1,
            "openblas_threads": 1,
            "mkl_threads": 1,
            "blis_threads": 1,
            "numexpr_threads": 1,
            "backend_prefer": "threads",
        }
        for point in reference
    )


def test_aliases_canonicalize_deterministically_and_fingerprints_are_stable() -> None:
    aliased = {
        "--backend-prefer": "processes",
        "NUMEXPR_NUM_THREADS": 1,
        "BLIS_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 6,
        "OMP_NUM_THREADS": 6,
        "--outer-jobs": 6,
    }
    canonical = canonicalize_config(aliased)

    assert canonical == _baseline()
    assert CanonicalConfig.from_mapping(aliased).to_dict() == _baseline()
    assert canonical_config_fingerprint(aliased) == canonical_config_fingerprint(_baseline())
    assert config_delta(_baseline(), _baseline(outer_jobs=12, backend_prefer="threads")) == {
        "outer_jobs": 12,
        "backend_prefer": "threads",
    }


@pytest.mark.parametrize(
    "invalid",
    [
        _baseline(outer_jobs=0),
        _baseline(omp_threads=49),
        _baseline(backend_prefer="loky"),
        _baseline(mkl_threads=True),
        {key: value for key, value in _baseline().items() if key != "blis_threads"},
        {**_baseline(), "unknown": 1},
    ],
)
def test_invalid_configurations_are_rejected(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        canonicalize_config(invalid)


def test_conflicting_aliases_are_rejected() -> None:
    values = _baseline()
    values["--joblib-n-jobs"] = 7

    with pytest.raises(ValueError, match="Conflicting aliases"):
        canonicalize_config(values)


def test_ledger_separates_proposals_from_first_unique_executions(tmp_path: Path) -> None:
    ledger = EventLedger(tmp_path / "events.jsonl")
    first_proposal = ledger.record_proposal(_baseline(), optimizer="synthetic")
    duplicate = ledger.record_proposal(_baseline(), optimizer="synthetic")
    rejection = ledger.record_rejection(
        {"outer_jobs": 99},
        optimizer="synthetic",
        reason="outside_domain",
    )
    first_start = ledger.record_execution_started(first_proposal["sequence"])
    ledger.record_execution_finished(
        first_start["sequence"],
        status="timeout",
        failure={"timeout_sec": 10},
    )
    second_proposal = ledger.record_proposal(
        _baseline(outer_jobs=12, backend_prefer="threads"),
        optimizer="synthetic",
    )
    second_start = ledger.record_execution_started(second_proposal["sequence"])
    ledger.record_execution_finished(
        second_start["sequence"],
        status="ok",
        metrics={"throughput": 42.0},
    )

    assert duplicate["disposition"] == "duplicate"
    assert duplicate["duplicate_of_proposal_sequence"] == first_proposal["sequence"]
    assert rejection["disposition"] == "rejected"
    with pytest.raises(ValueError, match="Only first-time"):
        ledger.record_execution_started(duplicate["sequence"])

    events = read_events(ledger.path)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    summary = summarize_events(events, metric_name="throughput", metric_mode="maximize")
    assert summary["proposal_count"] == 4
    assert summary["candidate_proposal_count"] == 2
    assert summary["duplicate_proposal_count"] == 1
    assert summary["rejected_proposal_count"] == 1
    assert summary["unique_execution_count"] == 2
    assert summary["completed_execution_count"] == 2
    assert summary["incomplete_execution_count"] == 0
    assert summary["execution_status_counts"] == {"ok": 1, "timeout": 1}
    assert summary["best_execution_index"] == 2
    assert summary["best_metric_value"] == 42.0


def test_partial_traces_remain_readable_and_are_not_appended_to(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    ledger = EventLedger(path)
    proposal = ledger.record_proposal(_baseline(), optimizer="synthetic")
    ledger.record_execution_started(proposal["sequence"])

    partial_summary = summarize_events(read_events(path), metric_name="throughput")
    assert partial_summary["unique_execution_count"] == 1
    assert partial_summary["completed_execution_count"] == 0
    assert partial_summary["incomplete_execution_count"] == 1

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":1,"sequence":3')

    recovered = read_events(path)
    assert len(recovered) == 2
    assert summarize_events(recovered)["incomplete_execution_count"] == 1
    with pytest.raises(LedgerFormatError, match="non-terminated"):
        EventLedger(path).record_rejection({}, optimizer="synthetic", reason="unused")


def test_completed_malformed_line_is_not_treated_as_partial(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"schema_version": 1, "sequence": 1}) + "\nnot-json\n")

    with pytest.raises(LedgerFormatError, match="line 2"):
        read_events(path)
