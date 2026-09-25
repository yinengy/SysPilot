from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

case_study = importlib.import_module("case_study")
analysis = importlib.import_module("case_study.analysis")
analyze_cli = importlib.import_module("case_study.analyze")


def _config(outer_jobs: int, *, backend: str = "threads", inner: int = 1) -> dict[str, object]:
    return {
        "outer_jobs": outer_jobs,
        "omp_threads": inner,
        "openblas_threads": inner,
        "mkl_threads": inner,
        "blis_threads": inner,
        "numexpr_threads": inner,
        "backend_prefer": backend,
    }


def _write_concord_ledger(path: Path, *, metric_name: str = "throughput") -> Path:
    ledger = case_study.EventLedger(path)
    baseline = ledger.record_baseline_started(case_study.load_manifest().baseline)
    ledger.record_baseline_finished(
        baseline["sequence"], status="ok", metrics={metric_name: 10.0, "wall_time_s": 2.0}
    )

    first = ledger.record_proposal_received(
        {"changes": {"outer_jobs": 2}, "reason": "reduce_nested_contention"},
        optimizer="concord-syspilot",
        metadata={"round": 1},
    )
    first_validation = ledger.record_proposal_validation(
        first["sequence"],
        config=_config(2),
        disposition="candidate",
        reason="safe_reversible_candidate",
    )
    first_start = ledger.record_execution_started(first["sequence"])
    ledger.record_execution_finished(
        first_start["sequence"],
        status="ok",
        metrics={metric_name: 15.0, "wall_time_s": 3.0},
    )

    duplicate = ledger.record_proposal_received(
        {"changes": {"outer_jobs": 2}, "reason": "repeat_candidate"},
        optimizer="concord-syspilot",
    )
    ledger.record_proposal_validation(
        duplicate["sequence"],
        config=_config(2),
        disposition="duplicate",
        reason="duplicate_configuration",
        duplicate_of_proposal_sequence=first["sequence"],
    )

    timeout = ledger.record_proposal(_config(3), optimizer="concord-syspilot")
    timeout_start = ledger.record_execution_started(timeout["sequence"])
    ledger.record_execution_finished(
        timeout_start["sequence"],
        status="timeout",
        metrics={"wall_time_s": 5.0},
        failure={"reason": "bounded_timeout"},
    )
    error = ledger.record_proposal(_config(4), optimizer="concord-syspilot")
    error_start = ledger.record_execution_started(error["sequence"])
    ledger.record_execution_finished(
        error_start["sequence"],
        status="error",
        metrics={"wall_time_s": 1.0},
        failure={"reason": "worker_failed"},
    )
    ledger.record_rejection(
        {"changes": {"outer_jobs": 99}},
        optimizer="concord-syspilot",
        reason="outside_domain",
    )
    ledger.record_terminal(reason="terminated_with_shortfall", metadata={"shortfall": 27})
    assert first_validation["config_fingerprint"] == case_study.canonical_config_fingerprint(
        _config(2)
    )
    return path


def _write_smac_ledger_and_native(directory: Path) -> tuple[Path, Path]:
    directory.mkdir()
    ledger_path = directory / "shared.jsonl"
    ledger = case_study.EventLedger(ledger_path)
    candidate = ledger.record_proposal(_config(2), optimizer="smac3")
    start = ledger.record_execution_started(candidate["sequence"])
    ledger.record_execution_finished(
        start["sequence"], status="ok", metrics={"throughput": 12.0, "wall_time_s": 1.0}
    )
    baseline_proposal = ledger.record_proposal_received(
        case_study.load_manifest().baseline,
        optimizer="smac3",
        metadata={"ask_order": 2},
    )
    baseline = ledger.record_baseline_started(case_study.load_manifest().baseline)
    ledger.record_proposal_validation(
        baseline_proposal["sequence"],
        config=case_study.load_manifest().baseline,
        disposition="duplicate",
        reason="baseline_configuration",
        duplicate_of_baseline_sequence=baseline["sequence"],
    )
    ledger.record_baseline_finished(
        baseline["sequence"], status="ok", metrics={"throughput": 10.0, "wall_time_s": 2.0}
    )
    ledger.record_terminal(reason="terminated_with_shortfall")

    rows = [
        {
            "schema_version": 1,
            "optimizer": "smac3",
            "ask_order": 1,
            "config_fingerprint": case_study.canonical_config_fingerprint(_config(2)),
            "origin": "initial_design.sobol",
            "origin_raw": "Initial Design: Sobol",
        },
        {
            "schema_version": 1,
            "optimizer": "smac3",
            "ask_order": 2,
            "config_fingerprint": case_study.canonical_config_fingerprint(
                case_study.load_manifest().baseline
            ),
            "origin": "initial_design.baseline",
            "origin_raw": "Additional Configurations",
        },
    ]
    native = directory / "optimizer_trace.jsonl"
    native.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (directory / "trace_manifest.json").write_text(
        json.dumps({"optimizer_version": "2.4.0", "acquisition_score": {"available": False}}),
        encoding="utf-8",
    )
    return ledger_path, directory


@pytest.mark.parametrize(
    ("mode", "candidate_metric", "expected_best"),
    [("maximize", 15.0, 15.0), ("minimize", 5.0, 5.0)],
)
def test_best_trajectory_supports_maximize_and_minimize(
    tmp_path: Path, mode: str, candidate_metric: float, expected_best: float
) -> None:
    path = tmp_path / f"{mode}.jsonl"
    ledger = case_study.EventLedger(path)
    baseline = ledger.record_baseline_started(case_study.load_manifest().baseline)
    ledger.record_baseline_finished(
        baseline["sequence"], status="ok", metrics={"score": 10.0, "wall_time_s": 2.0}
    )
    proposal = ledger.record_proposal(_config(2), optimizer="concord-syspilot")
    start = ledger.record_execution_started(proposal["sequence"])
    ledger.record_execution_finished(
        start["sequence"],
        status="ok",
        metrics={"score": candidate_metric, "wall_time_s": 3.0},
    )

    summary, rows = case_study.analyze_ledger(
        "concord", path, metric_name="score", metric_mode=mode
    )

    assert summary["best_metric"] == expected_best
    assert summary["proposals_to_best"] == 1
    assert summary["unique_executions_to_best"] == 1
    assert summary["wall_time_to_best_s"] == 5.0
    assert rows[-1]["best_so_far_metric"] == expected_best
    assert rows[-1]["baseline_delta"] == 0.5


def test_accounting_shortfall_failures_duplicates_and_missing_native(tmp_path: Path) -> None:
    path = _write_concord_ledger(tmp_path / "concord.jsonl")

    summary, rows = case_study.analyze_ledger(
        "concord", path, metric_name="throughput", metric_mode="maximize"
    )

    assert summary["proposal_count"] == 5
    assert summary["validated_proposal_count"] == 5
    assert summary["duplicate_proposal_count"] == 1
    assert summary["rejected_proposal_count"] == 1
    assert summary["unique_launch_count"] == 3
    assert summary["execution_status_counts"] == {"error": 1, "ok": 1, "timeout": 1}
    assert summary["failed_or_timed_out_launch_count"] == 2
    assert summary["shortfall"] == 27
    assert summary["native_trace_supplied"] is False
    assert summary["coverage"]["exact_ratio"] == "3/24461180928"
    assert summary["coverage"]["scientific_fraction"] == "1.226433e-10"
    assert rows[1]["selection_provenance"] == "reduce_nested_contention"
    assert rows[2]["duplicate_of_proposal_index"] == 1
    assert rows[3]["best_so_far_metric"] == 15.0


def test_native_trace_provenance_and_mismatch_validation(tmp_path: Path) -> None:
    ledger_path, native_dir = _write_smac_ledger_and_native(tmp_path / "smac")

    summary, rows = case_study.analyze_ledger(
        "smac3",
        ledger_path,
        metric_name="throughput",
        metric_mode="maximize",
        native_path=native_dir,
    )

    assert summary["native_trace_row_count"] == 2
    assert rows[1]["origin"] == "initial_design.sobol"
    assert rows[2]["origin"] == "initial_design.baseline"

    native_path = native_dir / "optimizer_trace.jsonl"
    native_rows = [json.loads(line) for line in native_path.read_text().splitlines()]
    native_rows[0]["config_fingerprint"] = "mismatch"
    native_path.write_text(
        "".join(json.dumps(row) + "\n" for row in native_rows), encoding="utf-8"
    )
    with pytest.raises(case_study.AnalysisValidationError, match="fingerprint mismatch"):
        case_study.analyze_ledger(
            "smac3",
            ledger_path,
            metric_name="throughput",
            metric_mode="maximize",
            native_path=native_dir,
        )


def test_missing_shared_or_empty_supplied_native_trace_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(case_study.AnalysisValidationError, match="does not exist"):
        case_study.analyze_ledger(
            "concord",
            tmp_path / "missing.jsonl",
            metric_name="throughput",
            metric_mode="maximize",
        )

    ledger = _write_concord_ledger(tmp_path / "concord.jsonl")
    empty_native = tmp_path / "empty.jsonl"
    empty_native.write_text("", encoding="utf-8")
    with pytest.raises(case_study.AnalysisValidationError, match="count mismatch"):
        case_study.analyze_ledger(
            "concord",
            ledger,
            metric_name="throughput",
            metric_mode="maximize",
            native_path=empty_native,
        )


def _opentuner_raw_config(outer_jobs: int) -> dict[str, object]:
    return {
        "--joblib-n-jobs": outer_jobs,
        "OMP_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
        "BLIS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
        "--backend-prefer": "threads",
    }


def _write_opentuner_ledger_and_native(
    tmp_path: Path,
    *,
    ledger_outer_jobs: int = 5,
    native_config: dict[str, object] | None = None,
    native_fingerprint: str | None = None,
) -> tuple[Path, Path]:
    ledger_path = tmp_path / "opentuner.jsonl"
    ledger = case_study.EventLedger(ledger_path)
    proposal = ledger.record_proposal(_config(ledger_outer_jobs), optimizer="opentuner")
    start = ledger.record_execution_started(proposal["sequence"])
    ledger.record_execution_finished(
        start["sequence"], status="ok", metrics={"throughput": 11.0, "wall_time_s": 1.0}
    )
    raw_config = native_config or _opentuner_raw_config(ledger_outer_jobs)
    native_path = tmp_path / "opentuner_native.jsonl"
    native_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "optimizer": "opentuner",
                "request_index": 1,
                "config": raw_config,
                "config_fingerprint": native_fingerprint or config_fingerprint(raw_config),
                "requestor": "AUCBanditMetaTechniqueA",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ledger_path, native_path


def test_opentuner_alias_config_joins_and_preserves_requestor_provenance(
    tmp_path: Path,
) -> None:
    ledger_path, native_path = _write_opentuner_ledger_and_native(tmp_path)

    summary, rows = case_study.analyze_ledger(
        "opentuner",
        ledger_path,
        metric_name="throughput",
        metric_mode="maximize",
        native_path=native_path,
    )

    assert summary["native_trace_row_count"] == 1
    assert rows[0]["requestor"] == "AUCBanditMetaTechniqueA"
    assert rows[0]["technique"] is None
    assert rows[0]["selection_provenance"] == "AUCBanditMetaTechniqueA"


def test_opentuner_corrupted_raw_fingerprint_fails_self_integrity(tmp_path: Path) -> None:
    ledger_path, native_path = _write_opentuner_ledger_and_native(
        tmp_path, native_fingerprint="corrupted"
    )

    with pytest.raises(
        case_study.AnalysisValidationError,
        match="native fingerprint self-integrity failure at proposal 1",
    ):
        case_study.analyze_ledger(
            "opentuner",
            ledger_path,
            metric_name="throughput",
            metric_mode="maximize",
            native_path=native_path,
        )


def test_opentuner_recomputed_raw_fingerprint_fails_canonical_join(tmp_path: Path) -> None:
    changed_config = _opentuner_raw_config(6)
    ledger_path, native_path = _write_opentuner_ledger_and_native(
        tmp_path,
        ledger_outer_jobs=5,
        native_config=changed_config,
        native_fingerprint=config_fingerprint(changed_config),
    )

    with pytest.raises(
        case_study.AnalysisValidationError,
        match="native/shared canonical fingerprint mismatch at proposal 1",
    ):
        case_study.analyze_ledger(
            "opentuner",
            ledger_path,
            metric_name="throughput",
            metric_mode="maximize",
            native_path=native_path,
        )


def test_opentuner_canonical_join_uses_supplied_manifest(tmp_path: Path) -> None:
    manifest = case_study.load_manifest()
    controls = tuple(
        replace(control, aliases=(*control.aliases, "native_outer_jobs"))
        if control.name == "outer_jobs"
        else control
        for control in manifest.controls
    )
    active_manifest = replace(manifest, controls=controls)
    raw_config = _opentuner_raw_config(5)
    raw_config["native_outer_jobs"] = raw_config.pop("--joblib-n-jobs")
    ledger_path, native_path = _write_opentuner_ledger_and_native(
        tmp_path, native_config=raw_config
    )

    summary, _ = case_study.analyze_ledger(
        "opentuner",
        ledger_path,
        metric_name="throughput",
        metric_mode="maximize",
        native_path=native_path,
        manifest=active_manifest,
    )

    assert summary["native_trace_row_count"] == 1


def test_reference_line_membership_and_coordinate_distance() -> None:
    assert case_study.reference_line_distance(_config(48)) == (True, 0)
    assert case_study.reference_line_distance(
        _config(7, backend="processes", inner=3)
    ) == (False, 11)


def test_outputs_are_byte_stable_and_latex_escapes(tmp_path: Path) -> None:
    ledger = _write_concord_ledger(tmp_path / "concord.jsonl")
    aggregate, trajectory = case_study.analyze_studies(
        {"concord": (ledger, None)}, metric_name="throughput", metric_mode="maximize"
    )
    first = case_study.write_analysis_artifacts(tmp_path / "first", aggregate, trajectory)
    second = case_study.write_analysis_artifacts(tmp_path / "second", aggregate, trajectory)

    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes()
    assert analysis._latex_escape("a_b&c%") == r"a\_b\&c\%"
    assert b"24,461,180,928" in first["report"].read_bytes()
    assert first["latex"].read_text(encoding="utf-8").endswith("\n")


def test_cli_writes_all_artifacts_for_a_partial_single_optimizer_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = _write_concord_ledger(tmp_path / "concord.jsonl")
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--concord-ledger",
            str(ledger),
            "--metric",
            "throughput",
            "--metric-mode",
            "maximize",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analyze_cli.main() == 0
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "aggregate_summary.json",
        "report.md",
        "table.tex",
        "trajectory.csv",
    ]
    stdout = capsys.readouterr().out
    assert "summary=" in stdout
    assert "trajectory=" in stdout
