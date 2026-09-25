from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _study_trace(tmp_path: Path, *, proposal_cap: int = 60, unique_target: int = 30):
    from experiments.LibContentBench.opentuner_integration.trace import (
        OpenTunerStudyTrace,
    )

    return OpenTunerStudyTrace(
        tmp_path / "case_study.jsonl",
        proposal_cap=proposal_cap,
        unique_execution_target=unique_target,
    )


def _opentuner_interface(tmp_path: Path, *, metric_direction: str = "minimize"):
    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )
    from experiments.LibContentBench.opentuner_integration.specs import load_workflow_spec

    args = SimpleNamespace(parallel_compile=False, parallelism=1)
    return LibContentBenchOpenTunerInterface(
        args,
        workflow_spec=load_workflow_spec("nestedcpu-blas"),
        metric_name="wall_time_sec",
        metric_direction=metric_direction,
        timeout_seconds=30,
        output_dir=tmp_path,
        repo_root=tmp_path,
    )


def _desired_result(interface, config: dict[str, object], *, desired_result_id: int = 11):
    configuration = SimpleNamespace(
        id=7,
        hash="opentuner-hash",
        data=config,
    )
    return SimpleNamespace(
        id=desired_result_id,
        generation=3,
        requestor="UniformGreedyMutation",
        configuration_id=7,
        configuration=configuration,
    )


@pytest.mark.parametrize("abort_run", [False, True])
def test_opentuner_cli_finalizes_trace_with_identity_captured_before_session_close(
    monkeypatch, tmp_path, abort_run
) -> None:
    import opentuner.tuningrunmain
    from sqlalchemy.orm.exc import DetachedInstanceError

    from experiments.LibContentBench.opentuner_integration import cli, trace

    instances = []
    finalized_run_ids = []

    class DetachedTuningRun:
        def __init__(self, session):
            self._session = session
            self._id = 42

        @property
        def id(self):
            if self._session.closed:
                raise DetachedInstanceError("tuning run is detached")
            return self._id

    class ClosingSession:
        def __init__(self):
            self.closed = False
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1

        def close(self):
            self.closed = True

    class ClosingTuningRunMain:
        def __init__(self, interface, args):
            self.interface = interface
            self.args = args
            self.session = ClosingSession()
            self.tuning_run = None
            instances.append(self)

        def init(self):
            if self.tuning_run is None:
                self.tuning_run = DetachedTuningRun(self.session)

        def main(self):
            self.init()
            self.session.close()
            if abort_run:
                raise RuntimeError("simulated aborted run")

    def capture_finalization(**kwargs):
        assert instances[0].session.closed is True
        finalized_run_ids.append(kwargs["tuning_run_id"])

    monkeypatch.setattr(opentuner.tuningrunmain, "TuningRunMain", ClosingTuningRunMain)
    monkeypatch.setattr(trace, "finalize_opentuner_trace", capture_finalization)

    argv = [
        "--workflow",
        "nestedcpu-blas",
        "--output-dir",
        str(tmp_path),
        "--test-limit",
        "1",
    ]
    if abort_run:
        with pytest.raises(RuntimeError, match="simulated aborted run"):
            cli.main(argv)
    else:
        assert cli.main(argv) == 0

    assert instances[0].session.flush_count == 1
    assert finalized_run_ids == [42]
    with pytest.raises(DetachedInstanceError):
        _ = instances[0].tuning_run.id


def test_opentuner_success_preserves_objective_conversion_and_request_metadata(
    monkeypatch, tmp_path
) -> None:
    interface = _opentuner_interface(tmp_path)
    desired_result = _desired_result(interface, dict(interface.baseline_config))
    monkeypatch.setattr(
        interface,
        "call_program",
        lambda *args, **kwargs: {
            "stdout": '{"wall_time_sec":12.5}\n',
            "stderr": "",
            "returncode": 0,
            "timeout": False,
            "time": 12.5,
        },
    )

    result = interface.run(desired_result, None, 30)
    artifact = json.loads(
        (tmp_path / "trials/trial_0001/trial_artifact.json").read_text(encoding="utf-8")
    )

    result_value = cast(Any, result)
    assert result_value.state == "OK"
    assert result_value.accuracy == -12.5
    assert math.isfinite(result_value.time)
    assert artifact["desired_result_id"] == 11
    assert artifact["desired_result_generation"] == 3
    assert artifact["desired_result_requestor"] == "UniformGreedyMutation"
    assert artifact["configuration_id"] == 7
    assert artifact["configuration_hash"] == "opentuner-hash"
    assert artifact["is_baseline"] is True


def test_opentuner_nonbaseline_failure_is_finite_and_dominated(monkeypatch, tmp_path) -> None:
    from experiments.LibContentBench.opentuner_integration.harness import (
        DOMINATED_ACCURACY,
        DOMINATED_TIME,
    )

    interface = _opentuner_interface(tmp_path)
    failed_config = dict(interface.baseline_config)
    failed_config["--joblib-n-jobs"] = int(failed_config["--joblib-n-jobs"]) - 1
    desired_result = _desired_result(interface, failed_config)
    monkeypatch.setattr(
        interface,
        "call_program",
        lambda *args, **kwargs: {
            "stdout": "",
            "stderr": "failed",
            "returncode": 2,
            "timeout": False,
            "time": 0.1,
        },
    )

    result = interface.run(desired_result, None, 30)
    artifact = json.loads(
        (tmp_path / "trials/trial_0001/trial_artifact.json").read_text(encoding="utf-8")
    )

    result_value = cast(Any, result)
    assert result_value.state == "ERROR"
    assert result_value.accuracy == DOMINATED_ACCURACY
    assert result_value.time == DOMINATED_TIME
    assert math.isfinite(result_value.accuracy)
    assert math.isfinite(result_value.time)
    assert artifact["result_state"] == "ERROR"
    assert artifact["result_accuracy"] == DOMINATED_ACCURACY
    assert artifact["result_time"] == DOMINATED_TIME


@pytest.mark.parametrize("metric_literal", ["NaN", "Infinity", "-Infinity"])
def test_opentuner_nonfinite_metric_is_no_metric_failure(
    monkeypatch, tmp_path, metric_literal
) -> None:
    from experiments.LibContentBench.opentuner_integration.harness import (
        DOMINATED_ACCURACY,
        DOMINATED_TIME,
    )

    interface = _opentuner_interface(tmp_path)
    failed_config = dict(interface.baseline_config)
    failed_config["--joblib-n-jobs"] = int(failed_config["--joblib-n-jobs"]) - 1
    desired_result = _desired_result(interface, failed_config)
    monkeypatch.setattr(
        interface,
        "call_program",
        lambda *args, **kwargs: {
            "stdout": f'{{"wall_time_sec":{metric_literal}}}\n',
            "stderr": "",
            "returncode": 0,
            "timeout": False,
            "time": 0.1,
        },
    )

    result = cast(Any, interface.run(desired_result, None, 30))
    artifact = json.loads(
        (tmp_path / "trials/trial_0001/trial_artifact.json").read_text(encoding="utf-8")
    )

    assert result.state == "ERROR"
    assert result.accuracy == DOMINATED_ACCURACY
    assert result.time == DOMINATED_TIME
    assert artifact["result_state"] == "NO_METRIC"
    assert artifact["metric_value"] is None


@pytest.mark.parametrize(
    ("completed", "expected_state"),
    [
        (
            {"stdout": "", "stderr": "timeout", "returncode": -9, "timeout": True, "time": 30},
            "TIMEOUT",
        ),
        (
            {"stdout": "", "stderr": "error", "returncode": 1, "timeout": False, "time": 1},
            "ERROR",
        ),
        (
            {"stdout": "not-json", "stderr": "", "returncode": 0, "timeout": False, "time": 1},
            "NO_METRIC",
        ),
    ],
)
def test_opentuner_baseline_failure_writes_evidence_and_aborts(
    monkeypatch, tmp_path, completed, expected_state
) -> None:
    from experiments.LibContentBench.opentuner_integration.harness import BaselineTrialFailed

    interface = _opentuner_interface(tmp_path)
    desired_result = _desired_result(interface, dict(interface.baseline_config))
    monkeypatch.setattr(interface, "call_program", lambda *args, **kwargs: completed)

    with pytest.raises(BaselineTrialFailed, match=f"state={expected_state}"):
        interface.run(desired_result, None, 30)

    artifact_path = tmp_path / "trials/trial_0001/trial_artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["result_state"] == expected_state
    assert artifact["is_baseline"] is True
    assert math.isfinite(artifact["result_accuracy"])
    assert math.isfinite(artifact["result_time"])


def test_opentuner_trace_rows_are_stable_and_link_reused_duplicates() -> None:
    from experiments.LibContentBench.opentuner_integration.trace import build_trace_rows

    configuration = SimpleNamespace(id=5, hash="db-hash", data={"threads": 4})
    result = SimpleNamespace(
        id=9,
        state="OK",
        accuracy=17.0,
        time=2.0,
        was_new_best=True,
    )
    requests = [
        SimpleNamespace(
            id=20,
            configuration_id=5,
            configuration=configuration,
            result=result,
            generation=1,
            requestor="UniformGreedyMutation",
            state="COMPLETE",
        ),
        SimpleNamespace(
            id=21,
            configuration_id=5,
            configuration=configuration,
            result=result,
            generation=2,
            requestor="DifferentialEvolutionAlt",
            state="COMPLETE",
        ),
    ]
    artifacts = {
        20: {
            "trial_index": 1,
            "artifact_path": "trials/trial_0001/trial_artifact.json",
            "result_state": "OK",
            "result_accuracy": 17.0,
            "result_time": 2.0,
            "trial_wall_time_sec": 2.0,
        }
    }

    first = build_trace_rows(requests, trial_artifacts=artifacts, optimizer_version="0.8.8")
    second = build_trace_rows(
        list(reversed(requests)),
        trial_artifacts=artifacts,
        optimizer_version="0.8.8",
    )

    assert first == second
    assert [row["requestor"] for row in first] == [
        "UniformGreedyMutation",
        "DifferentialEvolutionAlt",
    ]
    assert first[0]["executed"] is True
    assert first[0]["execution_index"] == 1
    assert first[0]["was_new_best"] is True
    assert first[1]["duplicate_of_request_index"] == 1
    assert first[1]["executed"] is False
    assert first[1]["reused_result"] is True
    assert first[1]["result_id"] == 9
    assert first[1]["was_new_best"] is False


def test_opentuner_repository_best_selection_ignores_failed_trials() -> None:
    from experiments.LibContentBench.opentuner_integration.cli import _is_better_trial
    from experiments.LibContentBench.opentuner_integration.harness import (
        select_best_trial_artifact,
    )

    incumbent = {
        "result_state": "OK",
        "metric_value": 4.0,
        "metric_direction": "maximize",
    }
    failed = {
        "result_state": "ERROR",
        "metric_value": 1000.0,
        "metric_direction": "maximize",
    }
    better = {
        "result_state": "OK",
        "metric_value": 5.0,
        "metric_direction": "maximize",
    }

    assert _is_better_trial(failed, incumbent) is False
    assert _is_better_trial(better, incumbent) is True
    assert select_best_trial_artifact([incumbent, failed, better], "maximize") is better


def test_opentuner_exact_version_preflight() -> None:
    from experiments.LibContentBench.opentuner_integration.trace import require_opentuner_0_8_8

    assert require_opentuner_0_8_8("0.8.8") == "0.8.8"
    with pytest.raises(RuntimeError, match="requires opentuner==0.8.8"):
        require_opentuner_0_8_8("0.8.7")


def test_opentuner_version_preflight_reads_distribution_metadata(monkeypatch) -> None:
    from experiments.LibContentBench.opentuner_integration import trace

    requested = []

    def fake_version(distribution_name):
        requested.append(distribution_name)
        return "0.8.8"

    monkeypatch.setattr(trace.importlib.metadata, "version", fake_version)

    assert trace.require_opentuner_0_8_8() == "0.8.8"
    assert requested == ["opentuner"]


def test_opentuner_study_is_disabled_by_default_and_overrides_require_opt_in() -> None:
    from experiments.LibContentBench.opentuner_integration.cli import (
        _validate_args,
        build_parser,
    )

    parser = build_parser()
    defaults = parser.parse_args(
        ["--workflow", "nestedcpu-blas", "--output-dir", "/tmp/opentuner-disabled"]
    )
    assert defaults.study_ledger is None
    assert defaults.study_proposal_cap is None
    assert defaults.study_unique_execution_target is None

    invalid = parser.parse_args(
        [
            "--workflow",
            "nestedcpu-blas",
            "--output-dir",
            "/tmp/opentuner-disabled",
            "--study-proposal-cap",
            "2",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(parser, invalid)

    pipelined = parser.parse_args(
        [
            "--workflow",
            "nestedcpu-blas",
            "--output-dir",
            "/tmp/opentuner-disabled",
            "--study-ledger",
            "/tmp/study.jsonl",
            "--pipelining",
            "1",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(parser, pipelined)

    extra_seed = parser.parse_args(
        [
            "--workflow",
            "nestedcpu-blas",
            "--output-dir",
            "/tmp/opentuner-disabled",
            "--study-ledger",
            "/tmp/study.jsonl",
            "--seed-configuration",
            "/tmp/seed.json",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(parser, extra_seed)


def test_opentuner_study_convergence_excludes_baseline_and_counts_failed_unique(
    monkeypatch, tmp_path
) -> None:
    study = _study_trace(tmp_path, proposal_cap=4, unique_target=1)
    interface = _opentuner_interface(tmp_path / "run")
    interface.study_trace = study
    interface.baseline_fingerprint = study.fingerprint(interface.baseline_config)
    setattr(interface, "extra_convergence_criteria", interface._study_convergence_criteria)

    monkeypatch.setattr(
        interface,
        "call_program",
        lambda *args, **kwargs: {
            "stdout": '{"wall_time_sec":10.0}\n',
            "stderr": "",
            "returncode": 0,
            "timeout": False,
            "time": 10.0,
        },
    )
    interface.run(_desired_result(interface, dict(interface.baseline_config)), None, 30)
    assert interface.study_unique_execution_count == 0
    assert interface.extra_convergence_criteria([]) is False

    failed_config = dict(interface.baseline_config)
    failed_config["--joblib-n-jobs"] = 2
    monkeypatch.setattr(
        interface,
        "call_program",
        lambda *args, **kwargs: {
            "stdout": "",
            "stderr": "failed",
            "returncode": 2,
            "timeout": False,
            "time": 0.1,
        },
    )
    interface.run(_desired_result(interface, failed_config, desired_result_id=12), None, 30)
    assert interface.study_unique_execution_count == 1
    assert interface.extra_convergence_criteria([]) is True


def _native_row(
    request_index: int,
    config: dict[str, Any],
    *,
    executed: bool,
    state: str | None,
    requestor: str | None,
    reused: bool = False,
    execution_index: int | None = None,
    metric_value: float | None = None,
) -> dict[str, object]:
    return {
        "request_index": request_index,
        "desired_result_id": 100 + request_index,
        "generation": request_index - 1,
        "requestor": requestor,
        "requestor_unavailable_reason": (
            None if requestor is not None else "not_recorded_by_opentuner_orm"
        ),
        "config": config,
        "executed": executed,
        "execution_index": execution_index,
        "trial_artifact": (
            f"trials/trial_{execution_index:04d}/trial_artifact.json"
            if execution_index is not None
            else None
        ),
        "trial_wall_time_sec": 0.1 if executed else None,
        "metric_value": metric_value,
        "result_state": state,
        "reused_result": reused,
    }


def test_opentuner_study_ledger_preserves_request_order_reuse_and_failures(tmp_path) -> None:
    import importlib

    study = _study_trace(tmp_path, proposal_cap=5, unique_target=2)
    baseline = {
        "--joblib-n-jobs": 6,
        "OMP_NUM_THREADS": 6,
        "OPENBLAS_NUM_THREADS": 6,
        "MKL_NUM_THREADS": 1,
        "BLIS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
        "--backend-prefer": "processes",
    }
    first = {**baseline, "--joblib-n-jobs": 2}
    failed = {**baseline, "--joblib-n-jobs": 3}
    unlaunched = {**baseline, "--joblib-n-jobs": 4}
    rows = [
        _native_row(
            1,
            baseline,
            executed=True,
            state="OK",
            requestor="seed",
            execution_index=1,
            metric_value=10.0,
        ),
        _native_row(
            2,
            first,
            executed=True,
            state="OK",
            requestor="UniformGreedyMutation",
            execution_index=2,
            metric_value=12.0,
        ),
        _native_row(
            3,
            first,
            executed=False,
            state="OK",
            requestor="DifferentialEvolutionAlt",
            reused=True,
            metric_value=None,
        ),
        _native_row(
            4,
            failed,
            executed=True,
            state="ERROR",
            requestor=None,
            execution_index=3,
        ),
        _native_row(
            5,
            unlaunched,
            executed=False,
            state=None,
            requestor="RandomNelderMead",
        ),
    ]

    summary = study.write_from_native_rows(
        rows,
        metric_name="wall_time_sec",
        metric_mode="minimize",
        optimizer_stop_reason="opentuner_completed",
    )
    case_study = importlib.import_module(
        "experiments.LibContentBench.nestedcpu-blas.case_study"
    )
    events = case_study.read_events(study.path)
    proposals = [event for event in events if event["event_type"] == "proposal.recorded"]
    validations = [event for event in events if event["event_type"] == "proposal.validated"]
    finishes = [event for event in events if event["event_type"] == "execution.finished"]

    assert [event["metadata"]["request_index"] for event in proposals] == [1, 2, 3, 4, 5]
    assert [event["metadata"]["requestor"] for event in proposals] == [
        "seed",
        "UniformGreedyMutation",
        "DifferentialEvolutionAlt",
        None,
        "RandomNelderMead",
    ]
    assert [event["disposition"] for event in validations] == [
        "duplicate",
        "candidate",
        "duplicate",
        "candidate",
        "rejected",
    ]
    assert validations[2]["duplicate_of_proposal_sequence"] == proposals[1]["sequence"]
    assert [event["status"] for event in finishes] == ["ok", "error"]
    assert summary["unique_execution_count"] == 2
    assert summary["baseline_recorded"] is True
    assert summary["terminal_reason"] == "unique_execution_target_reached"


def test_opentuner_study_proposal_cap_writes_shortfall_and_baseline_failure(
    tmp_path,
) -> None:
    import importlib

    baseline = {
        "--joblib-n-jobs": 6,
        "OMP_NUM_THREADS": 6,
        "OPENBLAS_NUM_THREADS": 6,
        "MKL_NUM_THREADS": 1,
        "BLIS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
        "--backend-prefer": "processes",
    }
    candidate = {**baseline, "--joblib-n-jobs": 2}
    study = _study_trace(tmp_path, proposal_cap=3, unique_target=2)
    rows = [
        _native_row(
            1,
            baseline,
            executed=True,
            state="ERROR",
            requestor="seed",
            execution_index=1,
        ),
        _native_row(
            2,
            candidate,
            executed=True,
            state="TIMEOUT",
            requestor="UniformGreedyMutation",
            execution_index=2,
        ),
        _native_row(
            3,
            candidate,
            executed=False,
            state="TIMEOUT",
            requestor="UniformGreedyMutation",
            reused=True,
        ),
    ]

    summary = study.write_from_native_rows(
        rows,
        metric_name="wall_time_sec",
        metric_mode="minimize",
        optimizer_stop_reason="opentuner_aborted",
    )
    case_study = importlib.import_module(
        "experiments.LibContentBench.nestedcpu-blas.case_study"
    )
    terminal = case_study.read_events(study.path)[-1]

    assert summary["baseline_status"] == "error"
    assert summary["unique_execution_count"] == 1
    assert terminal["reason"] == "proposal_cap_reached_with_shortfall"
    assert terminal["metadata"]["shortfall"] == 1


def test_opentuner_partial_orm_trace_finalization_writes_all_artifacts(tmp_path) -> None:
    from opentuner import resultsdb
    from opentuner.resultsdb import models

    from experiments.LibContentBench.libcontentbench.common import write_json
    from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint
    from experiments.LibContentBench.opentuner_integration.trace import finalize_opentuner_trace

    database_url = f"sqlite:///{tmp_path / 'opentuner.db'}"
    _, session_factory = resultsdb.connect(database_url)
    session = session_factory()
    tuning_run = cast(Any, models.TuningRun)(
        uuid="partial-run", name="partial", state="ABORTED"
    )
    configuration = cast(Any, models.Configuration)(hash="db-hash", data={"threads": 4})
    session.add_all([tuning_run, configuration])
    session.flush()
    request = cast(Any, models.DesiredResult)(
        tuning_run=tuning_run,
        configuration=configuration,
        generation=0,
        requestor="seed",
        state="RUNNING",
    )
    session.add(request)
    session.commit()
    tuning_run_id = int(cast(Any, tuning_run.id))
    request_id = int(cast(Any, request.id))
    session.close()

    artifact_path = tmp_path / "trials/trial_0001/trial_artifact.json"
    write_json(
        artifact_path,
        {
            "trial_index": 1,
            "desired_result_id": request_id,
            "result_state": "ERROR",
            "result_accuracy": -sys.float_info.max,
            "result_time": sys.float_info.max,
            "trial_wall_time_sec": 0.1,
        },
    )
    baseline_fingerprint = config_fingerprint({"threads": 4})

    finalize_opentuner_trace(
        database_url=database_url,
        tuning_run_id=tuning_run_id,
        output_dir=tmp_path,
        baseline_fingerprint=baseline_fingerprint,
        configured_techniques=None,
    )

    trace_lines = (tmp_path / "optimizer_trace.jsonl").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((tmp_path / "trace_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "trace_summary.json").read_text(encoding="utf-8"))
    row = json.loads(trace_lines[0])

    assert len(trace_lines) == 1
    assert row["request_state"] == "RUNNING"
    assert row["executed"] is True
    assert row["result_id"] is None
    assert row["result_state"] == "ERROR"
    assert manifest["default_technique"] == "AUCBanditMetaTechniqueA"
    assert manifest["tuning_run_state"] == "ABORTED"
    assert summary["requested_results"] == 1
    assert summary["executed_results"] == 1
    assert summary["incomplete_requests"] == 1


def test_opentuner_incompatible_sqlite_schema_raises_trace_availability_error(
    tmp_path,
) -> None:
    from experiments.LibContentBench.opentuner_integration.trace import (
        TraceAvailabilityError,
        finalize_opentuner_trace,
    )

    database_path = tmp_path / "outdated.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE program (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(TraceAvailabilityError, match="trace extraction failed"):
        finalize_opentuner_trace(
            database_url=f"sqlite:///{database_path}",
            tuning_run_id=1,
            output_dir=tmp_path,
            baseline_fingerprint="baseline",
            configured_techniques=None,
        )
