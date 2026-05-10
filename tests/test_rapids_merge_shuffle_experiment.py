from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

import pytest


class ComparisonPair(TypedDict):
    pair_id: str
    baseline_case_id: str
    contender_case_id: str


class AcceptanceConfig(TypedDict):
    comparison_pairs: list[ComparisonPair]


class CaseConfig(TypedDict):
    case_id: str
    rows_left: int
    shuffle_rounds: int
    env: dict[str, str]


class ComparisonReportEntry(TypedDict):
    pair_id: str
    accepted: bool
    speedup_ratio: float


class ComparisonReport(TypedDict):
    comparison_count: int
    comparisons: list[ComparisonReportEntry]


class AcceptanceReport(TypedDict):
    passed: bool
    accepted_pairs: list[str]
    failed_cases: list[str]


class MatrixConfig(TypedDict):
    experiment_name: str
    acceptance: AcceptanceConfig
    cases: list[CaseConfig]


class MatrixCaseResult(TypedDict, total=False):
    case_id: str
    exit_code: int
    timed_out: bool


class MatrixModule(Protocol):
    def load_matrix(self, path: Path) -> MatrixConfig: ...

    def build_case_command(
        self,
        python_executable: str,
        worker_script: Path,
        case: CaseConfig,
        output_json: Path,
    ) -> list[str]: ...

    def execute_matrix(
        self,
        matrix_path: Path,
        runs_root: Path,
        python_executable: str,
        run_label: str | None,
    ) -> tuple[Path, list[MatrixCaseResult]]: ...


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError("expected JSON object")
    return {str(key): item for key, item in value.items()}


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError("expected JSON list")
    return list(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise AssertionError("expected integer-compatible value")
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        raise AssertionError("expected float-compatible value")
    return float(value)


def _workspace_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "RAPIDS"
        / "rapids-dask-cudf-merge-shuffle"
    )


def _load_module() -> MatrixModule:
    path = _workspace_dir() / "run_matrix.py"
    spec = importlib.util.spec_from_file_location("rapids_merge_shuffle_run_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(MatrixModule, cast(object, module))


def test_load_matrix_merges_defaults_and_acceptance_pairs() -> None:
    module = _load_module()

    matrix = module.load_matrix(_workspace_dir() / "matrix.yaml")

    assert matrix["experiment_name"] == "rapids_dask_cudf_merge_shuffle_p100"
    first_case = matrix["cases"][0]
    assert first_case["case_id"] == "single_worker_bad_pressure"
    assert first_case["rows_left"] == 2400000
    assert first_case["shuffle_rounds"] == 2
    assert first_case["env"] == {"CUDA_VISIBLE_DEVICES": "0"}

    pair = matrix["acceptance"]["comparison_pairs"][0]
    assert pair["pair_id"] == "single_worker_vs_dual_worker"
    assert pair["baseline_case_id"] == "single_worker_bad_pressure"
    assert pair["contender_case_id"] == "dual_worker_spill_enabled"


def test_build_case_command_includes_merge_shuffle_flags() -> None:
    module = _load_module()
    matrix = module.load_matrix(_workspace_dir() / "matrix.yaml")
    case = matrix["cases"][2]
    output_json = Path("/tmp/case/result.json")
    worker_script = _workspace_dir() / "worker_case.py"

    command = module.build_case_command(
        python_executable="python3",
        worker_script=worker_script,
        case=case,
        output_json=output_json,
    )

    assert command[:2] == ["python3", str(worker_script)]
    assert command[command.index("--case-id") + 1] == "quad_worker_pool8g"
    assert command[command.index("--n-workers") + 1] == "4"
    assert command[command.index("--shuffle-partitions") + 1] == "64"
    assert command[command.index("--protocol") + 1] == "tcp"
    assert command[command.index("--device-memory-limit") + 1] == "14GB"
    assert command[command.index("--rmm-pool-size") + 1] == "8GB"
    assert command[command.index("--output-json") + 1] == str(output_json)
    assert "--scaffold-only" in command


def test_execute_matrix_writes_summary_and_acceptance_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    workspace = _workspace_dir()
    source_matrix = module.load_matrix(workspace / "matrix_smoke.yaml")
    matrix_path = tmp_path / "matrix.yaml"
    _ = matrix_path.write_text(
        json.dumps(source_matrix, indent=2) + "\n",
        encoding="utf-8",
    )
    worker_copy = tmp_path / "worker_case.py"
    _ = worker_copy.write_text(
        (workspace / "worker_case.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    class FakeRunCall(TypedDict):
        command: list[str]
        cwd: Path
        env: dict[str, str]
        case_id: str

    calls: list[FakeRunCall] = []

    def fake_run(
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout == 60
        assert cwd == tmp_path

        output_json = Path(command[command.index("--output-json") + 1])
        case_id = command[command.index("--case-id") + 1]
        workers = int(command[command.index("--n-workers") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        _ = output_json.write_text(
            json.dumps(
                {
                    "schema_version": "rapids_merge_shuffle_result_v1",
                    "case_id": case_id,
                    "status": "scaffold",
                    "implementation_status": "pending_live_gpu_merge_shuffle_workload",
                    "scaffold_only": True,
                    "elapsed_s": 2.0 if workers == 1 else 1.0,
                    "cluster_start_s": 0.1,
                    "shuffle_elapsed_s": 0.8,
                    "merge_elapsed_s": 0.7,
                    "worker_count": workers,
                    "visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                    "protocol": command[command.index("--protocol") + 1],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        calls.append({"command": command, "cwd": cwd, "env": env, "case_id": case_id})
        return subprocess.CompletedProcess(command, 0, stdout=f"ok:{case_id}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    summary_path, rows = module.execute_matrix(
        matrix_path=matrix_path,
        runs_root=tmp_path / "runs",
        python_executable="python3",
        run_label="smoke-fixed",
    )

    assert len(calls) == 2
    assert calls[0]["case_id"] != calls[1]["case_id"]
    assert calls[0]["env"]["PYTHONNOUSERSITE"] == "1"
    assert calls[1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"

    run_dir = tmp_path / "runs" / "smoke-fixed"
    assert summary_path == run_dir / "summary.csv"
    assert summary_path.exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "acceptance_report.json").exists()
    assert len(rows) == 2

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert [row["case_id"] for row in summary_rows] == [
        "single_worker_small",
        "dual_worker_small",
    ]
    assert [row["exit_code"] for row in summary_rows] == ["0", "0"]
    assert summary_rows[0]["status"] == "scaffold"

    comparison_report_obj = cast(
        object,
        json.loads(
        (run_dir / "comparison_report.json").read_text(encoding="utf-8")
        ),
    )
    comparison_report_data = _as_dict(comparison_report_obj)
    comparison_report: ComparisonReport = {
        "comparison_count": _as_int(comparison_report_data["comparison_count"]),
        "comparisons": [
            {
                "pair_id": str(entry["pair_id"]),
                "accepted": bool(entry["accepted"]),
                "speedup_ratio": _as_float(entry["speedup_ratio"]),
            }
            for entry_obj in _as_list(comparison_report_data["comparisons"])
            for entry in [_as_dict(entry_obj)]
        ],
    }
    assert comparison_report["comparison_count"] == 1
    comparison = comparison_report["comparisons"][0]
    assert comparison["pair_id"] == "smoke_bad_vs_good"
    assert comparison["accepted"] is True
    assert pytest.approx(comparison["speedup_ratio"], rel=1e-6) == 2.0

    acceptance_report_obj = cast(
        object,
        json.loads(
        (run_dir / "acceptance_report.json").read_text(encoding="utf-8")
        ),
    )
    acceptance_report_data = _as_dict(acceptance_report_obj)
    acceptance_report: AcceptanceReport = {
        "passed": bool(acceptance_report_data["passed"]),
        "accepted_pairs": [
            str(value) for value in _as_list(acceptance_report_data["accepted_pairs"])
        ],
        "failed_cases": [
            str(value) for value in _as_list(acceptance_report_data["failed_cases"])
        ],
    }
    assert acceptance_report["passed"] is True
    assert acceptance_report["accepted_pairs"] == ["smoke_bad_vs_good"]
    assert acceptance_report["failed_cases"] == []
