from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import pytest


class MatrixModule(Protocol):
    def load_matrix(self, path: Path) -> dict[str, Any]: ...

    def build_case_command(
        self,
        python_executable: str,
        worker_script: Path,
        case: dict[str, Any],
        output_json: Path,
    ) -> list[str]: ...

    def execute_matrix(
        self,
        matrix_path: Path,
        runs_root: Path,
        python_executable: str,
        run_label: str | None,
    ) -> tuple[Path, list[dict[str, Any]]]: ...


def _workspace_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "XGBoost"
        / "sklearn-xgboost-contention"
    )


def _load_module() -> MatrixModule:
    path = _workspace_dir() / "run_matrix.py"
    spec = importlib.util.spec_from_file_location("xgb_sklearn_run_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(MatrixModule, cast(object, module))


def test_load_matrix_merges_defaults_and_preserves_case_env() -> None:
    module = _load_module()

    matrix = module.load_matrix(_workspace_dir() / "matrix.yaml")

    assert matrix["experiment_name"] == "demo1_sklearn_xgboost_contention"
    first_case = matrix["cases"][0]
    assert first_case["case_id"] == "serial_outer1_inner1"
    assert first_case["samples"] == 6000
    assert first_case["cv_folds"] == 4
    assert first_case["env"] == {"OMP_NUM_THREADS": "1"}


def test_build_case_command_includes_worker_flags() -> None:
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
    assert "--case-id" in command
    assert command[command.index("--case-id") + 1] == "outer1_inner24"
    assert command[command.index("--sklearn-n-jobs") + 1] == "1"
    assert command[command.index("--xgb-n-jobs") + 1] == "24"
    assert command[command.index("--output-json") + 1] == str(output_json)


def test_execute_matrix_uses_fresh_subprocesses_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    workspace = _workspace_dir()
    source_matrix = module.load_matrix(workspace / "matrix.yaml")
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        json.dumps(
            {
                "experiment_name": source_matrix["experiment_name"],
                "defaults": source_matrix["defaults"],
                "cases": source_matrix["cases"][:2],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    worker_copy = tmp_path / "worker_case.py"
    worker_copy.write_text(
        (workspace / "worker_case.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        assert cwd == tmp_path

        output_json = Path(command[command.index("--output-json") + 1])
        case_id = command[command.index("--case-id") + 1]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "elapsed_s": 1.25 if case_id.endswith("1") else 2.5,
                    "accuracy_mean": 0.9,
                    "accuracy_std": 0.01,
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
        run_label="fixed-run",
    )

    assert len(calls) == 2
    assert calls[0]["case_id"] != calls[1]["case_id"]
    assert calls[0]["env"]["OMP_NUM_THREADS"] == "1"
    assert calls[1]["env"]["OMP_NUM_THREADS"] == "1"

    assert summary_path == tmp_path / "runs" / "fixed-run" / "summary.csv"
    assert summary_path.exists()
    assert len(rows) == 2
    assert all(Path(row["output_dir"]).exists() for row in rows)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert [row["case_id"] for row in summary_rows] == [
        "serial_outer1_inner1",
        "outer4_inner1",
    ]
    assert [row["exit_code"] for row in summary_rows] == ["0", "0"]
