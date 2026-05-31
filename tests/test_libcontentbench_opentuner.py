from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.LibContentBench.opentuner_integration.specs import (
    load_workflow_spec,
    supported_workflow_ids,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_supported_workflow_set_is_exactly_five() -> None:
    assert supported_workflow_ids() == (
        "nestedcpu-blas",
        "arrow-dask-duckdb-memspill",
        "hf-datasets-tokenizers-dataloader",
        "torchdata-fsspec-s3-ddp",
        "dali-nvjpeg-pytorch-cudnn",
    )


def test_static_spec_module_does_not_import_opentuner() -> None:
    for name in [
        name for name in sys.modules if name == "opentuner" or name.startswith("opentuner.")
    ]:
        sys.modules.pop(name)

    module = importlib.import_module("experiments.LibContentBench.opentuner_integration.specs")
    assert isinstance(module, ModuleType)
    assert not any(name == "opentuner" or name.startswith("opentuner.") for name in sys.modules)


def test_every_exposed_tunable_is_whitelist_authorized() -> None:
    for workflow_id in supported_workflow_ids():
        workflow_spec = load_workflow_spec(workflow_id)
        whitelist = set(workflow_spec.load_whitelist())
        for tunable in workflow_spec.tunables:
            assert any(name in whitelist for name in tunable.whitelist_names())


def test_workload_shape_flags_are_not_exposed_in_parameter_space() -> None:
    forbidden = {
        "nestedcpu-blas": {"--matrix-size", "--task-count", "--repetitions"},
        "arrow-dask-duckdb-memspill": {"--num-files", "--rows-per-file", "--row-group-size"},
        "hf-datasets-tokenizers-dataloader": {
            "--dataset-rows",
            "--tokens-per-row",
            "--max-length",
            "--tokenizer-vocab-size",
            "--hidden-dim",
            "--num-classes",
            "--batch-size",
            "--epochs",
            "--learning-rate",
            "--tokenize-batch-size",
        },
        "torchdata-fsspec-s3-ddp": {
            "--world-size",
            "--samples-per-rank",
            "--shards-per-rank",
            "--input-dim",
            "--hidden-dim",
            "--num-classes",
            "--batch-size",
            "--epochs",
            "--learning-rate",
        },
        "dali-nvjpeg-pytorch-cudnn": {
            "--timeout-seconds",
            "--warmup-seconds",
            "--measure-seconds",
            "--dataset-images",
            "--image-min-size",
            "--image-max-size",
            "--jpeg-quality",
            "--num-classes",
            "--resize-size",
            "--crop-size",
            "--dali-output-dtype",
            "--dali-decode-backend",
            "--torch-model",
            "--torch-precision",
        },
    }

    for workflow_id, disallowed in forbidden.items():
        parameter_names = {
            tunable.parameter_name for tunable in load_workflow_spec(workflow_id).tunables
        }
        assert parameter_names.isdisjoint(disallowed)


def test_manipulator_parameter_names_match_expected_whitelisted_controls() -> None:
    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )

    expected = {
        "nestedcpu-blas": {
            "--joblib-n-jobs",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "--backend-prefer",
        },
        "arrow-dask-duckdb-memspill": {
            "BENCHMARK_N_WORKERS",
            "BENCHMARK_THREADS_PER_WORKER",
            "BENCHMARK_WORKER_MEMORY_LIMIT",
            "BENCHMARK_ARROW_CPU_COUNT",
            "BENCHMARK_ARROW_IO_THREADS",
            "BENCHMARK_DUCKDB_MEMORY_LIMIT",
            "BENCHMARK_DUCKDB_THREADS",
            "BENCHMARK_PATH_LAYOUT",
        },
        "hf-datasets-tokenizers-dataloader": {
            "BENCHMARK_TORCH_NUM_THREADS",
            "BENCHMARK_DATASETS_NUM_PROC",
            "BENCHMARK_TOKENIZERS_PARALLELISM",
            "BENCHMARK_NUM_WORKERS",
            "BENCHMARK_PREFETCH_FACTOR",
            "BENCHMARK_PERSISTENT_WORKERS",
        },
        "torchdata-fsspec-s3-ddp": {
            "BENCHMARK_TORCH_NUM_THREADS",
            "BENCHMARK_S3_BLOCK_SIZE",
            "BENCHMARK_S3_CACHE_TYPE",
            "BENCHMARK_S3_USE_LISTINGS_CACHE",
            "BENCHMARK_DDP_BUCKET_CAP_MB",
        },
        "dali-nvjpeg-pytorch-cudnn": {
            "BENCHMARK_DALI_BATCH_SIZE",
            "BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH",
            "BENCHMARK_DALI_HW_DECODER_LOAD",
            "BENCHMARK_DALI_NUM_THREADS",
            "BENCHMARK_DALI_MPS_SHARE",
            "BENCHMARK_TORCH_BATCH_SIZE",
            "BENCHMARK_TORCH_MPS_SHARE",
        },
    }

    args = SimpleNamespace(parallel_compile=False, parallelism=1)
    for workflow_id, parameter_names in expected.items():
        interface = LibContentBenchOpenTunerInterface(
            args,
            workflow_spec=load_workflow_spec(workflow_id),
            metric_name="wall_time_sec",
            metric_direction="minimize",
            timeout_seconds=60,
            output_dir=Path("/tmp") / f"lcb-opentuner-{workflow_id}",
            repo_root=_repo_root(),
        )
        manipulator = interface.manipulator()
        names = {parameter.name for parameter in manipulator.params}
        assert names == parameter_names


def test_harness_returns_result_time_from_metric_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )

    args = SimpleNamespace(parallel_compile=False, parallelism=1)
    interface = LibContentBenchOpenTunerInterface(
        args,
        workflow_spec=load_workflow_spec("nestedcpu-blas"),
        metric_name="wall_time_sec",
        metric_direction="minimize",
        timeout_seconds=60,
        output_dir=tmp_path,
        repo_root=_repo_root(),
    )

    def fake_call_program(command: str, limit: int, cwd: str) -> dict[str, object]:
        assert "run_nestedcpu_blas_case.py" in command
        assert f"--output-json {tmp_path}/trials/trial_0001/output.json" in command
        assert limit == 60
        assert cwd == str(_repo_root())
        return {
            "returncode": 0,
            "stdout": '{"wall_time_sec": 12.5, "throughput": 42.0}\n',
            "stderr": "",
            "timeout": False,
            "time": 0.01,
        }

    monkeypatch.setattr(interface, "call_program", fake_call_program)
    seed_config = interface.seed_configurations()[0]
    desired_result = SimpleNamespace(configuration=SimpleNamespace(data=seed_config))

    result = interface.run(desired_result, None, 0.0)

    assert result.time >= 0.0
    assert result.accuracy == -12.5
    artifact_path = tmp_path / "trials" / "trial_0001" / "trial_artifact.json"
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["metric_value"] == 12.5
    assert artifact["objective_value"] == -12.5
    assert artifact["result_state"] == "OK"
    assert artifact["trial_wall_time_sec"] >= 0.0
    assert artifact["trial_finished_run_wall_time_sec"] >= artifact["trial_wall_time_sec"]
    assert result.state == "OK"


def test_harness_maximizes_throughput_via_negative_objective(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )

    args = SimpleNamespace(parallel_compile=False, parallelism=1)
    interface = LibContentBenchOpenTunerInterface(
        args,
        workflow_spec=load_workflow_spec("nestedcpu-blas"),
        metric_name="throughput",
        metric_direction="maximize",
        timeout_seconds=60,
        output_dir=tmp_path,
        repo_root=_repo_root(),
    )

    def fake_call_program(command: str, limit: int, cwd: str) -> dict[str, object]:
        del command, limit, cwd
        return {
            "returncode": 0,
            "stdout": '{"wall_time_sec": 12.5, "throughput": 42.0}\n',
            "stderr": "",
            "timeout": False,
            "time": 0.02,
        }

    monkeypatch.setattr(interface, "call_program", fake_call_program)
    seed_config = interface.seed_configurations()[0]
    desired_result = SimpleNamespace(configuration=SimpleNamespace(data=seed_config))

    result = interface.run(desired_result, None, 0.0)

    output = capsys.readouterr().out
    assert "LibContentBench trial 1" in output
    assert "throughput=42.0" in output
    assert "goal=maximize" in output
    assert result.time >= 0.0
    assert result.accuracy == 42.0
    artifact_path = tmp_path / "trials" / "trial_0001" / "trial_artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["metric_value"] == 42.0
    assert artifact["objective_value"] == 42.0
    assert artifact["metric_direction"] == "maximize"
    assert artifact["trial_wall_time_sec"] >= 0.0
    assert artifact["trial_finished_run_wall_time_sec"] >= artifact["trial_wall_time_sec"]
    assert result.state == "OK"


def test_print_run_summary_reports_best_trial_and_wall_times(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from experiments.LibContentBench.opentuner_integration.cli import print_run_summary

    for trial_index, metric_value, finished_at in [(1, 10.0, 1.5), (2, 20.0, 3.0)]:
        trial_dir = tmp_path / "trials" / f"trial_{trial_index:04d}"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial_artifact.json").write_text(
            json.dumps(
                {
                    "trial_index": trial_index,
                    "result_state": "OK",
                    "metric_name": "throughput",
                    "metric_direction": "maximize",
                    "metric_value": metric_value,
                    "trial_finished_run_wall_time_sec": finished_at,
                }
            ),
            encoding="utf-8",
        )

    print_run_summary(tmp_path, total_process_wall_time_sec=3.5)

    output = capsys.readouterr().out
    assert "trials_completed: 2" in output
    assert "trial_metrics:" in output
    assert "trial 1: state=OK, throughput=10.0" in output
    assert "trial 2: state=OK, throughput=20.0" in output
    assert "best_trial: 2" in output
    assert "best_metric_value: 20.0" in output
    assert "wall_time_until_best_trial_finished_sec: 3.0" in output
    assert "total_process_wall_time_sec: 3.5" in output


def test_nestedcpu_seed_matches_documented_medium_env() -> None:
    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )

    args = SimpleNamespace(parallel_compile=False, parallelism=1)
    interface = LibContentBenchOpenTunerInterface(
        args,
        workflow_spec=load_workflow_spec("nestedcpu-blas"),
        metric_name="wall_time_sec",
        metric_direction="minimize",
        timeout_seconds=60,
        output_dir=Path("/tmp") / "lcb-opentuner-nestedcpu-seed",
        repo_root=_repo_root(),
    )

    assert interface.seed_configurations()[0] == {
        "--joblib-n-jobs": 6,
        "OMP_NUM_THREADS": 6,
        "OPENBLAS_NUM_THREADS": 6,
        "MKL_NUM_THREADS": 1,
        "BLIS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
        "--backend-prefer": "processes",
    }


def test_torchdata_open_tuner_spec_uses_pinned_minio_image() -> None:
    spec = load_workflow_spec("torchdata-fsspec-s3-ddp")

    assert spec.fixed_workload_payload["minio_image"] != "quay.io/minio/minio:latest"
    assert "RELEASE." in str(spec.fixed_workload_payload["minio_image"])


def test_run_opentuner_main_passes_existing_interface_to_tuningrunmain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import opentuner.tuningrunmain as tuningrunmain_module

    from experiments.LibContentBench import run_opentuner
    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )

    captured: dict[str, object] = {}

    class FakeTuningRunMain:
        def __init__(self, measurement_interface: object, args: object) -> None:
            captured["interface"] = measurement_interface
            captured["args"] = args

        def main(self) -> None:
            captured["main_called"] = True

    monkeypatch.setattr(tuningrunmain_module, "TuningRunMain", FakeTuningRunMain)

    exit_code = run_opentuner.main(
        [
            "--workflow",
            "nestedcpu-blas",
            "--output-dir",
            str(tmp_path / "run"),
            "--metric",
            "wall_time_sec",
            "--test-limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["main_called"] is True
    interface = captured["interface"]
    assert isinstance(interface, LibContentBenchOpenTunerInterface)
    assert interface.workflow_spec.workflow_id == "nestedcpu-blas"
    assert interface.metric_direction == "minimize"
    args = captured["args"]
    assert getattr(args, "workflow") == "nestedcpu-blas"
    assert getattr(args, "output_dir") == str(tmp_path / "run")


def test_metric_direction_auto_resolves_throughput_to_maximize() -> None:
    from experiments.LibContentBench.opentuner_integration.cli import resolve_metric_direction

    assert resolve_metric_direction("throughput", "auto") == "maximize"
    assert resolve_metric_direction("samples_per_sec", "auto") == "maximize"
    assert resolve_metric_direction("wall_time_sec", "auto") == "minimize"
    assert resolve_metric_direction("p95_latency_ms", "auto") == "minimize"


def test_run_opentuner_main_passes_metric_direction_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import opentuner.tuningrunmain as tuningrunmain_module

    from experiments.LibContentBench import run_opentuner

    captured: dict[str, object] = {}

    class FakeTuningRunMain:
        def __init__(self, measurement_interface: object, args: object) -> None:
            captured["interface"] = measurement_interface
            captured["args"] = args

        def main(self) -> None:
            captured["main_called"] = True

    monkeypatch.setattr(tuningrunmain_module, "TuningRunMain", FakeTuningRunMain)

    exit_code = run_opentuner.main(
        [
            "--workflow",
            "nestedcpu-blas",
            "--output-dir",
            str(tmp_path / "run-max"),
            "--metric",
            "throughput",
            "--metric-direction",
            "auto",
            "--test-limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["main_called"] is True
    interface = captured["interface"]
    assert getattr(interface, "metric_direction") == "maximize"


def test_run_opentuner_main_passes_tuning_goal_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import opentuner.tuningrunmain as tuningrunmain_module

    from experiments.LibContentBench import run_opentuner

    captured: dict[str, object] = {}

    class FakeTuningRunMain:
        def __init__(self, measurement_interface: object, args: object) -> None:
            captured["interface"] = measurement_interface
            captured["args"] = args

        def main(self) -> None:
            captured["main_called"] = True

    monkeypatch.setattr(tuningrunmain_module, "TuningRunMain", FakeTuningRunMain)

    exit_code = run_opentuner.main(
        [
            "--workflow",
            "nestedcpu-blas",
            "--output-dir",
            str(tmp_path / "run-goal"),
            "--metric",
            "throughput",
            "--tuning-goal",
            "minimize",
            "--test-limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["main_called"] is True
    interface = captured["interface"]
    assert getattr(interface, "metric_direction") == "minimize"


def test_cli_help_and_invalid_workflow_under_agent_python(tmp_path: Path) -> None:
    python_executable = "/home/yinengy/miniconda3/envs/agent/bin/python"
    script_path = (
        _repo_root() / "experiments" / "LibContentBench" / "opentuner_integration" / "cli.py"
    )

    help_result = subprocess.run(
        [python_executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--workflow" in help_result.stdout
    assert "--test-limit" in help_result.stdout
    assert "--metric-direction" in help_result.stdout
    assert "--tuning-goal" in help_result.stdout

    invalid_result = subprocess.run(
        [
            python_executable,
            str(script_path),
            "--workflow",
            "not-a-workflow",
            "--output-dir",
            str(tmp_path / "invalid"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid_result.returncode != 0
    combined = invalid_result.stdout + invalid_result.stderr
    assert "Unsupported workflow" in combined


def test_compatibility_shim_lists_workflows_under_agent_python() -> None:
    python_executable = "/home/yinengy/miniconda3/envs/agent/bin/python"
    script_path = _repo_root() / "experiments" / "LibContentBench" / "run_opentuner.py"

    result = subprocess.run(
        [python_executable, str(script_path), "--list-workflows"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == list(supported_workflow_ids())
