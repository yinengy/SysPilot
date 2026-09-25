from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from opentuner import ConfigurationManipulator, MeasurementInterface, Result
from opentuner.search.manipulator import EnumParameter, FloatParameter, IntegerParameter
from opentuner.search.objective import MaximizeAccuracy

from experiments.LibContentBench.libcontentbench.common import (
    load_json_object,
    sanitize_label,
    write_json,
)
from experiments.LibContentBench.libcontentbench.direct_commands import build_direct_command
from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint
from experiments.LibContentBench.opentuner_integration.specs import (
    OpenTunerParameterSpec,
    OpenTunerWorkflowSpec,
)

DOMINATED_ACCURACY = -sys.float_info.max
DOMINATED_TIME = sys.float_info.max


class BaselineTrialFailed(RuntimeError):  # noqa: N818
    def __init__(self, *, state: str, artifact_path: Path) -> None:
        self.state = state
        self.artifact_path = artifact_path
        super().__init__(
            f"OpenTuner baseline trial failed with state={state}; evidence: {artifact_path}"
        )


def _encode_parameter_value(tunable: OpenTunerParameterSpec, raw_value: Any) -> Any:
    if tunable.parameter_kind == "integer":
        return int(raw_value)
    if tunable.parameter_kind == "float":
        return float(raw_value)
    if tunable.value_type == "bool":
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return "true"
            if normalized in {"0", "false", "no", "off"}:
                return "false"
            raise ValueError(
                f"Invalid boolean seed value for {tunable.parameter_name}: {raw_value!r}"
            )
        return "true" if bool(raw_value) else "false"
    if tunable.value_type in {"int", "float"}:
        return str(raw_value)
    return str(raw_value)


def _coerce_parameter_value(tunable: OpenTunerParameterSpec, raw_value: Any) -> Any:
    if tunable.value_type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(f"Invalid boolean value for {tunable.parameter_name}: {raw_value!r}")
    if tunable.value_type == "float":
        return float(raw_value)
    if tunable.value_type == "int":
        return int(raw_value)
    return str(raw_value)


def _metric_from_payload(payload: dict[str, Any], metric_name: str) -> float | None:
    raw_value = payload.get(metric_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    metric_value = float(raw_value)
    return metric_value if math.isfinite(metric_value) else None


def _payload_from_stdout(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    for line in reversed(
        [candidate.strip() for candidate in text.splitlines() if candidate.strip()]
    ):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _required_bounds(tunable: OpenTunerParameterSpec) -> tuple[int | float, int | float]:
    if tunable.min_value is None or tunable.max_value is None:
        raise ValueError(f"Missing bounds for {tunable.parameter_name}")
    return tunable.min_value, tunable.max_value


def select_best_trial_artifact(
    artifacts: list[dict[str, Any]], metric_direction: str
) -> dict[str, Any] | None:
    candidates = [
        artifact
        for artifact in artifacts
        if artifact.get("result_state") == "OK" and artifact.get("metric_value") is not None
    ]
    if not candidates:
        return None
    if metric_direction == "maximize":
        return max(candidates, key=lambda artifact: float(artifact["metric_value"]))
    return min(candidates, key=lambda artifact: float(artifact["metric_value"]))


class LibContentBenchOpenTunerInterface(MeasurementInterface):
    def __init__(
        self,
        args: Any,
        *,
        workflow_spec: OpenTunerWorkflowSpec,
        metric_name: str,
        metric_direction: str,
        timeout_seconds: int,
        output_dir: Path,
        repo_root: Path,
        study_trace: Any | None = None,
    ) -> None:
        super().__init__(
            args=args,
            project_name="LibContentBench",
            program_name=f"libcontentbench-opentuner-{workflow_spec.workflow_id}",
            program_version="v1",
        )
        self.workflow_spec = workflow_spec
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.timeout_seconds = timeout_seconds
        self.output_dir = output_dir
        self.repo_root = repo_root
        self.study_trace = study_trace
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = self.output_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self._trial_index = 0
        self._run_started_at = time.perf_counter()
        self.baseline_config = self.seed_configurations()[0]
        self.baseline_fingerprint = (
            config_fingerprint(self.baseline_config)
            if self.study_trace is None
            else self.study_trace.fingerprint(self.baseline_config)
        )
        self.study_unique_execution_count = 0
        if self.study_trace is not None:
            setattr(self, "extra_convergence_criteria", self._study_convergence_criteria)

    def _study_convergence_criteria(self, result: list[Any]) -> bool:
        del result
        return bool(
            self.study_trace is not None
            and self.study_unique_execution_count
            >= self.study_trace.unique_execution_target
        )

    def objective(self) -> MaximizeAccuracy:
        return MaximizeAccuracy()

    def manipulator(self) -> ConfigurationManipulator:
        manipulator = ConfigurationManipulator()
        for tunable in self.workflow_spec.tunables:
            if tunable.parameter_kind == "integer":
                min_value, max_value = _required_bounds(tunable)
                manipulator.add_parameter(
                    IntegerParameter(tunable.parameter_name, int(min_value), int(max_value))
                )
                continue
            if tunable.parameter_kind == "float":
                min_value, max_value = _required_bounds(tunable)
                manipulator.add_parameter(
                    FloatParameter(tunable.parameter_name, float(min_value), float(max_value))
                )
                continue
            manipulator.add_parameter(EnumParameter(tunable.parameter_name, list(tunable.options)))
        return manipulator

    def seed_configurations(self) -> list[dict[str, Any]]:
        baseline: dict[str, Any] = {}
        for tunable in self.workflow_spec.tunables:
            baseline[tunable.parameter_name] = _encode_parameter_value(
                tunable,
                self.workflow_spec.fixed_workload_payload[tunable.field_name],
            )
        return [baseline]

    def save_final_config(self, config: Any) -> None:
        del config
        artifacts = [
            load_json_object(path)
            for path in sorted(self.trials_dir.glob("trial_*/trial_artifact.json"))
        ]
        best = select_best_trial_artifact(artifacts, self.metric_direction)
        if best is None:
            return
        write_json(
            self.output_dir / "best_config.json",
            {
                "workflow": self.workflow_spec.workflow_id,
                "trial_index": best["trial_index"],
                "metric_name": self.metric_name,
                "metric_direction": self.metric_direction,
                "metric_value": best["metric_value"],
                "config": best["config"],
            },
        )

    def run(self, desired_result: Any, input: Any, limit: float) -> Result:
        del input
        del limit
        self._trial_index += 1
        trial_name = f"trial_{self._trial_index:04d}"
        trial_dir = self.trials_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = trial_dir / "output.json"
        case_output_dir = trial_dir / "case_output"
        runtime_root = trial_dir / "runtime"
        runtime_base = (
            Path("/var/tmp/libcontentbench/tuner-runtime") / self.output_dir.name / trial_name
        )
        config_data = dict(desired_result.configuration.data)
        current_fingerprint = (
            config_fingerprint(config_data)
            if self.study_trace is None
            else self.study_trace.fingerprint(config_data)
        )
        is_baseline = current_fingerprint == self.baseline_fingerprint
        if self.study_trace is not None and not is_baseline:
            self.study_unique_execution_count += 1

        payload = dict(self.workflow_spec.fixed_workload_payload)
        payload.update(
            {
                "case_id": trial_name,
                "label": sanitize_label(f"{self.workflow_spec.workflow_id}-{trial_name}"),
                "mode": "opentuner",
                "output_json": str(output_json_path),
            }
        )
        if any(
            field.field_name == "case_output_dir"
            for field in self.workflow_spec.direct_spec.cli_fields
        ):
            if self.workflow_spec.workflow_id == "torchdata-fsspec-s3-ddp":
                case_output_dir = runtime_base / "case_output"
            payload["case_output_dir"] = str(case_output_dir)
        if any(
            field.field_name == "runtime_root"
            for field in self.workflow_spec.direct_spec.cli_fields
        ):
            runtime_root = runtime_base / "runtime"
            payload["runtime_root"] = str(runtime_root)

        for tunable in self.workflow_spec.tunables:
            payload[tunable.field_name] = _coerce_parameter_value(
                tunable, config_data[tunable.parameter_name]
            )

        command = build_direct_command(self.workflow_spec.workflow_id, payload)
        trial_started_at = time.perf_counter()
        completed = self.call_program(command, limit=self.timeout_seconds, cwd=str(self.repo_root))
        trial_wall_time_sec = time.perf_counter() - trial_started_at
        trial_finished_run_wall_time_sec = time.perf_counter() - self._run_started_at

        stdout_text = _as_text(completed["stdout"])
        stderr_text = _as_text(completed["stderr"])
        stdout_payload = _payload_from_stdout(stdout_text)
        metric_value = (
            None
            if stdout_payload is None
            else _metric_from_payload(stdout_payload, self.metric_name)
        )

        fallback_payload = None
        if metric_value is None and output_json_path.exists():
            fallback_payload = load_json_object(output_json_path)
            metric_value = _metric_from_payload(fallback_payload, self.metric_name)

        state = "OK"
        result_time = DOMINATED_TIME
        objective_value = DOMINATED_ACCURACY
        if completed["timeout"]:
            state = "TIMEOUT"
        elif completed["returncode"] != 0:
            state = "ERROR"
        elif metric_value is None:
            state = "NO_METRIC"
        else:
            result_time = trial_wall_time_sec
            objective_value = float(metric_value)
            if self.metric_direction == "minimize":
                objective_value = -objective_value

        artifact = {
            "workflow": self.workflow_spec.workflow_id,
            "trial_index": self._trial_index,
            "trial_name": trial_name,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "metric_value": metric_value,
            "objective_value": objective_value,
            "result_state": state,
            "result_accuracy": objective_value,
            "result_time": result_time,
            "is_baseline": is_baseline,
            "config_fingerprint": current_fingerprint,
            "desired_result_id": getattr(desired_result, "id", None),
            "desired_result_generation": getattr(desired_result, "generation", None),
            "desired_result_requestor": getattr(desired_result, "requestor", None),
            "configuration_id": getattr(desired_result, "configuration_id", None),
            "configuration_hash": getattr(desired_result.configuration, "hash", None),
            "command": command,
            "config": config_data,
            "materialized_payload": payload,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": completed["returncode"],
            "timed_out": completed["timeout"],
            "trial_wall_time_sec": trial_wall_time_sec,
            "trial_finished_run_wall_time_sec": trial_finished_run_wall_time_sec,
            "call_program_time_sec": completed.get("time"),
            "output_json": str(output_json_path),
            "stdout_json": stdout_payload,
            "output_json_payload": fallback_payload,
        }
        artifact_path = trial_dir / "trial_artifact.json"
        write_json(artifact_path, artifact)
        if self.workflow_spec.workflow_id in {
            "dali-nvjpeg-pytorch-cudnn",
            "torchdata-fsspec-s3-ddp",
        }:
            shutil.rmtree(runtime_root.parent, ignore_errors=True)
        print(
            "LibContentBench trial "
            f"{self._trial_index}: state={state}, "
            f"{self.metric_name}={metric_value}, "
            f"goal={self.metric_direction}, "
            f"trial_wall_time_sec={trial_wall_time_sec}",
            flush=True,
        )
        if is_baseline and state != "OK":
            raise BaselineTrialFailed(state=state, artifact_path=artifact_path)
        result = Result()
        setattr(result, "time", result_time)
        setattr(result, "accuracy", objective_value)
        setattr(result, "state", state if state in {"OK", "TIMEOUT", "ERROR"} else "ERROR")
        return result
