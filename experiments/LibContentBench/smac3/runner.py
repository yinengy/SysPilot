"""SMAC3 target function for LibContentBench direct runners.

Each trial:
  1. materializes the full command via ``build_direct_command`` (the suite's
     single source of truth), using the size-tier workload shape + the trial's
     tuned knobs + per-trial identity/output paths;
  2. runs it as a subprocess in the repo root with ``PYTHONHASHSEED=0`` for
     determinism and an outer wall-clock timeout;
  3. parses the metric-only JSON from stdout (falling back to the
     ``--output-json`` artifact);
  4. returns a SMAC *cost* (``-metric`` for maximize objectives, ``+metric`` for
     minimize), or a large finite ``crash_cost`` on failure/timeout;
  5. writes a ``trials/trial_XXXX/trial_artifact.json`` record and appends an
     in-memory record used by :mod:`optimize` to build the run summary.

Trials use per-trial ``runtime_root`` / ``case_output_dir`` directories for
isolation, mirroring the OpenTuner integration.
"""

from __future__ import annotations

import json
import numbers
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from experiments.LibContentBench.libcontentbench.common import (
    load_json_object,
    sanitize_label,
    write_json,
)
from experiments.LibContentBench.libcontentbench.direct_commands import build_direct_command
from experiments.LibContentBench.smac3.workflows import WorkflowSpec, build_trial_payload


def _payload_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Return the last JSON object printed on stdout, if any."""
    text = stdout.strip()
    if not text:
        return None
    for line in reversed([c.strip() for c in text.splitlines() if c.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}
    return None


def _metric_from_payload(payload: dict[str, Any], metric_name: str) -> float | None:
    raw = payload.get(metric_name)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _normalize_config(config: Any) -> dict[str, Any]:
    """Plain-Python copy of a ConfigSpace Configuration.

    ConfigSpace 1.x Categorical values come back as ``numpy.str_`` (and integers
    can be numpy scalars); converting to native ``str``/``int``/``float`` keeps
    artifacts, JSON, and printed configs clean and avoids ``np.str_('...')`` noise.
    """
    normalized: dict[str, Any] = {}
    for key, value in dict(config).items():
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, str):
            normalized[key] = str(value)
        elif isinstance(value, numbers.Integral):
            normalized[key] = int(value)
        elif isinstance(value, numbers.Real):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


class TrialEvaluator:
    """Picklable SMAC3 target callable for one (workflow, size, budget) job."""

    def __init__(
        self,
        *,
        spec: WorkflowSpec,
        size: str,
        metric_name: str,
        metric_mode: str,
        timeout_seconds: int,
        output_dir: Path,
        repo_root: Path,
        crash_cost: float,
    ) -> None:
        self.spec = spec
        self.size = size
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.timeout_seconds = int(timeout_seconds)
        self.output_dir = Path(output_dir)
        self.repo_root = Path(repo_root)
        self.crash_cost = float(crash_cost)

        self.trials_dir = self.output_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._counter = 0
        self._start_monotonic = time.monotonic()
        self.records: list[dict[str, Any]] = []

    # SMAC calls the instance as ``target(config, seed)``.
    def __call__(self, config: Any, seed: int = 0) -> float:
        config_values = _normalize_config(config)

        with self._lock:
            self._counter += 1
            trial_index = self._counter

        trial_name = f"trial_{trial_index:04d}"
        trial_dir = self.trials_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = trial_dir / "output.json"

        per_trial: dict[str, Any] = {
            "case_id": trial_name,
            "label": sanitize_label(f"{self.spec.workflow_id}-{self.size}-{trial_name}"),
            "mode": "smac3",
            "output_json": str(output_json_path),
        }
        direct_field_names = {f.field_name for f in self.spec.direct_spec.cli_fields}
        if "runtime_root" in direct_field_names:
            per_trial["runtime_root"] = str(trial_dir / "runtime")
        if "case_output_dir" in direct_field_names:
            per_trial["case_output_dir"] = str(trial_dir / "case_output")

        payload = build_trial_payload(self.spec, self.size, config_values, per_trial)
        command = build_direct_command(self.spec.workflow_id, payload)

        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"

        trial_start = time.monotonic()
        returncode: int | None = None
        timed_out = False
        stdout_text = ""
        stderr_text = ""
        try:
            completed = subprocess.run(
                shlex.split(command),
                cwd=str(self.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout_text = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(
                exc.stdout, bytes
            ) else (exc.stdout or "")
            stderr_text = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(
                exc.stderr, bytes
            ) else (exc.stderr or "")
        elapsed_sec = time.monotonic() - trial_start

        stdout_payload = _payload_from_stdout(stdout_text)
        metric_value = (
            None
            if stdout_payload is None
            else _metric_from_payload(stdout_payload, self.metric_name)
        )
        fallback_payload = None
        if metric_value is None and output_json_path.exists():
            try:
                fallback_payload = load_json_object(output_json_path)
                metric_value = _metric_from_payload(fallback_payload, self.metric_name)
            except (ValueError, OSError, json.JSONDecodeError):
                fallback_payload = None

        if timed_out:
            state = "TIMEOUT"
        elif returncode != 0:
            state = "ERROR"
        elif metric_value is None:
            state = "NO_METRIC"
        else:
            state = "OK"

        if state == "OK":
            assert metric_value is not None
            cost = -metric_value if self.metric_mode == "maximize" else metric_value
        else:
            cost = self.crash_cost

        cumulative_wall_sec = time.monotonic() - self._start_monotonic

        artifact = {
            "workflow": self.spec.workflow_id,
            "size": self.size,
            "trial_index": trial_index,
            "trial_name": trial_name,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
            "metric_value": metric_value,
            "cost": cost,
            "result_state": state,
            "command": command,
            "config": config_values,
            "materialized_payload": payload,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_sec": elapsed_sec,
            "cumulative_wall_sec": cumulative_wall_sec,
            "output_json": str(output_json_path),
            "stdout_tail": _tail(stdout_text),
            "stderr_tail": _tail(stderr_text),
            "stdout_json": stdout_payload,
        }
        write_json(trial_dir / "trial_artifact.json", artifact)

        with self._lock:
            self.records.append(
                {
                    "trial_index": trial_index,
                    "state": state,
                    "metric_value": metric_value,
                    "cost": cost,
                    "config": config_values,
                    "elapsed_sec": elapsed_sec,
                    "cumulative_wall_sec": cumulative_wall_sec,
                }
            )

        return float(cost)
