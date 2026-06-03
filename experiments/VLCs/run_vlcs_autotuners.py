#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
# pyright: reportMissingImports=false, reportUnannotatedClassAttribute=false
# pyright: reportUnusedCallResult=false, reportImplicitStringConcatenation=false
# pyright: reportImplicitOverride=false

import argparse
import contextlib
import importlib.util
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
VLCS_DIR = Path(__file__).resolve().parent
RUNNER = "./run_vlcs.py"
CORE_COUNT = 24
METRIC_NAME = "walltime"
METRIC_MODE = "minimize"
DEFAULT_BUDGET = 60
DEFAULT_TIMEOUT_SEC = 300
TUNERS = ("smac3", "opentuner")
BASELINE_ASSIGNMENT = {core: 0 if core < 12 else 1 for core in range(CORE_COUNT)}


@dataclass(frozen=True)
class TrialRecord:
    trial_index: int
    tuner: str
    state: str
    metric_value: float | None
    objective_cost: float
    assignment: dict[int, int]
    command: str
    returncode: int | None
    timed_out: bool
    elapsed_sec: float
    cumulative_wall_sec: float
    stdout_path: str
    stderr_path: str
    artifact_path: str


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            _ = stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SMAC3 and/or OpenTuner on experiments/VLCs/run_vlcs.py core "
            "assignment flags, saving raw logs and best-series CSVs under runs/."
        )
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Total evaluations per tuner, including the baseline seed (default: 60).")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Per-trial timeout in seconds (default: 300).")
    parser.add_argument("--seed", type=int, default=0, help="Optimizer seed (default: 0).")
    parser.add_argument("--tuner", choices=("both", *TUNERS), default="both", help="Which tuner to run (default: both).")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory under runs/. Defaults to runs/vlcs_autotuners_budget<BUDGET>_<UTC timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and paths without importing/running SMAC3 or OpenTuner.")
    parser.add_argument("--crash-cost", type=float, default=1e9, help="Cost assigned to failed/invalid/timed-out trials (default: 1e9).")
    args = parser.parse_args()
    if cast(int, args.budget) < 1:
        parser.error("--budget must be >= 1")
    if cast(int, args.timeout_sec) < 1:
        parser.error("--timeout-sec must be >= 1")
    return args


def selected_tuners(choice: str) -> list[str]:
    if choice == "both":
        return list(TUNERS)
    return [choice]


def default_output_root(budget: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "runs" / f"vlcs_autotuners_budget{budget}_{stamp}"


def assignment_to_command(assignment: dict[int, int]) -> list[str]:
    command = [RUNNER]
    for core in range(CORE_COUNT):
        command.extend([f"--core-{core}", str(assignment[core])])
    return command


def shell_join(parts: Iterable[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in parts)


def assignment_from_config(config: dict[str, Any]) -> dict[int, int]:
    assignment: dict[int, int] = {}
    for core in range(CORE_COUNT):
        value = config.get(f"core_{core}", BASELINE_ASSIGNMENT[core])
        assignment[core] = int(value)
    return assignment


def is_valid_assignment(assignment: dict[int, int]) -> bool:
    return any(value == 0 for value in assignment.values()) and any(value == 1 for value in assignment.values())


def parse_metric(stdout_text: str) -> float | None:
    for line in reversed([candidate.strip() for candidate in stdout_text.splitlines() if candidate.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw_value = payload.get(METRIC_NAME)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if math.isfinite(value):
            return value
    return None


def speedup(metric: float | None, baseline: float | None) -> float | None:
    if metric is None or baseline is None or metric == 0:
        return None
    return baseline / metric


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class VlcTrialRunner:
    def __init__(self, *, tuner: str, output_dir: Path, timeout_sec: int, crash_cost: float) -> None:
        self.tuner = tuner
        self.output_dir = output_dir
        self.timeout_sec = timeout_sec
        self.crash_cost = crash_cost
        self.trials_dir = output_dir / "trials"
        self.records: list[TrialRecord] = []
        self.started_at = time.perf_counter()
        self.trials_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, assignment: dict[int, int]) -> float:
        trial_index = len(self.records) + 1
        trial_dir = self.trials_dir / f"trial_{trial_index:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = trial_dir / "stdout.txt"
        stderr_path = trial_dir / "stderr.txt"
        artifact_path = trial_dir / "trial_artifact.json"
        command = assignment_to_command(assignment)
        command_text = shell_join(command)
        (trial_dir / "command.sh").write_text(command_text + "\n")

        state = "OK"
        metric_value: float | None = None
        returncode: int | None = None
        timed_out = False
        stdout_text = ""
        stderr_text = ""
        trial_started_at = time.perf_counter()

        if not is_valid_assignment(assignment):
            state = "INVALID"
            elapsed_sec = 0.0
        else:
            env = dict(os.environ)
            env.setdefault("PYTHONHASHSEED", "0")
            try:
                completed = subprocess.run(
                    command,
                    cwd=VLCS_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_sec,
                )
                elapsed_sec = time.perf_counter() - trial_started_at
                stdout_text = completed.stdout
                stderr_text = completed.stderr
                returncode = completed.returncode
                if completed.returncode != 0:
                    state = "ERROR"
                else:
                    metric_value = parse_metric(stdout_text)
                    if metric_value is None:
                        state = "NO_METRIC"
            except subprocess.TimeoutExpired as exc:
                elapsed_sec = time.perf_counter() - trial_started_at
                timed_out = True
                state = "TIMEOUT"
                stdout_text = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr_text = exc.stderr if isinstance(exc.stderr, str) else ""

        stdout_path.write_text(stdout_text)
        stderr_path.write_text(stderr_text)
        cumulative_wall_sec = time.perf_counter() - self.started_at
        objective_cost = metric_value if state == "OK" and metric_value is not None else self.crash_cost
        record = TrialRecord(
            trial_index=trial_index,
            tuner=self.tuner,
            state=state,
            metric_value=metric_value,
            objective_cost=objective_cost,
            assignment=dict(assignment),
            command=command_text,
            returncode=returncode,
            timed_out=timed_out,
            elapsed_sec=elapsed_sec,
            cumulative_wall_sec=cumulative_wall_sec,
            stdout_path=str(stdout_path.relative_to(self.output_dir)),
            stderr_path=str(stderr_path.relative_to(self.output_dir)),
            artifact_path=str(artifact_path.relative_to(self.output_dir)),
        )
        self.records.append(record)
        write_json(
            artifact_path,
            {
                "trial_index": trial_index,
                "tuner": self.tuner,
                "state": state,
                "metric_name": METRIC_NAME,
                "metric_mode": METRIC_MODE,
                "metric_value": metric_value,
                "objective_cost": objective_cost,
                "assignment": assignment,
                "command": command_text,
                "returncode": returncode,
                "timed_out": timed_out,
                "elapsed_sec": elapsed_sec,
                "cumulative_wall_sec": cumulative_wall_sec,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            },
        )
        print(
            f"{self.tuner} trial {trial_index:04d}: state={state}, "
            f"{METRIC_NAME}={metric_value}, elapsed_sec={elapsed_sec:.4f}",
            flush=True,
        )
        return objective_cost


def select_best(records: list[TrialRecord]) -> TrialRecord | None:
    ok = [record for record in records if record.state == "OK" and record.metric_value is not None]
    if not ok:
        return None
    return min(ok, key=lambda record: cast(float, record.metric_value))


def baseline_record(records: list[TrialRecord]) -> TrialRecord | None:
    return next((record for record in records if record.assignment == BASELINE_ASSIGNMENT and record.state == "OK"), None)


def write_best_series(tuner: str, records: list[TrialRecord], output_root: Path) -> None:
    baseline = baseline_record(records)
    baseline_metric = baseline.metric_value if baseline is not None else None
    current_best: float | None = None
    rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.trial_index):
        metric = record.metric_value if record.state == "OK" else None
        if metric is not None and (current_best is None or metric < current_best):
            current_best = metric
        rows.append(
            {
                "trial": record.trial_index,
                "metric": "" if metric is None else repr(metric),
                "current_best_metric": "" if current_best is None else repr(current_best),
                "speedup_vs_baseline": "" if current_best is None else repr(speedup(current_best, baseline_metric)),
                "state": record.state,
                "config": json.dumps({f"core_{core}": value for core, value in sorted(record.assignment.items())}, sort_keys=True),
            }
        )
    write_csv(
        output_root / "best_series" / f"vlcs_{tuner}_b{len(records)}.csv",
        rows,
        ["trial", "metric", "current_best_metric", "speedup_vs_baseline", "state", "config"],
    )


def write_tuner_summary(tuner: str, output_dir: Path, runner: VlcTrialRunner) -> None:
    records = sorted(runner.records, key=lambda item: item.trial_index)
    best = select_best(records)
    baseline = baseline_record(records)
    baseline_metric = baseline.metric_value if baseline is not None else None
    best_metric = best.metric_value if best is not None else None
    trial_rows: list[dict[str, object]] = []
    for record in records:
        trial_rows.append(
            {
                "trial_index": record.trial_index,
                "state": record.state,
                "metric_value": "" if record.metric_value is None else record.metric_value,
                "speedup_vs_baseline": "" if record.metric_value is None else speedup(record.metric_value, baseline_metric),
                "elapsed_sec": record.elapsed_sec,
                "cumulative_wall_sec": record.cumulative_wall_sec,
                **{f"core_{core}": record.assignment[core] for core in range(CORE_COUNT)},
            }
        )
    write_csv(
        output_dir / "trials.csv",
        trial_rows,
        [
            "trial_index",
            "state",
            "metric_value",
            "speedup_vs_baseline",
            "elapsed_sec",
            "cumulative_wall_sec",
            *[f"core_{core}" for core in range(CORE_COUNT)],
        ],
    )
    write_json(
        output_dir / "summary.json",
        {
            "benchmark": "vlcs",
            "tuner": tuner,
            "budget": len(records),
            "metric": METRIC_NAME,
            "metric_mode": METRIC_MODE,
            "total_trials": len(records),
            "total_rounds": len(records),
            "trials_ok": sum(1 for record in records if record.state == "OK"),
            "baseline_metric": baseline_metric,
            "best_metric": best_metric,
            "speedup_vs_baseline": speedup(best_metric, baseline_metric),
            "best_trial": None if best is None else best.trial_index,
            "trials_to_best": None if best is None else best.trial_index,
            "wall_time_sec": records[-1].cumulative_wall_sec if records else 0.0,
            "wall_time_to_best_sec": None if best is None else best.cumulative_wall_sec,
            "baseline_assignment": BASELINE_ASSIGNMENT,
            "best_assignment": None if best is None else best.assignment,
            "best_command": None if best is None else best.command,
            "best_series_csv": str((Path("best_series") / f"vlcs_{tuner}_b{len(records)}.csv")),
        },
    )


def write_master_summary(output_root: Path, tuners: list[str]) -> None:
    rows: list[dict[str, object]] = []
    for tuner in tuners:
        summary_path = output_root / "tuner_outputs" / tuner / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "tuner": tuner,
                "budget": summary.get("budget"),
                "total_trials": summary.get("total_trials"),
                "trials_ok": summary.get("trials_ok"),
                "baseline_metric": summary.get("baseline_metric"),
                "best_metric": summary.get("best_metric"),
                "speedup_vs_baseline": summary.get("speedup_vs_baseline"),
                "best_trial": summary.get("best_trial"),
                "wall_time_sec": summary.get("wall_time_sec"),
                "wall_time_to_best_sec": summary.get("wall_time_to_best_sec"),
                "best_series_csv": summary.get("best_series_csv"),
            }
        )
    write_csv(
        output_root / "summary.csv",
        rows,
        [
            "tuner",
            "budget",
            "total_trials",
            "trials_ok",
            "baseline_metric",
            "best_metric",
            "speedup_vs_baseline",
            "best_trial",
            "wall_time_sec",
            "wall_time_to_best_sec",
            "best_series_csv",
        ],
    )


def run_smac3(*, budget: int, seed: int, output_root: Path, timeout_sec: int, crash_cost: float) -> None:
    from ConfigSpace import Categorical, Configuration, ConfigurationSpace
    from smac import HyperparameterOptimizationFacade, Scenario

    output_dir = output_root / "tuner_outputs" / "smac3"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = VlcTrialRunner(tuner="smac3", output_dir=output_dir, timeout_sec=timeout_sec, crash_cost=crash_cost)
    cs = ConfigurationSpace(seed=seed)
    cs.add([Categorical(f"core_{core}", items=[0, 1], default=BASELINE_ASSIGNMENT[core]) for core in range(CORE_COUNT)])
    baseline_cfg = Configuration(cs, values={f"core_{core}": value for core, value in BASELINE_ASSIGNMENT.items()})
    scenario = Scenario(
        cs,
        name=f"vlcs-smac3-b{budget}-s{seed}",
        output_directory=output_dir / "smac3_runs",
        deterministic=True,
        objectives="cost",
        crash_cost=crash_cost,
        n_trials=budget,
        seed=seed,
        n_workers=1,
    )

    def target_function(config: Any, seed: int = 0) -> float:
        del seed
        return runner.evaluate(assignment_from_config(dict(config)))

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=max(0, budget - 1),
        additional_configs=[baseline_cfg],
    )
    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function,
        initial_design=initial_design,
        overwrite=True,
    )
    smac.optimize()
    write_tuner_summary("smac3", output_dir, runner)
    write_best_series("smac3", runner.records, output_root)


def opentuner_argparser() -> argparse.ArgumentParser:
    from opentuner.measurement.driver import argparser as measurement_driver_argparser
    from opentuner.measurement.interface import argparser as measurement_interface_argparser
    from opentuner.search.driver import argparser as search_driver_argparser
    from opentuner.search.manipulator import argparser as manipulator_argparser
    from opentuner.search.plugin import argparser as plugin_argparser
    from opentuner.search.technique import argparser as technique_argparser
    from opentuner.tuningrunmain import argparser as tuning_run_argparser

    parser = argparse.ArgumentParser(
        parents=[
            measurement_interface_argparser,
            measurement_driver_argparser,
            search_driver_argparser,
            technique_argparser,
            plugin_argparser,
            manipulator_argparser,
            tuning_run_argparser,
        ]
    )
    parser.set_defaults(parallelism=1)
    return parser


def run_opentuner(*, budget: int, seed: int, output_root: Path, timeout_sec: int, crash_cost: float) -> None:
    del seed
    from opentuner import ConfigurationManipulator, MeasurementInterface, Result
    from opentuner.search.manipulator import EnumParameter
    from opentuner.search.objective import MaximizeAccuracy
    from opentuner.tuningrunmain import TuningRunMain

    output_dir = output_root / "tuner_outputs" / "opentuner"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = VlcTrialRunner(tuner="opentuner", output_dir=output_dir, timeout_sec=timeout_sec, crash_cost=crash_cost)

    class VlcOpenTunerInterface(MeasurementInterface):
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args=args, project_name="VLCs", program_name="vlcs-opentuner", program_version="v1")

        def objective(self) -> MaximizeAccuracy:
            return MaximizeAccuracy()

        def manipulator(self) -> ConfigurationManipulator:
            manipulator = ConfigurationManipulator()
            for core in range(CORE_COUNT):
                manipulator.add_parameter(EnumParameter(f"core_{core}", ["0", "1"]))
            return manipulator

        def seed_configurations(self) -> list[dict[str, str]]:
            return [{f"core_{core}": str(value) for core, value in BASELINE_ASSIGNMENT.items()}]

        def save_final_config(self, config: Any) -> None:
            write_json(output_dir / "opentuner_best_config_raw.json", dict(config.data))

        def run(self, desired_result: Any, input: Any, limit: float) -> Result:
            del input
            del limit
            assignment = assignment_from_config(dict(desired_result.configuration.data))
            metric = runner.evaluate(assignment)
            record = runner.records[-1]
            result = Result()
            setattr(result, "time", record.elapsed_sec if record.state == "OK" else float("inf"))
            setattr(result, "accuracy", -metric)
            setattr(result, "state", record.state)
            return result

    parser = opentuner_argparser()
    ot_args = parser.parse_args(
        [
            "--test-limit",
            str(budget),
            "--parallelism",
            "1",
            "--database",
            str(output_dir / "opentuner.db"),
            "--label",
            f"vlcs-opentuner-b{budget}",
        ]
    )
    interface = VlcOpenTunerInterface(ot_args)
    TuningRunMain(interface, ot_args).main()
    write_tuner_summary("opentuner", output_dir, runner)
    write_best_series("opentuner", runner.records, output_root)


def run_with_raw_log(tuner: str, log_root: Path, func: Any, **kwargs: Any) -> None:
    raw_path = log_root / "raw" / f"{tuner}.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w") as raw_handle:
        tee_stdout = Tee(sys.stdout, raw_handle)
        tee_stderr = Tee(sys.stderr, raw_handle)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"Starting {tuner} at {datetime.now(timezone.utc).isoformat()}", flush=True)
            func(**kwargs)
            print(f"Finished {tuner} at {datetime.now(timezone.utc).isoformat()}", flush=True)


def print_dry_run(output_root: Path, tuners: list[str], budget: int, timeout_sec: int, seed: int) -> None:
    print("VLCs autotuner dry run")
    print(f"repo_root:   {REPO_ROOT}")
    print(f"vlcs_dir:    {VLCS_DIR}")
    print(f"budget:      {budget} total evaluations per tuner, baseline included")
    print(f"timeout:     {timeout_sec}s per trial")
    print(f"seed:        {seed}")
    print(f"output_root: {output_root}")
    print(f"tuners:      {', '.join(tuners)}")
    print("baseline command:")
    print(f"  cd {VLCS_DIR} && {shell_join(assignment_to_command(BASELINE_ASSIGNMENT))}")
    print("artifacts:")
    for tuner in tuners:
        print(f"  raw log:     {output_root / 'raw' / f'{tuner}.log'}")
        print(f"  best series: {output_root / 'best_series' / f'vlcs_{tuner}_b{budget}.csv'}")
        print(f"  trials:      {output_root / 'tuner_outputs' / tuner / 'trials'}")


def iter_tuner_functions(tuners: list[str]) -> Iterator[tuple[str, Any]]:
    for tuner in tuners:
        if tuner == "smac3":
            yield tuner, run_smac3
        elif tuner == "opentuner":
            yield tuner, run_opentuner
        else:
            raise ValueError(f"unknown tuner: {tuner}")


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def preflight_tuner_dependencies(tuners: list[str]) -> None:
    missing: list[str] = []
    if "smac3" in tuners:
        missing_smac = [
            module_name
            for module_name in ("ConfigSpace", "smac")
            if not module_available(module_name)
        ]
        if missing_smac:
            missing.append(
                "SMAC3 requires installed packages `smac` and `ConfigSpace`; "
                f"missing modules: {', '.join(missing_smac)}"
            )
    if "opentuner" in tuners and not module_available("opentuner"):
        missing.append("OpenTuner requires installed package `opentuner`")
    if missing:
        raise RuntimeError("Missing tuner dependencies. " + " ".join(missing))


def main() -> int:
    args = parse_args()
    budget = cast(int, args.budget)
    timeout_sec = cast(int, args.timeout_sec)
    seed = cast(int, args.seed)
    crash_cost = cast(float, args.crash_cost)
    tuners = selected_tuners(cast(str, args.tuner))
    output_root = cast(Path | None, args.output_root) or default_output_root(budget)
    output_root = output_root.resolve()

    if args.dry_run:
        print_dry_run(output_root, tuners, budget, timeout_sec, seed)
        return 0

    try:
        preflight_tuner_dependencies(tuners)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "run_config.json",
        {
            "budget": budget,
            "timeout_sec": timeout_sec,
            "seed": seed,
            "tuners": tuners,
            "metric": METRIC_NAME,
            "metric_mode": METRIC_MODE,
            "baseline_assignment": BASELINE_ASSIGNMENT,
            "baseline_command": shell_join(assignment_to_command(BASELINE_ASSIGNMENT)),
        },
    )

    for tuner, func in iter_tuner_functions(tuners):
        run_with_raw_log(
            tuner,
            output_root,
            func,
            budget=budget,
            seed=seed,
            output_root=output_root,
            timeout_sec=timeout_sec,
            crash_cost=crash_cost,
        )
    write_master_summary(output_root, tuners)
    print(f"Wrote VLCs autotuner artifacts to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
