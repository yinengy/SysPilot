#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _supported_workflow_ids() -> tuple[str, ...]:
    from experiments.LibContentBench.opentuner_integration.specs import supported_workflow_ids

    return supported_workflow_ids()


def _load_workflow_spec(workflow_id: str) -> object:
    from experiments.LibContentBench.opentuner_integration.specs import load_workflow_spec

    return load_workflow_spec(workflow_id)


def _is_better_trial(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if candidate.get("result_state") != "OK" or candidate.get("metric_value") is None:
        return False
    if incumbent is None:
        return True
    candidate_metric = float(candidate["metric_value"])
    incumbent_metric = float(incumbent["metric_value"])
    if candidate.get("metric_direction") == "maximize":
        return candidate_metric > incumbent_metric
    return candidate_metric < incumbent_metric


def _load_trial_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    import json

    rows = []
    for artifact_path in sorted((output_dir / "trials").glob("trial_*/trial_artifact.json")):
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def print_run_summary(output_dir: Path, total_process_wall_time_sec: float) -> None:
    trials = _load_trial_artifacts(output_dir)
    best: dict[str, Any] | None = None
    for trial in trials:
        if _is_better_trial(trial, best):
            best = trial

    print("\nLibContentBench OpenTuner summary")
    print(f"  trials_completed: {len(trials)}")
    if trials:
        print("  trial_metrics:")
        for trial in trials:
            print(
                "    trial "
                f"{trial.get('trial_index')}: state={trial.get('result_state')}, "
                f"{trial.get('metric_name')}={trial.get('metric_value')}, "
                f"trial_wall_time_sec={trial.get('trial_wall_time_sec')}, "
                "finished_run_wall_time_sec="
                f"{trial.get('trial_finished_run_wall_time_sec')}"
            )
    if best is None:
        print("  best_trial: none")
    else:
        print(f"  best_trial: {best['trial_index']}")
        print(f"  best_metric_name: {best['metric_name']}")
        print(f"  best_metric_value: {best['metric_value']}")
        print(f"  best_metric_direction: {best['metric_direction']}")
        print(
            "  wall_time_until_best_trial_finished_sec: "
            f"{best.get('trial_finished_run_wall_time_sec')}"
        )
    print(f"  total_process_wall_time_sec: {total_process_wall_time_sec}")


def resolve_metric_direction(metric_name: str, requested_direction: str) -> str:
    normalized_metric = metric_name.strip().lower()
    if requested_direction != "auto":
        return requested_direction
    if normalized_metric == "throughput":
        return "maximize"
    if normalized_metric.endswith("_per_sec"):
        return "maximize"
    if "throughput" in normalized_metric:
        return "maximize"
    if "bandwidth" in normalized_metric:
        return "maximize"
    if normalized_metric.endswith("_rate") or normalized_metric.endswith("rate"):
        return "maximize"
    if re.search(r"(^|_)(latency|duration|time|wall_time)(_|$)", normalized_metric):
        return "minimize"
    return "minimize"


def build_parser() -> argparse.ArgumentParser:
    from opentuner.measurement.driver import argparser as measurement_driver_argparser
    from opentuner.measurement.interface import argparser as measurement_interface_argparser
    from opentuner.search.driver import argparser as search_driver_argparser
    from opentuner.search.manipulator import argparser as manipulator_argparser
    from opentuner.search.plugin import argparser as plugin_argparser
    from opentuner.search.technique import argparser as technique_argparser
    from opentuner.tuningrunmain import argparser as tuning_run_argparser

    parser = argparse.ArgumentParser(
        description="Run OpenTuner against fixed-workload LibContentBench direct runners.",
        parents=[
            measurement_interface_argparser,
            measurement_driver_argparser,
            search_driver_argparser,
            technique_argparser,
            plugin_argparser,
            manipulator_argparser,
            tuning_run_argparser,
        ],
    )
    parser.add_argument("--workflow", help="LibContentBench workflow to tune")
    parser.add_argument(
        "--output-dir",
        help="Directory for OpenTuner database and per-trial artifacts",
    )
    parser.add_argument(
        "--timeout-sec", type=int, default=600, help="Per-trial subprocess timeout in seconds"
    )
    parser.add_argument(
        "--metric",
        default="wall_time_sec",
        help=(
            "Metric key to optimize from the direct runner's metric JSON stdout "
            "or fallback output JSON"
        ),
    )
    parser.add_argument(
        "--metric-direction",
        choices=("auto", "minimize", "maximize"),
        default="auto",
        help=(
            "Optimization direction for --metric. 'auto' maximizes obvious "
            "throughput/rate metrics and minimizes time/latency metrics."
        ),
    )
    parser.add_argument(
        "--tuning-goal",
        choices=("minimize", "maximize"),
        help="Explicit tuning goal for --metric. Overrides --metric-direction when set.",
    )
    parser.add_argument(
        "--list-workflows", action="store_true", help="Print supported workflow ids and exit"
    )
    parser.set_defaults(parallelism=1)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.list_workflows:
        return
    if not args.workflow:
        parser.error("--workflow is required unless --list-workflows is used")
    if not args.output_dir:
        parser.error("--output-dir is required unless --list-workflows is used")
    supported_ids = _supported_workflow_ids()
    if args.workflow not in supported_ids:
        parser.error(
            f"Unsupported workflow {args.workflow!r}. "
            f"Supported workflows: {', '.join(supported_ids)}"
        )
    if args.timeout_sec < 1:
        parser.error("--timeout-sec must be >= 1")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.list_workflows:
        for workflow_id in _supported_workflow_ids():
            print(workflow_id)
        return 0

    from opentuner.tuningrunmain import TuningRunMain

    from experiments.LibContentBench.opentuner_integration.harness import (
        LibContentBenchOpenTunerInterface,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.database:
        args.database = str(output_dir / "opentuner.db")
    if not args.label:
        args.label = f"libcontentbench-{args.workflow}"

    workflow_spec = _load_workflow_spec(args.workflow)
    requested_direction = args.tuning_goal or args.metric_direction
    metric_direction = resolve_metric_direction(str(args.metric), str(requested_direction))
    interface = LibContentBenchOpenTunerInterface(
        args,
        workflow_spec=workflow_spec,
        metric_name=str(args.metric),
        metric_direction=metric_direction,
        timeout_seconds=int(args.timeout_sec),
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )
    process_started_at = time.perf_counter()
    TuningRunMain(interface, args).main()
    total_process_wall_time_sec = time.perf_counter() - process_started_at
    print_run_summary(output_dir, total_process_wall_time_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
