#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _supported_workflow_ids() -> tuple[str, ...]:
    from experiments.LibContentBench.opentuner_integration.specs import supported_workflow_ids

    return supported_workflow_ids()


def _load_workflow_spec(workflow_id: str) -> object:
    from experiments.LibContentBench.opentuner_integration.specs import load_workflow_spec

    return load_workflow_spec(workflow_id)


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
            "Metric key to minimize from the direct runner's metric JSON stdout "
            "or fallback output JSON"
        ),
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
    interface = LibContentBenchOpenTunerInterface(
        args,
        workflow_spec=workflow_spec,
        metric_name=str(args.metric),
        timeout_seconds=int(args.timeout_sec),
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )
    TuningRunMain(interface, args).main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
