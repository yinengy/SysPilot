#!/usr/bin/env python3
"""Command-line entry point for SMAC3 tuning of LibContentBench workflows.

Examples
--------
List workflows (no SMAC3 import needed)::

    python3 experiments/LibContentBench/smac3/cli.py --list-workflows

Print the baseline command + search space without running (host-safe)::

    python3 experiments/LibContentBench/smac3/cli.py \\
        --workflow nestedcpu-blas --size small --dry-run

Tune a workflow (``--budget`` == total executions including baseline)::

    python3 experiments/LibContentBench/smac3/cli.py \\
        --workflow nestedcpu-blas --size small --budget 20 \\
        --output-dir runs/smac3/nestedcpu-blas/small-b20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIZES = ("small", "medium", "large")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SMAC3 Bayesian optimization against a LibContentBench workflow."
    )
    parser.add_argument("--workflow", help="Workflow id to tune")
    parser.add_argument(
        "--size", choices=SIZES, default="medium", help="Workload size tier (default: medium)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help=(
            "Total benchmark executions, INCLUDING the baseline seed. "
            "With --budget 1, only the baseline runs. Default: 20"
        ),
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric key to optimize (default: each workflow's primary metric)",
    )
    parser.add_argument(
        "--metric-mode",
        choices=["maximize", "minimize"],
        default=None,
        help="Optimization direction (default: the workflow's configured mode)",
    )
    parser.add_argument("--output-dir", help="Directory for SMAC3 artifacts and reports")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Per-trial subprocess wall timeout (default: per-size safety value)",
    )
    parser.add_argument("--seed", type=int, default=0, help="SMAC3 / ConfigSpace seed (default: 0)")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Parallel trial workers. Keep 1 for valid contention measurements.",
    )
    parser.add_argument(
        "--crash-cost",
        type=float,
        default=1e9,
        help="Cost assigned to failed/timed-out trials (default: 1e9)",
    )
    parser.add_argument(
        "--list-workflows", action="store_true", help="Print supported workflow ids and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the baseline command and search space, then exit (no SMAC3 run)",
    )
    parser.add_argument(
        "--study-ledger",
        help=(
            "Opt in to the nestedcpu-blas deep-trace study and write the shared "
            "case-study JSONL ledger to this path"
        ),
    )
    parser.add_argument(
        "--study-proposal-cap",
        type=int,
        default=None,
        help="Override the study proposal cap (default from the shared manifest: 60)",
    )
    parser.add_argument(
        "--study-unique-execution-target",
        type=int,
        default=None,
        help="Override the unique non-baseline execution target (default: 30)",
    )
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.list_workflows:
        return
    if not args.workflow:
        parser.error("--workflow is required unless --list-workflows is used")
    if not args.dry_run and not args.output_dir:
        parser.error("--output-dir is required unless --dry-run or --list-workflows is used")
    if args.budget < 1:
        parser.error("--budget must be >= 1")
    if args.n_workers < 1:
        parser.error("--n-workers must be >= 1")
    has_study_override = (
        args.study_proposal_cap is not None or args.study_unique_execution_target is not None
    )
    if has_study_override and not args.study_ledger:
        parser.error(
            "--study-proposal-cap and --study-unique-execution-target require --study-ledger"
        )
    if args.study_ledger:
        if args.workflow != "nestedcpu-blas" or args.size != "small":
            parser.error("--study-ledger requires --workflow nestedcpu-blas --size small")
        if args.n_workers != 1:
            parser.error("--study-ledger requires --n-workers 1")
        if args.study_proposal_cap is not None and args.study_proposal_cap < 1:
            parser.error("--study-proposal-cap must be >= 1")
        if (
            args.study_unique_execution_target is not None
            and args.study_unique_execution_target < 1
        ):
            parser.error("--study-unique-execution-target must be >= 1")
        if (
            args.study_proposal_cap is not None
            and args.study_unique_execution_target is not None
            and args.study_proposal_cap < args.study_unique_execution_target
        ):
            parser.error("--study-proposal-cap cannot be below --study-unique-execution-target")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)

    from experiments.LibContentBench.smac3.workflows import (
        get_workflow,
        supported_workflow_ids,
    )

    if args.list_workflows:
        for workflow_id in supported_workflow_ids():
            spec = get_workflow(workflow_id)
            print(f"{workflow_id}\t{spec.primary_metric}\t{spec.metric_mode}")
        return 0

    spec = get_workflow(args.workflow)
    metric_name = args.metric or spec.primary_metric
    metric_mode = args.metric_mode or spec.metric_mode
    timeout_seconds = args.timeout_sec or spec.default_timeout_seconds(args.size)

    # Determinism: hash seed must be set before child processes are spawned.
    os.environ.setdefault("PYTHONHASHSEED", "0")

    if args.dry_run:
        from experiments.LibContentBench.libcontentbench.direct_commands import (
            build_direct_command,
        )
        from experiments.LibContentBench.smac3.workflows import (
            baseline_config_values,
            build_trial_payload,
        )

        baseline_values = baseline_config_values(spec)
        direct_fields = {f.field_name for f in spec.direct_spec.cli_fields}
        per_trial = {
            "case_id": "baseline",
            "label": f"{spec.workflow_id}-{args.size}-baseline",
            "mode": "smac3",
            "output_json": f"/tmp/smac3-dryrun/{spec.workflow_id}-{args.size}.json",
        }
        if "runtime_root" in direct_fields:
            per_trial["runtime_root"] = f"/tmp/smac3-dryrun/{spec.workflow_id}-{args.size}/runtime"
        if "case_output_dir" in direct_fields:
            per_trial["case_output_dir"] = (
                f"/tmp/smac3-dryrun/{spec.workflow_id}-{args.size}/case_output"
            )
        payload = build_trial_payload(spec, args.size, baseline_values, per_trial)
        command = build_direct_command(spec.workflow_id, payload)

        print(f"workflow:      {spec.workflow_id}")
        print(f"size:          {args.size}")
        print(f"metric:        {metric_name} ({metric_mode})")
        print(f"budget:        {args.budget} total executions (baseline included; "
              f"up to {max(0, args.budget - 1)} searched candidates)")
        print(f"timeout:       {timeout_seconds}s per trial")
        print(f"cpu bound:     {spec.tunables[0].max_value if spec.tunables else 'n/a'}")
        print("search space:")
        for tunable in spec.tunables:
            if tunable.kind == "integer":
                domain = f"int[{tunable.min_value}, {tunable.max_value}]"
            elif tunable.kind == "float":
                domain = f"float[{tunable.min_value}, {tunable.max_value}]"
            else:
                domain = f"categorical{list(tunable.options)}"
            default = tunable.encode_default(spec.baseline_runtime[tunable.field_name])
            print(f"  {tunable.field_name:<22} {domain}  default={default}")
        print("baseline command:")
        print(f"  {command}")
        return 0

    from experiments.LibContentBench.smac3.optimize import run_smac

    output_dir = Path(args.output_dir).resolve()
    summary = run_smac(
        spec=spec,
        size=args.size,
        budget=args.budget,
        metric_name=metric_name,
        metric_mode=metric_mode,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        seed=args.seed,
        n_workers=args.n_workers,
        repo_root=REPO_ROOT,
        crash_cost=args.crash_cost,
        study_ledger=(Path(args.study_ledger).resolve() if args.study_ledger else None),
        study_proposal_cap=args.study_proposal_cap,
        study_unique_execution_target=args.study_unique_execution_target,
    )

    print(
        f"workflow:            {summary['workflow']} "
        f"({summary['size']}, budget {summary['budget']})"
    )
    print(f"metric:              {summary['metric']} ({summary['metric_mode']})")
    print(f"total executions:    {summary['total_executions']} (baseline included; "
          f"searched candidates: {summary['n_search_trials']})")
    if summary.get("study"):
        study = summary["study"]
        print(
            f"study proposals:     {study['proposal_count']} / {study['proposal_cap']} "
            f"(unique launches: {study['unique_execution_count']} / "
            f"{study['unique_execution_target']})"
        )
        print(f"study terminal:      {study['terminal_reason']}")
    print(f"executions ok:       {summary['n_trials_ok']} / {summary['total_executions']}")
    print(f"trials to best:      {summary['trials_to_best']} "
          f"(rounds==trials for SMAC)")
    print(f"baseline metric:     {summary['baseline_metric']}"
          f"{'  [baseline is best]' if summary['baseline_is_best'] else ''}")
    print(f"best metric:         {summary['best_metric']} (trial {summary['best_trial_index']})")
    print(f"speedup vs baseline: {summary['speedup_vs_baseline']}")
    print(f"wall time:           {summary['wall_time_sec']}s "
          f"(to best: {summary['wall_time_to_best_sec']}s)")
    print(f"best config:         {summary['best_config']}")
    print(f"artifacts:           {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
