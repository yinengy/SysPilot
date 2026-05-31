#!/usr/bin/env python3
"""Run the full SMAC3 fair-comparison sweep over LibContentBench workflows.

One command runs every (workflow, size, budget) tuning job **sequentially**
(fairness — concurrent trials would contend and corrupt timing), writes each
run's artifacts under ``<output-root>/<workflow>/<size>-b<budget>-s<seed>/``, and
aggregates every run's ``run_report.csv`` into ``<output-root>/master_report.csv``
— row-for-row comparable to SysPilot's fixed-budget results.

The matrix defaults mirror the SysPilot protocol in
``docs/fixed_budget_exp_prompt.md``: 4 workflows x {small,medium,large} x
budgets {10,20,30}. Each SMAC run is the "smac3" mode alongside SysPilot's
"contract" and "manual" modes for the same (workflow, size, budget).

The sweep is **resumable**: a run whose ``summary.json`` already exists is reused
(skipped) unless ``--force`` is given.

Usage (single command):

    sh experiments/LibContentBench/smac3/run_experiment.sh

or directly with the driver venv python:

    experiments/LibContentBench/smac3/.venv/bin/python \\
      experiments/LibContentBench/smac3/experiment.py [--workflows ...] [--sizes ...] [--budgets ...]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

SMAC3_DIR = Path(__file__).resolve().parent
CLI = SMAC3_DIR / "cli.py"

ALL_WORKFLOWS = (
    "nestedcpu-blas",
    "arrow-dask-duckdb-memspill",
    "hf-datasets-tokenizers-dataloader",
    "torchdata-fsspec-s3-ddp",
    "dali-nvjpeg-pytorch-cudnn",
)
ALL_SIZES = ("small", "medium", "large")
DEFAULT_BUDGETS = (10, 20, 30)
DOCKER_WORKFLOWS = {"torchdata-fsspec-s3-ddp"}  # needs Docker (local MinIO) at run time
GPU_WORKFLOWS = {"dali-nvjpeg-pytorch-cudnn"}  # needs CUDA + NVIDIA DALI + CUDA MPS

MASTER_FIELDS = [
    "workflow",
    "size",
    "mode",
    "budget",
    "total_trials",
    "total_rounds",
    "trials_to_best",
    "rounds_to_best",
    "wall_time_sec",
    "wall_time_to_best_sec",
    "baseline_metric",
    "best_metric",
    "speedup_vs_baseline",
    "status",
    "output_dir",
]


def _docker_usable() -> tuple[bool, str]:
    """Return (usable, reason). Docker is only usable if the CLI is present AND
    the daemon is reachable by this user — having `docker` on PATH is not enough
    (a user not in the `docker` group gets a permission-denied socket error)."""
    if shutil.which("docker") is None:
        return False, "docker CLI not found"
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20, check=False
        )
    except Exception as exc:  # noqa: BLE001 - any failure means not usable
        return False, f"docker info failed ({exc})"
    if proc.returncode != 0:
        return False, "docker daemon not reachable (not in `docker` group? try sudo)"
    return True, "ok"


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in _csv_list(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full SMAC3 fair-comparison sweep and aggregate a master report."
    )
    parser.add_argument(
        "--workflows",
        type=_csv_list,
        default=list(ALL_WORKFLOWS),
        help=f"comma-separated workflow ids (default: all = {','.join(ALL_WORKFLOWS)})",
    )
    parser.add_argument(
        "--sizes",
        type=_csv_list,
        default=list(ALL_SIZES),
        help="comma-separated sizes (default: small,medium,large)",
    )
    parser.add_argument(
        "--budgets",
        type=_int_list,
        default=list(DEFAULT_BUDGETS),
        help="comma-separated search-trial budgets, excl. baseline (default: 10,20,30)",
    )
    parser.add_argument(
        "--output-root",
        default=str(SMAC3_DIR / "runs"),
        help="root directory for all run artifacts + master_report.csv (default: smac3/runs)",
    )
    parser.add_argument("--seed", type=int, default=0, help="SMAC3 seed (default: 0)")
    parser.add_argument(
        "--timeout-sec", type=int, default=None, help="override per-trial timeout for all runs"
    )
    parser.add_argument(
        "--n-workers", type=int, default=1, help="parallel trial workers per run (keep 1 for fairness)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run even if a run's summary.json already exists"
    )
    parser.add_argument(
        "--skip-torchdata",
        action="store_true",
        help="skip torchdata-fsspec-s3-ddp (needs Docker)",
    )
    parser.add_argument(
        "--skip-dali",
        action="store_true",
        help="skip dali-nvjpeg-pytorch-cudnn (needs CUDA + NVIDIA DALI + CUDA MPS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned run matrix and exit without running",
    )
    return parser


def _read_run_report(run_dir: Path) -> dict[str, str] | None:
    report_path = run_dir / "run_report.csv"
    if not report_path.exists():
        return None
    with report_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def _blank_row(workflow: str, size: str, budget: int, status: str, run_dir: Path) -> dict[str, str]:
    row = {field: "" for field in MASTER_FIELDS}
    row.update(
        {
            "workflow": workflow,
            "size": size,
            "mode": "smac3",
            "budget": str(budget),
            "status": status,
            "output_dir": str(run_dir),
        }
    )
    return row


def _write_master(output_root: Path, rows: list[dict[str, str]]) -> Path:
    master_path = output_root / "master_report.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with master_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MASTER_FIELDS})
    return master_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Determinism for the orchestrator interpreter (children get it via the runner).
    os.environ.setdefault("PYTHONHASHSEED", "0")

    for workflow in args.workflows:
        if workflow not in ALL_WORKFLOWS:
            parser.error(f"unknown workflow {workflow!r}; choose from {', '.join(ALL_WORKFLOWS)}")
    for size in args.sizes:
        if size not in ALL_SIZES:
            parser.error(f"unknown size {size!r}; choose from {', '.join(ALL_SIZES)}")

    workflows = list(args.workflows)

    docker_ok, docker_reason = _docker_usable()
    if args.skip_torchdata or not docker_ok:
        before = set(workflows) & DOCKER_WORKFLOWS
        workflows = [w for w in workflows if w not in DOCKER_WORKFLOWS]
        if before:
            why = "--skip-torchdata" if args.skip_torchdata else docker_reason
            print(f"[plan] skipping {', '.join(sorted(before))} ({why})")

    gpu_missing = shutil.which("nvidia-smi") is None
    if args.skip_dali or gpu_missing:
        before = set(workflows) & GPU_WORKFLOWS
        workflows = [w for w in workflows if w not in GPU_WORKFLOWS]
        if before:
            why = "no GPU (nvidia-smi not found)" if gpu_missing else "--skip-dali"
            print(f"[plan] skipping {', '.join(sorted(before))} ({why})")

    output_root = Path(args.output_root).resolve()
    plan = [(w, s, b) for w in workflows for s in args.sizes for b in args.budgets]

    print(f"[plan] {len(plan)} runs (sequential) -> {output_root}")
    print(f"[plan] workflows={workflows}")
    print(f"[plan] sizes={args.sizes} budgets={args.budgets} seed={args.seed}")
    for workflow, size, budget in plan:
        print(f"        - {workflow} {size} b{budget}")
    if args.dry_run:
        return 0

    rows: list[dict[str, str]] = []
    for index, (workflow, size, budget) in enumerate(plan, start=1):
        run_dir = output_root / workflow / f"{size}-b{budget}-s{args.seed}"
        tag = f"[{index}/{len(plan)}] {workflow} {size} b{budget}"

        if (run_dir / "summary.json").exists() and not args.force:
            report = _read_run_report(run_dir)
            if report is not None:
                report["status"] = "reused"
                report["output_dir"] = str(run_dir)
                rows.append(report)
                print(f"{tag}: reused (summary.json exists)")
                _write_master(output_root, rows)
                continue

        cmd = [
            sys.executable,
            str(CLI),
            "--workflow",
            workflow,
            "--size",
            size,
            "--budget",
            str(budget),
            "--seed",
            str(args.seed),
            "--n-workers",
            str(args.n_workers),
            "--output-dir",
            str(run_dir),
        ]
        if args.timeout_sec is not None:
            cmd += ["--timeout-sec", str(args.timeout_sec)]

        print(f"{tag}: running ...")
        completed = subprocess.run(cmd, cwd=str(SMAC3_DIR.parents[2]), check=False)

        report = _read_run_report(run_dir)
        if completed.returncode == 0 and report is not None:
            report["status"] = "ok"
            report["output_dir"] = str(run_dir)
            rows.append(report)
            print(
                f"{tag}: done  best={report.get('best_metric')} "
                f"speedup={report.get('speedup_vs_baseline')}"
            )
        else:
            status = "error" if completed.returncode != 0 else "no_report"
            rows.append(_blank_row(workflow, size, budget, status, run_dir))
            print(f"{tag}: FAILED (rc={completed.returncode}, status={status})")

        # Persist incrementally so progress survives an interruption.
        _write_master(output_root, rows)

    master_path = _write_master(output_root, rows)
    ok = sum(1 for r in rows if r.get("status") in {"ok", "reused"})
    print(f"\n[done] {ok}/{len(rows)} runs available")
    print(f"[done] master report: {master_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
