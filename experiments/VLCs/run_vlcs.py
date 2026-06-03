#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import cast


CORE_COUNT = 24
WORKFLOW_IDS = (0, 1)
ROOT = Path(__file__).resolve().parent
DEFAULT_BINARY = ROOT / "bin" / "parallel_vlc"
DEFAULT_CONFIG = ROOT / "config" / "transformer.json"
VLC_END_TIME_PATTERN = re.compile(
    r"VLC\s+(?P<vlc_id>[12])\(transformer\)\s+end time:\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)s"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the packaged VLC transformer binary with explicit workflow "
            "assignment flags. Each provided --core-N value must be 0 or 1."
        )
    )
    for core_id in range(CORE_COUNT):
        _ = parser.add_argument(
            f"--core-{core_id}",
            dest=f"core_{core_id}",
            type=int,
            choices=WORKFLOW_IDS,
            default=None,
            metavar="{0,1}",
            help=(
                f"Assign physical core {core_id} to workflow 0 or 1. "
                "Omit this flag to leave the core unassigned."
            ),
        )
    _ = parser.add_argument(
        "--binary",
        default=str(DEFAULT_BINARY),
        help="Path to the parallel_vlc executable.",
    )
    _ = parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the VLC transformer JSON config.",
    )
    return parser


def _core_assignments(args: argparse.Namespace) -> dict[int, list[int]]:
    assignments: dict[int, list[int]] = {workflow_id: [] for workflow_id in WORKFLOW_IDS}
    for core_id in range(CORE_COUNT):
        workflow_id = cast(int | None, getattr(args, f"core_{core_id}"))
        if workflow_id is None:
            continue
        assignments[workflow_id].append(core_id)

    empty_workflows = [
        str(workflow_id)
        for workflow_id, cores in assignments.items()
        if not cores
    ]
    if empty_workflows:
        raise ValueError(
            "Each workflow must receive at least one core; empty workflow(s): "
            + ", ".join(empty_workflows)
        )
    return assignments


def _core_list(cores: list[int]) -> str:
    return ",".join(str(core_id) for core_id in sorted(cores))


def _parse_workflow_walltimes(output: str) -> tuple[float, float]:
    vlc_walltimes: dict[int, float] = {}
    for match in VLC_END_TIME_PATTERN.finditer(output):
        vlc_walltimes[int(match.group("vlc_id"))] = float(match.group("seconds"))
    if 1 not in vlc_walltimes or 2 not in vlc_walltimes:
        raise ValueError("Unable to parse VLC 1 and VLC 2 end times from program output")
    return vlc_walltimes[1], vlc_walltimes[2]


def run(args: argparse.Namespace) -> int:
    binary = Path(cast(str, args.binary)).resolve()
    config = Path(cast(str, args.config)).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    assignments = _core_assignments(args)
    workflow_0_cores = _core_list(assignments[0])
    workflow_1_cores = _core_list(assignments[1])
    command = [str(binary), str(config), workflow_0_cores, workflow_1_cores]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    walltime_sec = time.perf_counter() - started
    if completed.returncode != 0:
        _ = sys.stderr.write(completed.stdout)
        return completed.returncode

    workflow_0_walltime, workflow_1_walltime = _parse_workflow_walltimes(
        completed.stdout
    )
    json.dump(
        {
            "walltime": walltime_sec,
            "workflow_0_walltime": workflow_0_walltime,
            "workflow_1_walltime": workflow_1_walltime,
        },
        sys.stdout,
        separators=(",", ":"),
    )
    _ = sys.stdout.write("\n")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
