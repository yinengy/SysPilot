#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationArgs:
    stream_source: Path
    binary_path: Path
    manual: str
    baseline_threads: int | None
    baseline_proc_bind: str | None
    timeout: int
    max_iterations: int
    array_size: int
    ntimes: int
    require_improvement: bool
    json_out: Path | None


def parse_args() -> ValidationArgs:
    parser = argparse.ArgumentParser(
        description="Build STREAM with OpenMP and validate SysPilot tuning with a composed manual."
    )
    parser.add_argument(
        "--stream-source",
        type=Path,
        default=Path("/tmp/STREAM/stream.c"),
        help="Path to the official STREAM C source file",
    )
    parser.add_argument(
        "--binary-path",
        type=Path,
        default=Path("/tmp/STREAM/stream_omp"),
        help="Where to place the compiled STREAM binary",
    )
    parser.add_argument(
        "--manual",
        default="stream_openmp_numactl_composed",
        help="SysPilot manual to load during tune",
    )
    parser.add_argument(
        "--baseline-threads",
        type=int,
        help="Optional baseline OMP_NUM_THREADS override. Omit for true default baseline.",
    )
    parser.add_argument(
        "--baseline-proc-bind",
        help="Optional baseline OMP_PROC_BIND override. Omit for true default baseline.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--array-size", type=int, default=80000000)
    parser.add_argument("--ntimes", type=int, default=20)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Exit non-zero if SysPilot does not report a positive improvement.",
    )
    parser.add_argument("--json-out", type=Path)

    ns = parser.parse_args()
    return ValidationArgs(
        stream_source=ns.stream_source,
        binary_path=ns.binary_path,
        manual=str(ns.manual),
        baseline_threads=int(ns.baseline_threads) if ns.baseline_threads is not None else None,
        baseline_proc_bind=str(ns.baseline_proc_bind)
        if ns.baseline_proc_bind is not None
        else None,
        timeout=int(ns.timeout),
        max_iterations=int(ns.max_iterations),
        array_size=int(ns.array_size),
        ntimes=int(ns.ntimes),
        require_improvement=bool(ns.require_improvement),
        json_out=ns.json_out,
    )


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def build_stream(args: ValidationArgs) -> list[str]:
    args.binary_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "gcc",
        "-O3",
        "-fopenmp",
        f"-DSTREAM_ARRAY_SIZE={args.array_size}",
        f"-DNTIMES={args.ntimes}",
        str(args.stream_source),
        "-o",
        str(args.binary_path),
    ]
    result = _run(command, timeout=600)
    if result.returncode != 0:
        raise SystemExit(f"STREAM build failed:\n{result.stderr or result.stdout}")
    return command


def capture_topology() -> dict[str, str]:
    topology: dict[str, str] = {}
    for name, command in {
        "lscpu": ["lscpu"],
        "numactl_hardware": ["numactl", "--hardware"],
    }.items():
        result = _run(command, timeout=30)
        topology[name] = (
            result.stdout if result.returncode == 0 else (result.stderr or result.stdout)
        )
    return topology


def build_tune_command(args: ValidationArgs) -> tuple[str, list[str]]:
    env_parts: list[str] = []
    if args.baseline_threads is not None:
        env_parts.append(f"OMP_NUM_THREADS={args.baseline_threads}")
    if args.baseline_proc_bind is not None:
        env_parts.append(f"OMP_PROC_BIND={args.baseline_proc_bind}")

    if env_parts:
        workload_command = f"env {' '.join(env_parts)} {shlex.quote(str(args.binary_path))}"
    else:
        workload_command = str(args.binary_path)
    command = [
        sys.executable,
        "-m",
        "syspilot.cli",
        "tune",
        "--ui-mode",
        "plain",
        "--command",
        workload_command,
        "--manual",
        args.manual,
        "--timeout",
        str(args.timeout),
        "--max-iterations",
        str(args.max_iterations),
    ]
    return workload_command, command


def parse_tune_output(output: str) -> dict[str, object]:
    baseline_match = re.search(
        r"Round 0 Step 0: Initial Baseline Run.*?Wall time:\s*([0-9.]+)s", output, re.S
    )
    success_match = re.search(r"Success!\s+([0-9.]+)% improvement with actions:\s*(.+)", output)
    comparison_match = re.search(
        r"│ Wall Time\s*│\s*([0-9.]+)s\s*│\s*([0-9.]+)s\s*│\s*\+?([0-9.]+)%",
        output,
    )

    baseline_wall_s = float(baseline_match.group(1)) if baseline_match else None
    tuned_wall_s = float(comparison_match.group(2)) if comparison_match else None
    improvement_pct = None
    if success_match:
        improvement_pct = float(success_match.group(1))
    elif comparison_match:
        improvement_pct = float(comparison_match.group(3))
    elif baseline_wall_s is not None and tuned_wall_s is not None and baseline_wall_s > 0:
        improvement_pct = max(0.0, ((baseline_wall_s - tuned_wall_s) / baseline_wall_s) * 100.0)

    actions = success_match.group(2).strip() if success_match else ""
    if not actions:
        best_command_match = re.search(r"Best command:\s*(.+)", output)
        best_env_match = re.search(r"Best environment overrides:\s*(\{.+\})", output)
        action_parts: list[str] = []
        if best_command_match:
            action_parts.append(best_command_match.group(1).strip())
        if best_env_match:
            action_parts.append(best_env_match.group(1).strip())
        actions = " | ".join(action_parts)

    return {
        "baseline_wall_s": baseline_wall_s,
        "tuned_wall_s": tuned_wall_s,
        "improvement_pct": improvement_pct,
        "actions": actions,
        "success": improvement_pct is not None and improvement_pct > 0,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.stream_source.exists():
        raise SystemExit(f"STREAM source not found: {args.stream_source}")

    build_command = build_stream(args)
    topology = capture_topology()
    workload_command, tune_command = build_tune_command(args)

    started = time.monotonic()
    result = _run(tune_command, timeout=args.timeout + 600)
    elapsed_s = time.monotonic() - started

    parsed = parse_tune_output(result.stdout)
    payload: dict[str, object] = {
        "stream_source": str(args.stream_source),
        "binary_path": str(args.binary_path),
        "manual": args.manual,
        "build_command": build_command,
        "workload_command": workload_command,
        "tune_command": tune_command,
        "returncode": result.returncode,
        "elapsed_s": elapsed_s,
        "topology": topology,
        "parsed_result": parsed,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-2000:],
    }

    if args.json_out is not None:
        write_json(args.json_out, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))

    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if args.require_improvement and not bool(parsed["success"]):
        raise SystemExit("STREAM validation completed but did not report a positive improvement")


if __name__ == "__main__":
    main()
