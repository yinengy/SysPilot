#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.LibContentBench.libcontentbench.common import default_run_label
from experiments.LibContentBench.libcontentbench.manifest import BenchmarkEntry, load_manifest


SUITE_ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = SUITE_ROOT / "suite_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LibContentBench suite workflows.")
    parser.add_argument("command", choices=[
        "prepare",
        "smoke",
        "run-baseline",
        "run-application-only",
        "run-library-aware",
        "run-oracle",
        "summarize",
    ])
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--run-label-prefix", default="suite")
    return parser


def select_benchmarks(entries: list[BenchmarkEntry], requested: list[str] | None, include_optional: bool) -> list[BenchmarkEntry]:
    requested_set = set(requested or [])
    selected: list[BenchmarkEntry] = []
    for entry in entries:
        if entry.optional and not include_optional and entry.benchmark_id not in requested_set:
            continue
        if requested is not None and entry.benchmark_id not in requested_set:
            continue
        selected.append(entry)
    return selected


def run_command(command: list[str], *, cwd: Path) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "cwd": str(cwd),
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def benchmark_python(entry: BenchmarkEntry) -> str:
    return str((SUITE_ROOT / entry.relative_path / entry.python_path).absolute())


def suite_run_label(prefix: str, entry: BenchmarkEntry, mode: str) -> str:
    return default_run_label(prefix=f"{prefix}_{entry.benchmark_id}_{mode}")


def dispatch(entry: BenchmarkEntry, suite_command: str, run_label_prefix: str) -> dict[str, object]:
    workspace = (SUITE_ROOT / entry.relative_path).resolve()
    if suite_command == "prepare":
        return run_command(["bash", "prepare.sh"], cwd=workspace)
    if suite_command == "summarize":
        return run_command([benchmark_python(entry), "build_report.py"], cwd=workspace)

    matrix = entry.smoke_matrix if suite_command == "smoke" else entry.main_matrix
    case_tag_map = {
        "run-baseline": "baseline",
        "run-application-only": "application_only",
        "run-library-aware": "library_aware",
        "run-oracle": "oracle",
    }
    command = [
        benchmark_python(entry),
        "run_matrix.py",
        "--matrix",
        matrix,
        "--python",
        benchmark_python(entry),
        "--run-label",
        suite_run_label(run_label_prefix, entry, suite_command),
    ]
    case_tag = case_tag_map.get(suite_command)
    if case_tag is not None:
        command.extend(["--case-tag", case_tag])
    return run_command(command, cwd=workspace)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = load_manifest(MANIFEST_PATH)
    selected = select_benchmarks(entries, args.benchmarks, args.include_optional)
    results = [dispatch(entry, args.command, args.run_label_prefix) for entry in selected]
    if args.command == "summarize":
        suite_summary = run_command([sys.executable, "build_suite_report.py"], cwd=SUITE_ROOT)
        results.append(suite_summary)
    print(json.dumps({"command": args.command, "results": results}, indent=2, sort_keys=True))
    return 0 if all(isinstance(result.get("exit_code"), int) and result["exit_code"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
