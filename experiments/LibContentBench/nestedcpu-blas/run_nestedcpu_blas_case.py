#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.LibContentBench.libcontentbench.common import (
    load_json_object,
    performance_metrics_payload,
    write_json,
)
from experiments.LibContentBench.libcontentbench.monitoring import ProcessTreeSampler


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    label: str
    mode: str
    output_json: str
    matrix_size: int
    task_count: int
    repetitions: int
    outer_jobs: int
    blas_threads: int
    backend_prefer: str


DEFAULTS = {
    "case_id": "baseline_direct",
    "label": "baseline direct",
    "mode": "baseline",
    "output_json": "work/direct_result.json",
    "matrix_size": 320,
    "task_count": 8,
    "repetitions": 6,
    "outer_jobs": 6,
    "backend_prefer": "processes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one NestedCPU-BLAS case.")
    parser.add_argument("--case-id", default=DEFAULTS["case_id"])
    parser.add_argument("--label", default=DEFAULTS["label"])
    parser.add_argument("--mode", default=DEFAULTS["mode"])
    parser.add_argument("--output-json", default=DEFAULTS["output_json"])
    parser.add_argument("--matrix-size", type=int, default=DEFAULTS["matrix_size"])
    parser.add_argument("--task-count", type=int, default=DEFAULTS["task_count"])
    parser.add_argument("--repetitions", type=int, default=DEFAULTS["repetitions"])
    parser.add_argument("--outer-jobs", "--joblib-n-jobs", dest="outer_jobs", type=int, default=DEFAULTS["outer_jobs"])
    parser.add_argument("--blas-threads", type=int, default=None)
    parser.add_argument("--backend-prefer", choices=["processes", "threads"], default=DEFAULTS["backend_prefer"])
    return parser.parse_args()


def load_case(path: Path) -> CaseConfig:
    payload = load_json_object(path)
    return CaseConfig(
        case_id=str(payload["case_id"]),
        label=str(payload.get("label", payload["case_id"])),
        mode=str(payload.get("mode", "unknown")),
        output_json=str(payload["output_json"]),
        matrix_size=int(payload["matrix_size"]),
        task_count=int(payload["task_count"]),
        repetitions=int(payload["repetitions"]),
        outer_jobs=int(payload["outer_jobs"]),
        blas_threads=int(payload["blas_threads"]),
        backend_prefer=str(payload.get("backend_prefer", "processes")),
    )


def config_from_args(args: argparse.Namespace) -> CaseConfig:
    return CaseConfig(
        case_id=str(args.case_id),
        label=str(args.label),
        mode=str(args.mode),
        output_json=str(args.output_json),
        matrix_size=int(args.matrix_size),
        task_count=int(args.task_count),
        repetitions=int(args.repetitions),
        outer_jobs=int(args.outer_jobs),
        blas_threads=resolve_blas_threads(args.blas_threads),
        backend_prefer=str(args.backend_prefer),
    )


def resolve_blas_threads(explicit_threads: int | None) -> int:
    if explicit_threads is not None:
        if explicit_threads < 1:
            raise ValueError("--blas-threads must be >= 1")
        return explicit_threads
    for env_name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = int(raw)
        if value < 1:
            raise ValueError(f"{env_name} must be >= 1, got {value}")
        return value
    return 1


def apply_thread_env(threads: int) -> dict[str, str]:
    env_updates = {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "BLIS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }
    os.environ.update(env_updates)
    return env_updates


def current_thread_env() -> dict[str, str | None]:
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "BLIS_NUM_THREADS": os.environ.get("BLIS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def worker(seed: int, matrix_size: int, repetitions: int) -> float:
    np = importlib.import_module("numpy")

    rng = np.random.default_rng(seed)
    left = rng.standard_normal((matrix_size, matrix_size), dtype=np.float64)
    right = rng.standard_normal((matrix_size, matrix_size), dtype=np.float64)
    result = left
    checksum = 0.0
    for _ in range(repetitions):
        result = result @ right
        checksum += float(result[0, 0])
    checksum += float(result.diagonal()[: min(8, matrix_size)].sum())
    return checksum


def run_case(config: CaseConfig, case_dir: Path) -> dict[str, Any]:
    thread_env_updates = current_thread_env()
    if int(thread_env_updates.get("OPENBLAS_NUM_THREADS") or 0) != config.blas_threads:
        thread_env_updates = apply_thread_env(config.blas_threads)
    effective_thread_env_path = case_dir / "effective_env.json"
    write_json(effective_thread_env_path, thread_env_updates)

    importlib.import_module("numpy")
    joblib = importlib.import_module("joblib")
    threadpoolctl = importlib.import_module("threadpoolctl")

    sampler = ProcessTreeSampler(os.getpid(), case_dir / "metrics_timeseries.csv")
    sampler.start()
    started = time.perf_counter()
    checksums = joblib.Parallel(n_jobs=config.outer_jobs, prefer=config.backend_prefer)(
        joblib.delayed(worker)(seed, config.matrix_size, config.repetitions)
        for seed in range(config.task_count)
    )
    wall_time = time.perf_counter() - started
    sampler_summary = sampler.stop()
    checksum = float(sum(checksums))
    throughput = float(config.task_count) / wall_time if wall_time > 0 else None

    return {
        "benchmark": "nestedcpu-blas",
        "resource": "CPU / scheduler",
        "contending_libraries": ["joblib", "numpy", "blas"],
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "status": "ok",
        "success": True,
        "wall_time_sec": wall_time,
        "throughput": throughput,
        "latency_p95": None,
        "resource_signal_primary": sampler_summary["context_switches_per_sec"],
        "resource_signal_secondary": sampler_summary["peak_threads"],
        "config_summary": f"outer_jobs={config.outer_jobs}, blas_threads={config.blas_threads}",
        "notes": "Real nested joblib + BLAS thread contention.",
        "checksum": checksum,
        "matrix_size": config.matrix_size,
        "task_count": config.task_count,
        "repetitions": config.repetitions,
        "outer_jobs": config.outer_jobs,
        "blas_threads": config.blas_threads,
        "threadpool_info": threadpoolctl.threadpool_info(),
        "sampler_summary": sampler_summary,
        "config_snapshot": asdict(config),
        "blas_env": thread_env_updates,
        "effective_env_path": str(effective_thread_env_path),
    }


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    output_json = Path(config.output_json).resolve()
    case_dir = output_json.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_case(config, case_dir)
    except Exception as exc:  # pragma: no cover - exercised in subprocess
        payload = {
            "benchmark": "nestedcpu-blas",
            "case_id": config.case_id,
            "label": config.label,
            "mode": config.mode,
            "status": "error",
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    write_json(output_json, payload)
    print(json.dumps(performance_metrics_payload(payload), sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
