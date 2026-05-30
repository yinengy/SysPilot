#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
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
    runtime_root: str
    num_files: int
    rows_per_file: int
    row_group_size: int
    n_workers: int
    threads_per_worker: int
    worker_memory_limit: str
    arrow_cpu_count: int
    arrow_io_threads: int
    duckdb_memory_limit: str
    duckdb_threads: int
    path_layout: str


INT_ENV_OVERRIDES: dict[str, tuple[str, int]] = {
    "n_workers": ("BENCHMARK_N_WORKERS", 1),
    "threads_per_worker": ("BENCHMARK_THREADS_PER_WORKER", 1),
    "arrow_cpu_count": ("BENCHMARK_ARROW_CPU_COUNT", 1),
    "arrow_io_threads": ("BENCHMARK_ARROW_IO_THREADS", 1),
    "duckdb_threads": ("BENCHMARK_DUCKDB_THREADS", 1),
}

STRING_ENV_OVERRIDES: dict[str, str] = {
    "worker_memory_limit": "BENCHMARK_WORKER_MEMORY_LIMIT",
    "duckdb_memory_limit": "BENCHMARK_DUCKDB_MEMORY_LIMIT",
    "path_layout": "BENCHMARK_PATH_LAYOUT",
}


DEFAULTS = {
    "case_id": "baseline_direct",
    "label": "baseline direct",
    "mode": "baseline",
    "output_json": "work/direct_result.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Arrow/Dask/DuckDB memspill case.")
    parser.add_argument("--case-config")
    parser.add_argument("--case-id", default=DEFAULTS["case_id"])
    parser.add_argument("--label", default=DEFAULTS["label"])
    parser.add_argument("--mode", default=DEFAULTS["mode"])
    parser.add_argument("--output-json", default=DEFAULTS["output_json"])
    parser.add_argument("--runtime-root")
    parser.add_argument("--num-files", type=int)
    parser.add_argument("--rows-per-file", type=int)
    parser.add_argument("--row-group-size", type=int)
    return parser.parse_args(argv)


def load_case(path: Path) -> CaseConfig:
    payload = load_json_object(path)
    return CaseConfig(
        case_id=str(payload["case_id"]),
        label=str(payload.get("label", payload["case_id"])),
        mode=str(payload.get("mode", "unknown")),
        output_json=str(payload["output_json"]),
        runtime_root=str(payload["runtime_root"]),
        num_files=int(payload["num_files"]),
        rows_per_file=int(payload["rows_per_file"]),
        row_group_size=int(payload["row_group_size"]),
        n_workers=int(payload["n_workers"]),
        threads_per_worker=int(payload["threads_per_worker"]),
        worker_memory_limit=str(payload["worker_memory_limit"]),
        arrow_cpu_count=int(payload["arrow_cpu_count"]),
        arrow_io_threads=int(payload["arrow_io_threads"]),
        duckdb_memory_limit=str(payload["duckdb_memory_limit"]),
        duckdb_threads=int(payload["duckdb_threads"]),
        path_layout=str(payload.get("path_layout", "shared")),
    )


def config_from_args(args: argparse.Namespace) -> CaseConfig:
    missing = [
        option
        for option, value in (
            ("--runtime-root", args.runtime_root),
            ("--num-files", args.num_files),
            ("--rows-per-file", args.rows_per_file),
            ("--row-group-size", args.row_group_size),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Direct Arrow/Dask/DuckDB runs require " + ", ".join(missing) + " when --case-config is not provided"
        )
    return CaseConfig(
        case_id=str(args.case_id),
        label=str(args.label),
        mode=str(args.mode),
        output_json=str(args.output_json),
        runtime_root=str(args.runtime_root),
        num_files=int(args.num_files),
        rows_per_file=int(args.rows_per_file),
        row_group_size=int(args.row_group_size),
        n_workers=1,
        threads_per_worker=1,
        worker_memory_limit="256MB",
        arrow_cpu_count=1,
        arrow_io_threads=1,
        duckdb_memory_limit="128MB",
        duckdb_threads=1,
        path_layout="shared",
    )


def apply_env_overrides(config: CaseConfig) -> tuple[CaseConfig, dict[str, Any]]:
    updates: dict[str, Any] = {}
    applied_env: dict[str, Any] = {}
    for field_name, (env_name, minimum) in INT_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = int(raw)
        if value < minimum:
            raise ValueError(f"{env_name} must be >= {minimum}, got {value}")
        updates[field_name] = value
        applied_env[env_name] = value
    for field_name, env_name in STRING_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = raw.strip()
        if not value:
            raise ValueError(f"{env_name} cannot be empty")
        if field_name == "path_layout" and value not in {"shared", "isolated"}:
            raise ValueError(f"{env_name} must be 'shared' or 'isolated', got {value!r}")
        updates[field_name] = value
        applied_env[env_name] = value
    if not updates:
        return config, {}
    return replace(config, **updates), applied_env


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


class DirectoryPeakSampler:
    def __init__(self, paths: dict[str, Path], interval_sec: float = 0.05) -> None:
        self._paths = paths
        self._interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._peaks: dict[str, int] = {name: 0 for name in paths}

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_sec):
            self._sample_once()

    def _sample_once(self) -> None:
        for name, path in self._paths.items():
            size = directory_size_bytes(path)
            if size > self._peaks[name]:
                self._peaks[name] = size

    def start(self) -> None:
        self._sample_once()
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._stop_event.set()
        self._thread.join(timeout=max(self._interval_sec * 2.0, 0.2))
        self._sample_once()
        return dict(self._peaks)


def build_paths(config: CaseConfig) -> dict[str, Path]:
    runtime_root = Path(config.runtime_root)
    dataset_root = runtime_root / "datasets" / f"rows{config.rows_per_file}_files{config.num_files}"
    if config.path_layout == "shared":
        shared_root = runtime_root / "shared"
        return {
            "dataset_root": dataset_root,
            "dask_local_directory": shared_root / "dask_local",
            "duckdb_temp_directory": shared_root / "duckdb_temp",
        }
    isolated_root = runtime_root / "isolated"
    return {
        "dataset_root": dataset_root,
        "dask_local_directory": isolated_root / f"{config.case_id}_dask_local",
        "duckdb_temp_directory": isolated_root / f"{config.case_id}_duckdb_temp",
    }


def ensure_dataset(config: CaseConfig, dataset_root: Path) -> list[Path]:
    np = importlib.import_module("numpy")
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")

    dataset_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(dataset_root.glob("part-*.parquet"))
    if len(existing) == config.num_files:
        return existing
    for old in existing:
        old.unlink()

    for index in range(config.num_files):
        rng = np.random.default_rng(index)
        table = pa.table(
            {
                "row_id": np.arange(config.rows_per_file, dtype=np.int64),
                "bucket": rng.integers(0, 64, size=config.rows_per_file, dtype=np.int64),
                "value_a": rng.standard_normal(config.rows_per_file),
                "value_b": rng.standard_normal(config.rows_per_file),
                "value_c": rng.standard_normal(config.rows_per_file),
            }
        )
        pq.write_table(
            table,
            dataset_root / f"part-{index:03d}.parquet",
            row_group_size=config.row_group_size,
        )
    return sorted(dataset_root.glob("part-*.parquet"))


def query_one(
    parquet_path: str,
    *,
    arrow_cpu_count: int,
    arrow_io_threads: int,
    duckdb_memory_limit: str,
    duckdb_threads: int,
    duckdb_temp_directory: str,
) -> dict[str, float | int]:
    duckdb = importlib.import_module("duckdb")
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")

    pa.set_cpu_count(arrow_cpu_count)
    pa.set_io_thread_count(arrow_io_threads)
    table = pq.read_table(parquet_path)
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
    conn.execute(f"SET threads={duckdb_threads}")
    conn.execute(f"SET temp_directory='{duckdb_temp_directory}'")
    conn.register("input_table", table)
    row = conn.execute(
        """
        SELECT
          SUM(score) AS total_score,
          SUM(cnt) AS total_rows
        FROM (
          SELECT bucket, SUM(value_a * value_b) AS score, COUNT(*) AS cnt
          FROM input_table
          GROUP BY bucket
          ORDER BY score DESC
        )
        """
    ).fetchone()
    conn.close()
    total_score = float(row[0]) if row and row[0] is not None else 0.0
    total_rows = int(row[1]) if row and row[1] is not None else 0
    return {"total_score": total_score, "total_rows": total_rows}


def run_case(config: CaseConfig, case_dir: Path, env_overrides: dict[str, Any]) -> dict[str, Any]:
    distributed = importlib.import_module("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    effective_env_path = case_dir / "effective_env_overrides.json"
    write_json(effective_env_path, env_overrides)

    paths = build_paths(config)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    parquet_files = ensure_dataset(config, paths["dataset_root"])
    if paths["duckdb_temp_directory"].exists():
        shutil.rmtree(paths["duckdb_temp_directory"])
        paths["duckdb_temp_directory"].mkdir(parents=True, exist_ok=True)
    if paths["dask_local_directory"].exists():
        shutil.rmtree(paths["dask_local_directory"])
        paths["dask_local_directory"].mkdir(parents=True, exist_ok=True)

    sampler = ProcessTreeSampler(os.getpid(), case_dir / "metrics_timeseries.csv")
    directory_sampler = DirectoryPeakSampler(
        {
            "dask_local": paths["dask_local_directory"],
            "duckdb_temp": paths["duckdb_temp_directory"],
        }
    )
    sampler.start()
    directory_sampler.start()
    started = time.perf_counter()
    cluster = LocalCluster(
        n_workers=config.n_workers,
        threads_per_worker=config.threads_per_worker,
        processes=True,
        memory_limit=config.worker_memory_limit,
        local_directory=str(paths["dask_local_directory"]),
        dashboard_address=None,
    )
    client = Client(cluster)
    try:
        futures = [
            client.submit(
                query_one,
                str(parquet_path),
                arrow_cpu_count=config.arrow_cpu_count,
                arrow_io_threads=config.arrow_io_threads,
                duckdb_memory_limit=config.duckdb_memory_limit,
                duckdb_threads=config.duckdb_threads,
                duckdb_temp_directory=str(paths["duckdb_temp_directory"]),
            )
            for parquet_path in parquet_files
        ]
        results = client.gather(futures)
    finally:
        client.close()
        cluster.close()
    wall_time = time.perf_counter() - started
    sampler_summary = sampler.stop()
    directory_peaks = directory_sampler.stop()

    rows_processed = sum(int(result["total_rows"]) for result in results)
    checksum = float(sum(float(result["total_score"]) for result in results))
    throughput = rows_processed / wall_time if wall_time > 0 else None
    dask_local_peak_bytes = int(directory_peaks.get("dask_local", 0))
    duckdb_temp_peak_bytes = int(directory_peaks.get("duckdb_temp", 0))
    dask_local_final_bytes = directory_size_bytes(paths["dask_local_directory"])
    duckdb_temp_final_bytes = directory_size_bytes(paths["duckdb_temp_directory"])

    return {
        "benchmark": "arrow-dask-duckdb-memspill",
        "resource": "host memory + spill I/O",
        "contending_libraries": ["dask", "pyarrow", "duckdb"],
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "status": "ok",
        "success": True,
        "wall_time_sec": wall_time,
        "throughput": throughput,
        "latency_p95": None,
        "resource_signal_primary": dask_local_peak_bytes,
        "resource_signal_secondary": duckdb_temp_peak_bytes,
        "config_summary": (
            f"workers={config.n_workers}, arrow={config.arrow_cpu_count}/{config.arrow_io_threads}, "
            f"duckdb_threads={config.duckdb_threads}, duckdb_memory={config.duckdb_memory_limit}, layout={config.path_layout}"
        ),
        "notes": "Real parquet scans with Dask worker concurrency and DuckDB query execution.",
        "rows_processed": rows_processed,
        "checksum": checksum,
        "dataset_root": str(paths["dataset_root"]),
        "dask_local_directory": str(paths["dask_local_directory"]),
        "duckdb_temp_directory": str(paths["duckdb_temp_directory"]),
        "dask_local_peak_bytes": dask_local_peak_bytes,
        "duckdb_temp_peak_bytes": duckdb_temp_peak_bytes,
        "dask_local_bytes": dask_local_final_bytes,
        "duckdb_temp_bytes": duckdb_temp_final_bytes,
        "sampler_summary": sampler_summary,
        "config_snapshot": asdict(config),
        "env_overrides": env_overrides,
        "effective_env_path": str(effective_env_path),
    }


def main() -> int:
    args = parse_args()
    if args.case_config:
        case_config_path = Path(args.case_config).resolve()
        case_dir = case_config_path.parent
        config = load_case(case_config_path)
    else:
        config = config_from_args(args)
        output_json = Path(config.output_json).resolve()
        case_dir = output_json.parent
    config, env_overrides = apply_env_overrides(config)
    output_json = Path(config.output_json).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_case(config, case_dir, env_overrides)
    except Exception as exc:  # pragma: no cover - subprocess path
        payload = {
            "benchmark": "arrow-dask-duckdb-memspill",
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
