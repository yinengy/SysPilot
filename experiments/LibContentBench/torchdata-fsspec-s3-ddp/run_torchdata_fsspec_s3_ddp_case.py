#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
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
    case_output_dir: str
    runtime_root: str
    world_size: int
    samples_per_rank: int
    shards_per_rank: int
    input_dim: int
    hidden_dim: int
    num_classes: int
    batch_size: int
    epochs: int
    learning_rate: float
    torch_num_threads: int
    s3_block_size: int
    s3_cache_type: str
    s3_use_listings_cache: bool
    ddp_bucket_cap_mb: int
    timeout_seconds: int
    minio_image: str


INT_ENV_OVERRIDES: dict[str, tuple[str, int]] = {
    "torch_num_threads": ("BENCHMARK_TORCH_NUM_THREADS", 1),
    "s3_block_size": ("BENCHMARK_S3_BLOCK_SIZE", 4096),
    "ddp_bucket_cap_mb": ("BENCHMARK_DDP_BUCKET_CAP_MB", 1),
}

BOOL_ENV_OVERRIDES: dict[str, str] = {
    "s3_use_listings_cache": "BENCHMARK_S3_USE_LISTINGS_CACHE",
}

STR_ENV_OVERRIDES: dict[str, set[str]] = {
    "s3_cache_type": {"none", "bytes", "readahead", "mmap"},
}


DEFAULTS = {
    "case_id": "baseline_direct",
    "label": "baseline direct",
    "mode": "baseline",
    "output_json": "work/direct_result.json",
    "case_output_dir": "work/direct_case_output",
    "minio_image": "quay.io/minio/minio:latest",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TorchData fsspec S3 DDP case.")
    parser.add_argument("--case-config")
    parser.add_argument("--case-id", default=DEFAULTS["case_id"])
    parser.add_argument("--label", default=DEFAULTS["label"])
    parser.add_argument("--mode", default=DEFAULTS["mode"])
    parser.add_argument("--output-json", default=DEFAULTS["output_json"])
    parser.add_argument("--case-output-dir", default=DEFAULTS["case_output_dir"])
    parser.add_argument("--runtime-root")
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--samples-per-rank", type=int)
    parser.add_argument("--shards-per-rank", type=int)
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--minio-image", default=DEFAULTS["minio_image"])
    return parser.parse_args(argv)


def load_case(path: Path) -> CaseConfig:
    payload = load_json_object(path)
    return CaseConfig(
        case_id=str(payload["case_id"]),
        label=str(payload.get("label", payload["case_id"])),
        mode=str(payload.get("mode", "unknown")),
        output_json=str(payload["output_json"]),
        case_output_dir=str(payload["case_output_dir"]),
        runtime_root=str(payload["runtime_root"]),
        world_size=int(payload["world_size"]),
        samples_per_rank=int(payload["samples_per_rank"]),
        shards_per_rank=int(payload["shards_per_rank"]),
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_classes=int(payload["num_classes"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        torch_num_threads=int(payload["torch_num_threads"]),
        s3_block_size=int(payload["s3_block_size"]),
        s3_cache_type=str(payload["s3_cache_type"]),
        s3_use_listings_cache=bool(payload["s3_use_listings_cache"]),
        ddp_bucket_cap_mb=int(payload["ddp_bucket_cap_mb"]),
        timeout_seconds=int(payload["timeout_seconds"]),
        minio_image=str(payload.get("minio_image", "quay.io/minio/minio:latest")),
    )


def config_from_args(args: argparse.Namespace) -> CaseConfig:
    missing = [
        option
        for option, value in (
            ("--runtime-root", args.runtime_root),
            ("--world-size", args.world_size),
            ("--samples-per-rank", args.samples_per_rank),
            ("--shards-per-rank", args.shards_per_rank),
            ("--input-dim", args.input_dim),
            ("--hidden-dim", args.hidden_dim),
            ("--num-classes", args.num_classes),
            ("--batch-size", args.batch_size),
            ("--epochs", args.epochs),
            ("--learning-rate", args.learning_rate),
            ("--timeout-seconds", args.timeout_seconds),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Direct TorchData fsspec S3 DDP runs require " + ", ".join(missing) + " when --case-config is not provided"
        )
    return CaseConfig(
        case_id=str(args.case_id),
        label=str(args.label),
        mode=str(args.mode),
        output_json=str(args.output_json),
        case_output_dir=str(args.case_output_dir),
        runtime_root=str(args.runtime_root),
        world_size=int(args.world_size),
        samples_per_rank=int(args.samples_per_rank),
        shards_per_rank=int(args.shards_per_rank),
        input_dim=int(args.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_classes=int(args.num_classes),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        torch_num_threads=4,
        s3_block_size=65536,
        s3_cache_type="none",
        s3_use_listings_cache=False,
        ddp_bucket_cap_mb=1,
        timeout_seconds=int(args.timeout_seconds),
        minio_image=str(args.minio_image),
    )


def _parse_bool_env(name: str, raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


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
    for field_name, env_name in BOOL_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = _parse_bool_env(env_name, raw)
        updates[field_name] = value
        applied_env[env_name] = value
    for field_name, allowed in STR_ENV_OVERRIDES.items():
        env_name = f"BENCHMARK_{field_name.upper()}"
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value not in allowed:
            raise ValueError(f"{env_name} must be one of {sorted(allowed)}, got {value!r}")
        updates[field_name] = value
        applied_env[env_name] = value
    if not updates:
        return config, {}
    return replace(config, **updates), applied_env


def load_readiness(workspace_root: Path) -> dict[str, Any]:
    readiness_path = workspace_root / "env/readiness.json"
    if not readiness_path.exists():
        return {"ready": False, "notes": "prepare.sh has not been run yet"}
    return load_json_object(readiness_path)


def parse_cache_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"none", "bytes", "readahead", "mmap"}:
        raise ValueError(f"Unsupported s3_cache_type: {value!r}")
    return normalized


def import_datapipe_iter() -> Any:
    try:
        return importlib.import_module("torchdata.datapipes.iter")
    except ModuleNotFoundError:
        return importlib.import_module("torch.utils.data.datapipes.iter")


def dataset_root_for(config: CaseConfig) -> Path:
    return (
        Path(config.runtime_root)
        / "datasets"
        / (
            f"world{config.world_size}_samples{config.samples_per_rank}_shards{config.shards_per_rank}_"
            f"input{config.input_dim}_classes{config.num_classes}"
        )
    )


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def check_docker_available() -> bool:
    return shutil.which("docker") is not None


def wait_for_http_ok(url: str, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error or 'unknown error'}")


def run_command(command: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def blocked_result(config: CaseConfig, *, readiness: dict[str, Any], env_overrides: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "benchmark": "torchdata-fsspec-s3-ddp",
        "resource": "object-store input pipeline + single-host DDP",
        "contending_libraries": ["pytorch.datapipes", "fsspec", "s3fs", "minio", "pytorch.ddp"],
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "status": "blocked",
        "success": False,
        "throughput": None,
        "wall_time_sec": None,
        "samples_per_sec": None,
        "steps_per_sec": None,
        "resource_signal_primary": None,
        "resource_signal_secondary": None,
        "config_summary": f"threads={config.torch_num_threads},s3_block={config.s3_block_size},cache={config.s3_cache_type},bucket_cap_mb={config.ddp_bucket_cap_mb}",
        "notes": notes,
        "readiness": readiness,
        "config_snapshot": asdict(config),
        "env_overrides": env_overrides,
    }


def ensure_local_dataset(config: CaseConfig, np_module: Any) -> tuple[list[list[Path]], int, int]:
    if config.samples_per_rank % config.shards_per_rank != 0:
        raise ValueError("samples_per_rank must be divisible by shards_per_rank")
    dataset_root = dataset_root_for(config)
    dataset_root.mkdir(parents=True, exist_ok=True)
    shard_rows = config.samples_per_rank // config.shards_per_rank
    record_dtype = np_module.dtype([("features", np_module.float32, (config.input_dim,)), ("label", np_module.int32)])
    shard_groups: list[list[Path]] = []
    for rank in range(config.world_size):
        rank_paths: list[Path] = []
        for shard_index in range(config.shards_per_rank):
            shard_path = dataset_root / f"rank{rank:02d}_shard{shard_index:02d}.bin"
            rank_paths.append(shard_path)
            if shard_path.exists():
                continue
            rng = np_module.random.default_rng(rank * 1000 + shard_index)
            features = rng.standard_normal((shard_rows, config.input_dim), dtype=np_module.float32)
            class_weights = np_module.linspace(0.25, 1.25, config.input_dim, dtype=np_module.float32)
            scores = features @ class_weights
            labels = np_module.mod(np_module.floor(np_module.abs(scores) * 13).astype(np_module.int32), config.num_classes)
            records = np_module.zeros(shard_rows, dtype=record_dtype)
            records["features"] = features
            records["label"] = labels
            shard_path.write_bytes(records.tobytes())
        shard_groups.append(rank_paths)
    manifest = {
        "world_size": config.world_size,
        "samples_per_rank": config.samples_per_rank,
        "shards_per_rank": config.shards_per_rank,
        "rows_per_shard": shard_rows,
        "input_dim": config.input_dim,
        "num_classes": config.num_classes,
        "record_bytes": int(record_dtype.itemsize),
    }
    write_json(dataset_root / "manifest.json", manifest)
    return shard_groups, shard_rows, directory_size_bytes(dataset_root)


def ensure_minio_container(config: CaseConfig, case_dir: Path) -> dict[str, Any]:
    if not check_docker_available():
        raise RuntimeError("docker binary is unavailable")
    port = find_free_port()
    container_name = f"lcb-tfds3ddp-{hashlib.sha1((config.case_id + str(os.getpid())).encode('utf-8')).hexdigest()[:10]}"
    data_root = case_dir / "minio-data"
    data_root.mkdir(parents=True, exist_ok=True)
    inspect = run_command(["docker", "image", "inspect", config.minio_image], timeout=30)
    if inspect.returncode != 0:
        pull = run_command(["docker", "pull", config.minio_image], timeout=600)
        if pull.returncode != 0:
            raise RuntimeError(f"failed to pull {config.minio_image}: {pull.stderr.strip() or pull.stdout.strip()}")
    command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "-p",
        f"127.0.0.1:{port}:9000",
        "-e",
        "MINIO_ROOT_USER=minioadmin",
        "-e",
        "MINIO_ROOT_PASSWORD=minioadmin123",
        "-v",
        f"{data_root}:/data",
        config.minio_image,
        "server",
        "/data",
    ]
    started = run_command(command, timeout=60)
    if started.returncode != 0:
        raise RuntimeError(f"failed to start MinIO: {started.stderr.strip() or started.stdout.strip()}")
    endpoint = f"http://127.0.0.1:{port}"
    wait_for_http_ok(f"{endpoint}/minio/health/live", timeout_seconds=60.0)
    return {
        "container_name": container_name,
        "endpoint_url": endpoint,
        "access_key": "minioadmin",
        "secret_key": "minioadmin123",
        "bucket": f"lcb-{hashlib.sha1(config.case_id.encode('utf-8')).hexdigest()[:12]}",
        "port": port,
    }


def stop_minio_container(container_name: str) -> None:
    run_command(["docker", "rm", "-f", container_name], timeout=30)


def upload_dataset(
    *,
    s3fs_module: Any,
    config: CaseConfig,
    minio: dict[str, Any],
    shard_groups: list[list[Path]],
) -> tuple[list[list[str]], int]:
    setup_write_block_size = max(config.s3_block_size, 5 * 1024 * 1024)
    fs = s3fs_module.S3FileSystem(
        anon=False,
        key=minio["access_key"],
        secret=minio["secret_key"],
        use_ssl=False,
        client_kwargs={"endpoint_url": minio["endpoint_url"]},
        default_block_size=setup_write_block_size,
        default_cache_type=config.s3_cache_type,
        use_listings_cache=config.s3_use_listings_cache,
        skip_instance_cache=True,
    )
    bucket = str(minio["bucket"])
    if not fs.exists(bucket):
        fs.mkdir(bucket)
    remote_groups: list[list[str]] = []
    total_uploaded = 0
    for rank, rank_paths in enumerate(shard_groups):
        remote_rank_paths: list[str] = []
        for shard_index, local_path in enumerate(rank_paths):
            remote_path = f"s3://{bucket}/rank{rank:02d}/shard{shard_index:02d}.bin"
            with local_path.open("rb") as local_handle, fs.open(remote_path, "wb", block_size=setup_write_block_size) as remote_handle:
                payload = local_handle.read()
                remote_handle.write(payload)
                total_uploaded += len(payload)
            remote_rank_paths.append(remote_path)
        remote_groups.append(remote_rank_paths)
    return remote_groups, total_uploaded


def ddp_rank_worker(
    rank: int,
    config_payload: dict[str, Any],
    all_rank_paths: list[list[str]],
    case_output_dir: str,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    master_port: int,
    rows_per_shard: int,
) -> None:
    torch_module = importlib.import_module("torch")
    nn_module = importlib.import_module("torch.nn")
    optim_module = importlib.import_module("torch.optim")
    dist_module = importlib.import_module("torch.distributed")
    mp_module = importlib.import_module("torch.multiprocessing")
    np_module = importlib.import_module("numpy")
    s3fs_module = importlib.import_module("s3fs")
    torchdata_iter = import_datapipe_iter()

    config = CaseConfig(**config_payload)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    if hasattr(torch_module, "set_num_threads"):
        torch_module.set_num_threads(max(1, config.torch_num_threads))
    if hasattr(torch_module, "set_num_interop_threads"):
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch_module.manual_seed(rank)
    rank_dir = Path(case_output_dir) / f"rank_{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    dist_module.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=config.world_size,
    )

    record_dtype = np_module.dtype([("features", np_module.float32, (config.input_dim,)), ("label", np_module.int32)])
    batch_bytes = record_dtype.itemsize * config.batch_size
    fs = s3fs_module.S3FileSystem(
        anon=False,
        key=access_key,
        secret=secret_key,
        use_ssl=False,
        client_kwargs={"endpoint_url": endpoint_url},
        default_block_size=config.s3_block_size,
        default_cache_type=config.s3_cache_type,
        use_listings_cache=config.s3_use_listings_cache,
        skip_instance_cache=True,
    )

    class BatchReader:
        def __call__(self, path: str) -> list[tuple[Any, Any]]:
            batches: list[tuple[Any, Any]] = []
            with fs.open(path, "rb", block_size=config.s3_block_size, cache_type=config.s3_cache_type) as handle:
                while True:
                    payload = handle.read(batch_bytes)
                    if not payload:
                        break
                    if len(payload) % record_dtype.itemsize != 0:
                        raise ValueError(f"Corrupt shard payload for {path}: {len(payload)} bytes")
                    records = np_module.frombuffer(payload, dtype=record_dtype)
                    features = np_module.asarray(records["features"], dtype=np_module.float32).copy()
                    labels = np_module.asarray(records["label"], dtype=np_module.int64).copy()
                    batches.append((features, labels))
            return batches

    model = nn_module.Sequential(
        nn_module.Linear(config.input_dim, config.hidden_dim),
        nn_module.ReLU(),
        nn_module.Linear(config.hidden_dim, config.num_classes),
    )
    ddp_model = nn_module.parallel.DistributedDataParallel(
        model,
        device_ids=None,
        bucket_cap_mb=config.ddp_bucket_cap_mb,
    )
    optimizer = optim_module.SGD(ddp_model.parameters(), lr=config.learning_rate)
    criterion = nn_module.CrossEntropyLoss()

    step_durations: list[float] = []
    loss_trace: list[float] = []
    total_samples = 0
    step_count = 0
    started = time.perf_counter()
    success = False
    failure_error: str | None = None
    try:
        for _epoch in range(config.epochs):
            source_dp = torchdata_iter.IterableWrapper(list(all_rank_paths[rank]))
            batch_dp = source_dp.map(BatchReader()).unbatch()
            for features_np, targets_np in batch_dp:
                batch_started = time.perf_counter()
                features = torch_module.from_numpy(features_np)
                targets = torch_module.from_numpy(targets_np)
                optimizer.zero_grad(set_to_none=True)
                outputs = ddp_model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                step_durations.append(time.perf_counter() - batch_started)
                loss_trace.append(float(loss.detach().item()))
                step_count += 1
                total_samples += int(targets.shape[0])
        success = True
    except Exception as exc:
        failure_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = max(time.perf_counter() - started, 1e-9)
        metrics = {
            "rank": rank,
            "status": "success" if success else "error",
            "success": success,
            "samples": total_samples,
            "step_count": step_count,
            "elapsed_sec": elapsed,
            "steps_per_sec": step_count / elapsed if step_count > 0 else 0.0,
            "samples_per_sec": total_samples / elapsed if total_samples > 0 else 0.0,
            "loss_last": loss_trace[-1] if loss_trace else None,
            "error": failure_error,
            "rows_per_shard": rows_per_shard,
            "paths": list(all_rank_paths[rank]),
            "torch_num_threads": config.torch_num_threads,
            "s3_block_size": config.s3_block_size,
            "s3_cache_type": config.s3_cache_type,
            "ddp_bucket_cap_mb": config.ddp_bucket_cap_mb,
        }
        write_json(rank_dir / "metrics.json", metrics)
        if dist_module.is_initialized():
            dist_module.destroy_process_group()


def run_case(config: CaseConfig, case_dir: Path, env_overrides: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parent
    readiness = load_readiness(workspace_root)
    effective_env_path = case_dir / "effective_env_overrides.json"
    write_json(effective_env_path, env_overrides)
    if not readiness.get("ready"):
        return blocked_result(
            config,
            readiness=readiness,
            env_overrides=env_overrides,
        notes="PyTorch DataPipes, s3fs/fsspec, or Docker is unavailable in the workspace virtualenv, so the benchmark stays explicitly blocked.",
        )

    if config.world_size < 2:
        raise ValueError("world_size must be >= 2 for the DDP benchmark")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    np_module = importlib.import_module("numpy")
    _ = importlib.import_module("torch")
    _ = import_datapipe_iter()
    s3fs_module = importlib.import_module("s3fs")

    runtime_root = Path(config.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    shard_groups, rows_per_shard, dataset_bytes = ensure_local_dataset(config, np_module)

    minio: dict[str, Any] | None = None
    sampler = ProcessTreeSampler(os.getpid(), case_dir / "metrics_timeseries.csv")
    sampler.start()
    started = time.perf_counter()
    try:
        minio = ensure_minio_container(config, case_dir)
        remote_groups, uploaded_bytes = upload_dataset(
            s3fs_module=s3fs_module,
            config=config,
            minio=minio,
            shard_groups=shard_groups,
        )
        master_port = find_free_port()
        torch_mp = importlib.import_module("torch.multiprocessing")
        torch_mp.spawn(
            ddp_rank_worker,
            args=(asdict(config), remote_groups, str(case_dir), str(minio["endpoint_url"]), str(minio["access_key"]), str(minio["secret_key"]), master_port, rows_per_shard),
            nprocs=config.world_size,
            join=True,
        )
        worker_rows: list[dict[str, Any]] = []
        for metrics_path in sorted(case_dir.glob("rank_*/metrics.json")):
            payload = load_json_object(metrics_path)
            worker_rows.append(payload)
        if len(worker_rows) != config.world_size:
            raise RuntimeError(f"expected {config.world_size} rank metric files, found {len(worker_rows)}")
        total_samples = sum(int(row.get("samples", 0)) for row in worker_rows)
        total_steps = sum(int(row.get("step_count", 0)) for row in worker_rows)
        elapsed = max(time.perf_counter() - started, 1e-9)
        samples_per_sec = total_samples / elapsed
        steps_per_sec = total_steps / elapsed
        monitoring_summary = sampler.stop()
        return {
            "benchmark": "torchdata-fsspec-s3-ddp",
            "resource": "object-store input pipeline + single-host DDP",
            "contending_libraries": ["pytorch.datapipes", "fsspec", "s3fs", "minio", "pytorch.ddp"],
            "case_id": config.case_id,
            "label": config.label,
            "mode": config.mode,
            "status": "ok",
            "success": True,
            "world_size": config.world_size,
            "samples_per_rank": config.samples_per_rank,
            "shards_per_rank": config.shards_per_rank,
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "num_classes": config.num_classes,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "torch_num_threads": config.torch_num_threads,
            "s3_block_size": config.s3_block_size,
            "s3_cache_type": config.s3_cache_type,
            "s3_use_listings_cache": config.s3_use_listings_cache,
            "ddp_bucket_cap_mb": config.ddp_bucket_cap_mb,
            "wall_time_sec": elapsed,
            "throughput": samples_per_sec,
            "samples_per_sec": samples_per_sec,
            "steps_per_sec": steps_per_sec,
            "resource_signal_primary": "samples_per_sec",
            "resource_signal_secondary": "process_tree_read_bytes",
            "config_summary": (
                f"threads={config.torch_num_threads},s3_block={config.s3_block_size},cache={config.s3_cache_type},"
                f"listings_cache={config.s3_use_listings_cache},bucket_cap_mb={config.ddp_bucket_cap_mb}"
            ),
            "notes": (
                f"Single-host MinIO endpoint {minio['endpoint_url']} exercises real s3fs/fsspec reads over loopback; "
                f"uploaded_bytes={uploaded_bytes}, dataset_bytes={dataset_bytes}."
            ),
            "dataset_bytes": dataset_bytes,
            "uploaded_bytes": uploaded_bytes,
            "rows_per_shard": rows_per_shard,
            "readiness": readiness,
            "config_snapshot": asdict(config),
            "env_overrides": env_overrides,
            "effective_env_path": str(effective_env_path),
            "monitoring": monitoring_summary,
            "worker_metrics": worker_rows,
            "datapipe_iter_module": import_datapipe_iter().__name__,
            "minio": {
                "endpoint_url": minio["endpoint_url"],
                "bucket": minio["bucket"],
                "port": minio["port"],
            },
        }
    except Exception as exc:
        monitoring_summary = sampler.stop()
        status = "blocked" if "docker" in str(exc).lower() or "minio" in str(exc).lower() else "error"
        return {
            "benchmark": "torchdata-fsspec-s3-ddp",
            "resource": "object-store input pipeline + single-host DDP",
            "contending_libraries": ["pytorch.datapipes", "fsspec", "s3fs", "minio", "pytorch.ddp"],
            "case_id": config.case_id,
            "label": config.label,
            "mode": config.mode,
            "status": status,
            "success": False,
            "throughput": None,
            "wall_time_sec": None,
            "samples_per_sec": None,
            "steps_per_sec": None,
            "resource_signal_primary": None,
            "resource_signal_secondary": None,
            "config_summary": f"threads={config.torch_num_threads},s3_block={config.s3_block_size},cache={config.s3_cache_type},bucket_cap_mb={config.ddp_bucket_cap_mb}",
            "notes": f"{type(exc).__name__}: {exc}",
            "error": traceback.format_exc(),
            "readiness": readiness,
            "config_snapshot": asdict(config),
            "env_overrides": env_overrides,
            "effective_env_path": str(effective_env_path),
            "monitoring": monitoring_summary,
        }
    finally:
        if minio is not None:
            stop_minio_container(str(minio["container_name"]))


def main() -> int:
    args = parse_args()
    if args.case_config:
        case_path = Path(args.case_config).resolve()
        config = load_case(case_path)
    else:
        config = config_from_args(args)
    case_dir = Path(config.case_output_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    config = replace(config, s3_cache_type=parse_cache_type(config.s3_cache_type))
    config, env_overrides = apply_env_overrides(config)
    result = run_case(config, case_dir, env_overrides)
    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(performance_metrics_payload(result), sort_keys=True))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
