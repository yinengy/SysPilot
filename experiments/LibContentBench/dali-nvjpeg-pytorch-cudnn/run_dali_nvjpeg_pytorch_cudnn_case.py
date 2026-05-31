#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import shutil
import signal
import site
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def isolate_virtualenv_pythonpath() -> None:
    """Keep outer developer PYTHONPATH packages from shadowing venv CUDA wheels."""
    if sys.prefix == sys.base_prefix:
        return
    pythonpath_entries = {
        str(Path(entry).resolve())
        for entry in os.environ.pop("PYTHONPATH", "").split(os.pathsep)
        if entry
    }
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = ""
    if user_site:
        pythonpath_entries.add(str(Path(user_site).resolve()))
    repo_root = str(REPO_ROOT.resolve())
    filtered: list[str] = []
    for entry in sys.path:
        if not entry:
            filtered.append(entry)
            continue
        resolved = str(Path(entry).resolve())
        if resolved in pythonpath_entries and resolved != repo_root:
            continue
        filtered.append(entry)
    sys.path[:] = filtered


isolate_virtualenv_pythonpath()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.LibContentBench.libcontentbench.common import (  # noqa: E402
    performance_metrics_payload,
    write_json,
)

BENCHMARK_ID = "dali-nvjpeg-pytorch-cudnn"
BENCHMARK_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    label: str
    mode: str
    output_json: str
    case_output_dir: str
    runtime_root: str
    timeout_seconds: int
    warmup_seconds: float
    measure_seconds: float
    cuda_device: str
    dataset_images: int
    image_min_size: int
    image_max_size: int
    jpeg_quality: int
    num_classes: int
    resize_size: int
    crop_size: int
    dali_output_dtype: str
    dali_batch_size: int
    dali_prefetch_queue_depth: int
    dali_hw_decoder_load: float
    dali_decode_backend: str
    dali_num_threads: int
    dali_mps_share: int
    torch_mps_share: int
    torch_batch_size: int
    torch_model: str
    torch_precision: str
    torch_channels_last: bool
    torch_cudnn_benchmark: bool
    benchmark: str = BENCHMARK_ID


DEFAULTS: dict[str, Any] = {
    "case_id": "baseline_dali_b64_prefetch2_hw90_mps20_80",
    "label": "baseline DALI b64 prefetch2 hw90 MPS20/80",
    "mode": "baseline",
    "output_json": "work/direct_result.json",
    "case_output_dir": "work/direct_case_output",
    "runtime_root": "/var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn",
    "timeout_seconds": 180,
    "warmup_seconds": 10.0,
    "measure_seconds": 45.0,
    "cuda_device": "0",
    "dataset_images": 5000,
    "image_min_size": 768,
    "image_max_size": 1024,
    "jpeg_quality": 90,
    "num_classes": 16,
    "resize_size": 256,
    "crop_size": 224,
    "dali_output_dtype": "float16",
    "dali_batch_size": 64,
    "dali_prefetch_queue_depth": 2,
    "dali_hw_decoder_load": 0.9,
    "dali_decode_backend": "mixed",
    "dali_num_threads": 4,
    "dali_mps_share": 20,
    "torch_mps_share": 80,
    "torch_batch_size": 64,
    "torch_model": "custom_cnn",
    "torch_precision": "tf32",
    "torch_channels_last": True,
    "torch_cudnn_benchmark": True,
    "benchmark": BENCHMARK_ID,
}


ENV_SPECS: dict[str, tuple[str, type]] = {
    "dali_batch_size": ("BENCHMARK_DALI_BATCH_SIZE", int),
    "dali_prefetch_queue_depth": ("BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH", int),
    "dali_hw_decoder_load": ("BENCHMARK_DALI_HW_DECODER_LOAD", float),
    "dali_num_threads": ("BENCHMARK_DALI_NUM_THREADS", int),
    "dali_mps_share": ("BENCHMARK_DALI_MPS_SHARE", int),
    "torch_batch_size": ("BENCHMARK_TORCH_BATCH_SIZE", int),
    "torch_mps_share": ("BENCHMARK_TORCH_MPS_SHARE", int),
}


class _BooleanOptionalAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        default: bool | None = None,
        required: bool = False,
        help: str | None = None,
    ) -> None:
        expanded_options: list[str] = []
        for option in option_strings:
            expanded_options.append(option)
            if option.startswith("--"):
                expanded_options.append(f"--no-{option[2:]}")
        super().__init__(
            option_strings=expanded_options,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        del parser, values
        setattr(namespace, self.dest, not str(option_string).startswith("--no-"))


BOOLEAN_OPTIONAL_ACTION = getattr(argparse, "BooleanOptionalAction", _BooleanOptionalAction)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a DALI/nvJPEG + PyTorch/cuDNN GPU contention case."
    )
    parser.add_argument("--case-id", default=DEFAULTS["case_id"])
    parser.add_argument("--label", default=DEFAULTS["label"])
    parser.add_argument("--mode", default=DEFAULTS["mode"])
    parser.add_argument("--output-json", default=DEFAULTS["output_json"])
    parser.add_argument("--case-output-dir", default=DEFAULTS["case_output_dir"])
    parser.add_argument("--runtime-root", default=DEFAULTS["runtime_root"])
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULTS["timeout_seconds"])
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULTS["warmup_seconds"])
    parser.add_argument("--measure-seconds", type=float, default=DEFAULTS["measure_seconds"])
    parser.add_argument("--cuda-device", default=DEFAULTS["cuda_device"])
    parser.add_argument("--dataset-images", type=int, default=DEFAULTS["dataset_images"])
    parser.add_argument("--image-min-size", type=int, default=DEFAULTS["image_min_size"])
    parser.add_argument("--image-max-size", type=int, default=DEFAULTS["image_max_size"])
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULTS["jpeg_quality"])
    parser.add_argument("--num-classes", type=int, default=DEFAULTS["num_classes"])
    parser.add_argument("--resize-size", type=int, default=DEFAULTS["resize_size"])
    parser.add_argument("--crop-size", type=int, default=DEFAULTS["crop_size"])
    parser.add_argument("--dali-output-dtype", default=DEFAULTS["dali_output_dtype"])
    parser.add_argument("--dali-decode-backend", default=DEFAULTS["dali_decode_backend"])
    parser.add_argument("--torch-model", default=DEFAULTS["torch_model"])
    parser.add_argument("--torch-precision", default=DEFAULTS["torch_precision"])
    parser.add_argument(
        "--torch-channels-last",
        action=BOOLEAN_OPTIONAL_ACTION,
        default=DEFAULTS["torch_channels_last"],
    )
    parser.add_argument(
        "--torch-cudnn-benchmark",
        action=BOOLEAN_OPTIONAL_ACTION,
        default=DEFAULTS["torch_cudnn_benchmark"],
    )
    parser.add_argument(
        "--scaffold-only",
        action="store_true",
        help="Validate config and emit deterministic JSON without importing GPU libraries.",
    )
    parser.add_argument(
        "--worker", choices=["orchestrator", "dali", "torch"], default="orchestrator"
    )
    parser.add_argument("--worker-output-json", default="")
    parser.add_argument("--benchmark", default=DEFAULTS["benchmark"])
    return parser


def _coerce_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def config_from_args(args: argparse.Namespace) -> CaseConfig:
    return CaseConfig(
        case_id=str(args.case_id),
        label=str(args.label),
        mode=str(args.mode),
        output_json=str(Path(str(args.output_json)).resolve()),
        case_output_dir=str(Path(str(args.case_output_dir)).resolve()),
        runtime_root=str(Path(str(args.runtime_root)).resolve()),
        timeout_seconds=int(args.timeout_seconds),
        warmup_seconds=float(args.warmup_seconds),
        measure_seconds=float(args.measure_seconds),
        cuda_device=str(args.cuda_device),
        dataset_images=int(args.dataset_images),
        image_min_size=int(args.image_min_size),
        image_max_size=int(args.image_max_size),
        jpeg_quality=int(args.jpeg_quality),
        num_classes=int(args.num_classes),
        resize_size=int(args.resize_size),
        crop_size=int(args.crop_size),
        dali_output_dtype=str(args.dali_output_dtype),
        dali_batch_size=int(DEFAULTS["dali_batch_size"]),
        dali_prefetch_queue_depth=int(DEFAULTS["dali_prefetch_queue_depth"]),
        dali_hw_decoder_load=float(DEFAULTS["dali_hw_decoder_load"]),
        dali_decode_backend=str(args.dali_decode_backend),
        dali_num_threads=int(DEFAULTS["dali_num_threads"]),
        dali_mps_share=int(DEFAULTS["dali_mps_share"]),
        torch_mps_share=int(DEFAULTS["torch_mps_share"]),
        torch_batch_size=int(DEFAULTS["torch_batch_size"]),
        torch_model=str(args.torch_model),
        torch_precision=str(args.torch_precision),
        torch_channels_last=bool(args.torch_channels_last),
        torch_cudnn_benchmark=bool(args.torch_cudnn_benchmark),
        benchmark=str(args.benchmark),
    )


def apply_env_overrides(config: CaseConfig) -> tuple[CaseConfig, dict[str, Any]]:
    updates: dict[str, Any] = {}
    env_overrides: dict[str, Any] = {}
    for field_name, (env_name, caster) in ENV_SPECS.items():
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        value: Any = _coerce_bool(raw_value) if caster is bool else caster(raw_value)
        updates[field_name] = value
        env_overrides[env_name] = value
    updated = replace(config, **updates) if updates else config
    validate_config(updated)
    return updated, env_overrides


def validate_config(config: CaseConfig) -> None:
    if config.image_min_size > config.image_max_size:
        raise ValueError("image-min-size must be <= image-max-size")
    if config.crop_size > config.resize_size:
        raise ValueError("crop-size must be <= resize-size")
    if config.dali_decode_backend not in {"mixed", "cpu"}:
        raise ValueError("dali decode backend must be 'mixed' or 'cpu'")
    if config.dali_output_dtype not in {"float16", "float32"}:
        raise ValueError("dali output dtype must be float16 or float32")
    for name in ("dali_mps_share", "torch_mps_share"):
        value = int(getattr(config, name))
        if value < 1 or value > 100:
            raise ValueError(f"{name} must be in [1, 100]")
    for name in ("dataset_images", "num_classes", "dali_batch_size", "torch_batch_size"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be positive")


def case_paths(config: CaseConfig) -> dict[str, Path]:
    case_dir = Path(config.case_output_dir)
    runtime_root = Path(config.runtime_root)
    case_digest = hashlib.sha1(config.case_id.encode("utf-8")).hexdigest()[:10]
    mps_case_dir = Path("/tmp/lcb-dali-mps") / case_digest
    paths = {
        "case_dir": case_dir,
        "dataset_root": runtime_root / "dataset",
        "mps_pipe": mps_case_dir / "pipe",
        "mps_log": mps_case_dir / "log",
        "worker_dir": case_dir / "workers",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def readiness() -> dict[str, Any]:
    torch_ok = module_available("torch")
    cuda_ok = False
    if torch_ok:
        try:
            torch = importlib.import_module("torch")
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    mps_control = shutil.which("nvidia-cuda-mps-control")
    return {
        "python_executable": sys.executable,
        "torch_available": torch_ok,
        "cuda_available": cuda_ok,
        "dali_available": module_available("nvidia.dali"),
        "pillow_available": module_available("PIL"),
        "numpy_available": module_available("numpy"),
        "mps_control": mps_control,
        "ready": bool(torch_ok and cuda_ok and module_available("nvidia.dali") and mps_control),
    }


def blocked_result(
    config: CaseConfig, env_overrides: dict[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_ID,
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "success": False,
        "status": "blocked",
        "throughput": None,
        "score": None,
        "notes": reason,
        "readiness": readiness(),
        "config_snapshot": asdict(config),
        "env_overrides": env_overrides,
    }


def scaffold_result(config: CaseConfig, env_overrides: dict[str, Any]) -> dict[str, Any]:
    pair_dali = (
        float(config.dali_batch_size * 40)
        * max(0.05, config.dali_mps_share / 100.0)
        * 0.92
    )
    pair_torch = (
        float(config.torch_batch_size * 85)
        * max(0.05, config.torch_mps_share / 100.0)
        * 0.94
    )
    score = pair_dali + pair_torch
    return {
        "benchmark": BENCHMARK_ID,
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "success": True,
        "status": "scaffold_only",
        "implementation_status": "host_safe_scaffold_no_gpu_measurement",
        "throughput": score,
        "score": score,
        "pair_dali_images_per_sec": pair_dali,
        "pair_torch_images_per_sec": pair_torch,
        "wall_time_sec": 0.0,
        "resource_signal_primary": "gpu.compute/memory/copy shared under CUDA MPS",
        "config_summary": format_config_summary(config),
        "notes": (
            "Scaffold output validates command/config/report plumbing only; run without "
            "--scaffold-only on a CUDA+DALI host for measurements."
        ),
        "config_snapshot": asdict(config),
        "env_overrides": env_overrides,
    }


def format_config_summary(config: CaseConfig) -> str:
    return (
        f"DALI b{config.dali_batch_size} "
        f"prefetch{config.dali_prefetch_queue_depth} "
        f"hw{config.dali_hw_decoder_load:g} "
        f"MPS {config.dali_mps_share}/{config.torch_mps_share}"
    )


DIRECT_STDOUT_KEYS = (
    "wall_time_sec",
    "pair_torch_images_per_sec",
    "pair_dali_images_per_sec",
    "score",
)


def direct_stdout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in DIRECT_STDOUT_KEYS if key in payload}


def generate_dataset(config: CaseConfig, dataset_root: Path) -> None:
    marker = dataset_root / ".complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if int(payload.get("dataset_images", 0)) >= config.dataset_images:
            return
    try:
        np = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
    except Exception as exc:
        raise RuntimeError(
            f"Pillow/numpy unavailable for synthetic JPEG generation: {exc}"
        ) from exc
    dataset_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(12345)
    for index in range(config.dataset_images):
        class_dir = dataset_root / f"class{index % config.num_classes:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        path = class_dir / f"img_{index:06d}.jpg"
        if path.exists():
            continue
        height = rng.randint(config.image_min_size, config.image_max_size)
        width = rng.randint(config.image_min_size, config.image_max_size)
        array = np.random.default_rng(index).integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        image_module.fromarray(array, mode="RGB").save(
            path, format="JPEG", quality=config.jpeg_quality
        )
    write_json(marker, {"dataset_images": config.dataset_images, "num_classes": config.num_classes})


def run_dali_worker(config: CaseConfig, dataset_root: Path) -> dict[str, Any]:
    dali_module = importlib.import_module("nvidia.dali")
    fn = getattr(dali_module, "fn")
    pipeline_def = getattr(dali_module, "pipeline_def")
    types = getattr(dali_module, "types")

    dtype = types.FLOAT16 if config.dali_output_dtype == "float16" else types.FLOAT

    @pipeline_def(
        batch_size=config.dali_batch_size,
        num_threads=config.dali_num_threads,
        device_id=int(config.cuda_device),
        prefetch_queue_depth=config.dali_prefetch_queue_depth,
    )
    def image_pipe() -> Any:
        encoded, _labels = fn.readers.file(
            file_root=str(dataset_root), random_shuffle=True, name="Reader"
        )
        kwargs: dict[str, Any] = {"device": config.dali_decode_backend, "output_type": types.RGB}
        if config.dali_decode_backend == "mixed":
            kwargs["hw_decoder_load"] = config.dali_hw_decoder_load
        images = fn.decoders.image(encoded, **kwargs)
        images = fn.resize(images, resize_x=config.resize_size, resize_y=config.resize_size)
        return fn.crop_mirror_normalize(
            images,
            dtype=dtype,
            output_layout="CHW",
            crop=(config.crop_size, config.crop_size),
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        )

    pipe = image_pipe()
    pipe.build()
    deadline = time.time() + config.warmup_seconds
    while time.time() < deadline:
        pipe.run()
    start = time.time()
    deadline = start + config.measure_seconds
    batches = 0
    while time.time() < deadline:
        pipe.run()
        batches += 1
    elapsed = max(time.time() - start, 1e-9)
    images = batches * config.dali_batch_size
    return {
        "worker": "dali",
        "images": images,
        "elapsed_sec": elapsed,
        "images_per_sec": images / elapsed,
    }


class CustomCnnFactory:
    @staticmethod
    def build(torch: Any) -> Any:
        nn = torch.nn
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1000),
        )


def run_torch_worker(config: CaseConfig) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    device = torch.device(f"cuda:{config.cuda_device}")
    torch.backends.cuda.matmul.allow_tf32 = config.torch_precision == "tf32"
    torch.backends.cudnn.allow_tf32 = config.torch_precision == "tf32"
    torch.backends.cudnn.benchmark = config.torch_cudnn_benchmark
    torch.backends.cudnn.deterministic = False
    model = CustomCnnFactory.build(torch).to(device).eval()
    memory_format = torch.channels_last if config.torch_channels_last else torch.contiguous_format
    model = model.to(memory_format=memory_format)
    x = torch.randn(
        config.torch_batch_size,
        3,
        config.crop_size,
        config.crop_size,
        device=device,
    ).to(memory_format=memory_format)
    with torch.inference_mode():
        deadline = time.time() + config.warmup_seconds
        while time.time() < deadline:
            model(x)
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        images = 0
        start_event.record()
        wall_start = time.time()
        deadline = wall_start + config.measure_seconds
        while time.time() < deadline:
            model(x)
            images += config.torch_batch_size
        end_event.record()
        torch.cuda.synchronize(device)
    elapsed = max(start_event.elapsed_time(end_event) / 1000.0, 1e-9)
    return {
        "worker": "torch",
        "images": images,
        "elapsed_sec": elapsed,
        "images_per_sec": images / elapsed,
    }


def worker_main(config: CaseConfig, worker: str, output_json: Path) -> int:
    paths = case_paths(config)
    if worker == "dali":
        generate_dataset(config, paths["dataset_root"])
        payload = run_dali_worker(config, paths["dataset_root"])
    else:
        payload = run_torch_worker(config)
    write_json(output_json, payload)
    print(json.dumps(performance_metrics_payload(payload), sort_keys=True))
    return 0


def _base_worker_command(config: CaseConfig, worker: str, output_json: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--worker-output-json",
        str(output_json),
        "--case-id",
        config.case_id,
        "--label",
        config.label,
        "--mode",
        config.mode,
        "--output-json",
        config.output_json,
        "--case-output-dir",
        config.case_output_dir,
        "--runtime-root",
        config.runtime_root,
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--warmup-seconds",
        str(config.warmup_seconds),
        "--measure-seconds",
        str(config.measure_seconds),
        "--cuda-device",
        config.cuda_device,
        "--dataset-images",
        str(config.dataset_images),
        "--image-min-size",
        str(config.image_min_size),
        "--image-max-size",
        str(config.image_max_size),
        "--jpeg-quality",
        str(config.jpeg_quality),
        "--num-classes",
        str(config.num_classes),
        "--resize-size",
        str(config.resize_size),
        "--crop-size",
        str(config.crop_size),
        "--dali-output-dtype",
        config.dali_output_dtype,
        "--dali-decode-backend",
        config.dali_decode_backend,
        "--torch-model",
        config.torch_model,
        "--torch-precision",
        config.torch_precision,
        "--benchmark",
        config.benchmark,
        "--torch-channels-last" if config.torch_channels_last else "--no-torch-channels-last",
        "--torch-cudnn-benchmark" if config.torch_cudnn_benchmark else "--no-torch-cudnn-benchmark",
    ]


def _base_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    driver_lib_dir = BENCHMARK_ROOT / "artifact" / "driver_lib"
    if driver_lib_dir.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            str(driver_lib_dir) if not current else str(driver_lib_dir) + os.pathsep + current
        )
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _configured_subprocess_env(config: CaseConfig) -> dict[str, str]:
    env = _base_subprocess_env()
    for field_name, (env_name, _caster) in ENV_SPECS.items():
        env[env_name] = str(getattr(config, field_name))
    return env


def _worker_env(config: CaseConfig, paths: dict[str, Path], share: int) -> dict[str, str]:
    env = _configured_subprocess_env(config)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": config.cuda_device,
            "CUDA_MPS_PIPE_DIRECTORY": str(paths["mps_pipe"]),
            "CUDA_MPS_LOG_DIRECTORY": str(paths["mps_log"]),
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(share),
            "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING": "1",
        }
    )
    return env


def start_mps(config: CaseConfig, paths: dict[str, Path]) -> bool:
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False
    terminate_user_mps_servers(set())
    reset_mps_directories(paths)
    env = _worker_env(config, paths, 100)
    completed = subprocess.run(
        ["nvidia-cuda-mps-control", "-d"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True
    terminate_user_mps_servers(set())
    completed = subprocess.run(
        ["nvidia-cuda-mps-control", "-d"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(format_mps_start_error(completed, paths))
    return True


def reset_mps_directories(paths: dict[str, Path]) -> None:
    for path_name in ("mps_pipe", "mps_log"):
        path = paths[path_name]
        path.mkdir(parents=True, exist_ok=True)
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)


def format_mps_start_error(
    completed: subprocess.CompletedProcess[str], paths: dict[str, Path]
) -> str:
    parts = [
        f"nvidia-cuda-mps-control -d failed with exit code {completed.returncode}",
    ]
    if completed.stdout.strip():
        parts.append(f"stdout: {completed.stdout.strip()[-2000:]}")
    if completed.stderr.strip():
        parts.append(f"stderr: {completed.stderr.strip()[-2000:]}")
    for log_name in ("control.log", "server.log"):
        log_path = paths["mps_log"] / log_name
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace").strip()
            if log_text:
                parts.append(f"{log_name}: {log_text[-2000:]}")
    return "\n".join(parts)


def current_user_mps_server_pids() -> set[int]:
    try:
        completed = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-x", "nvidia-cuda-mps"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()
    if completed.returncode not in {0, 1}:
        return set()
    return {int(line) for line in completed.stdout.splitlines() if line.strip().isdigit()}


def terminate_user_mps_servers(keep_pids: set[int]) -> None:
    for pid in current_user_mps_server_pids() - keep_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not (current_user_mps_server_pids() - keep_pids):
            return
        time.sleep(0.1)


def stop_mps(
    config: CaseConfig,
    paths: dict[str, Path],
    before_pids: set[int] | None = None,
) -> None:
    if shutil.which("nvidia-cuda-mps-control") is None:
        return
    env = _worker_env(config, paths, 100)
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        capture_output=True,
        check=False,
    )
    if before_pids is None:
        return
    time.sleep(2.0)
    terminate_user_mps_servers(before_pids)


def run_pair(
    config: CaseConfig, before_mps_pids: set[int] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = case_paths(config)
    initial_mps_pids = (
        current_user_mps_server_pids() if before_mps_pids is None else before_mps_pids
    )
    if not start_mps(config, paths):
        raise RuntimeError("nvidia-cuda-mps-control not available")
    try:
        dali_json = paths["worker_dir"] / "dali_pair.json"
        torch_json = paths["worker_dir"] / "torch_pair.json"
        dali_proc = subprocess.Popen(
            _base_worker_command(config, "dali", dali_json),
            env=_worker_env(config, paths, config.dali_mps_share),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        torch_proc = subprocess.Popen(
            _base_worker_command(config, "torch", torch_json),
            env=_worker_env(config, paths, config.torch_mps_share),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        dali_out, dali_err = dali_proc.communicate(timeout=config.timeout_seconds)
        torch_out, torch_err = torch_proc.communicate(timeout=config.timeout_seconds)
        if dali_proc.returncode != 0:
            raise RuntimeError(f"DALI pair worker failed: {dali_err or dali_out}")
        if torch_proc.returncode != 0:
            raise RuntimeError(f"Torch pair worker failed: {torch_err or torch_out}")
        return (
            json.loads(dali_json.read_text(encoding="utf-8")),
            json.loads(torch_json.read_text(encoding="utf-8")),
        )
    finally:
        stop_mps(config, paths, initial_mps_pids)


def write_timeseries(case_dir: Path, payload: dict[str, Any]) -> None:
    path = case_dir / "metrics_timeseries.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in (
            "wall_time_sec",
            "pair_dali_images_per_sec",
            "pair_torch_images_per_sec",
            "score",
        ):
            writer.writerow({"metric": key, "value": payload.get(key, "")})


def run_benchmark(config: CaseConfig, env_overrides: dict[str, Any]) -> dict[str, Any]:
    ready = readiness()
    if not ready["ready"]:
        return blocked_result(
            config, env_overrides, f"GPU benchmark dependencies unavailable: {ready}"
        )
    paths = case_paths(config)
    initial_mps_pids = set()
    start = time.time()
    generate_dataset(config, paths["dataset_root"])
    pair_dali, pair_torch = run_pair(config, initial_mps_pids)
    pair_dali_images_per_sec = float(pair_dali["images_per_sec"])
    pair_torch_images_per_sec = float(pair_torch["images_per_sec"])
    score = pair_dali_images_per_sec + pair_torch_images_per_sec
    payload = {
        "benchmark": BENCHMARK_ID,
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "success": True,
        "status": "ok",
        "throughput": score,
        "score": score,
        "pair_dali_images_per_sec": pair_dali_images_per_sec,
        "pair_torch_images_per_sec": pair_torch_images_per_sec,
        "wall_time_sec": time.time() - start,
        "resource_signal_primary": "gpu_shared_resource_contention",
        "config_summary": format_config_summary(config),
        "notes": "score = pair_dali_images_per_sec + pair_torch_images_per_sec",
        "config_snapshot": asdict(config),
        "env_overrides": env_overrides,
        "readiness": ready,
    }
    write_timeseries(paths["case_dir"], payload)
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    config, env_overrides = apply_env_overrides(config)
    paths = case_paths(config)
    write_json(paths["case_dir"] / "effective_env_overrides.json", env_overrides)
    if args.worker != "orchestrator":
        return worker_main(config, str(args.worker), Path(str(args.worker_output_json)))
    if bool(args.scaffold_only):
        payload = scaffold_result(config, env_overrides)
    else:
        payload = run_benchmark(config, env_overrides)
    write_json(Path(config.output_json), payload)
    print(json.dumps(direct_stdout_payload(payload), sort_keys=True))
    return (
        0 if payload.get("success") or payload.get("status") in {"blocked", "scaffold_only"} else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
