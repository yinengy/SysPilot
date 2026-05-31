"""Static SMAC3 workflow metadata for LibContentBench.

This module is intentionally import-light: it only depends on the standard
library plus ``libcontentbench.direct_commands`` (the suite's single source of
truth for command materialization). It does **not** import SMAC3 or ConfigSpace,
so the supported workflows, fixed workload shapes, and whitelisted tunables can
be inspected statically without installing the optimizer.

Design mirrors ``opentuner_integration/specs.py`` but is self-contained so the
SMAC3 package can be deployed and reasoned about on its own.

Parameter and metric choices follow ``docs/fixed_budget_exp_prompt.md``:

    nestedcpu-blas                     -> throughput      (maximize)
    arrow-dask-duckdb-memspill         -> throughput      (maximize)
    hf-datasets-tokenizers-dataloader  -> throughput      (maximize)
    torchdata-fsspec-s3-ddp            -> samples_per_sec  (maximize)
    dali-nvjpeg-pytorch-cudnn          -> score            (maximize)  [GPU + CUDA MPS]

The tunable knobs for each workflow are exactly the entries authorized by that
workflow's ``tunable_whitelist.txt``; workload-shape flags (matrix size, dataset
rows, shard counts, batch size, etc.) stay fixed per size tier and are never
tuned. The three size tiers come from ``workflow_size_commands.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.LibContentBench.libcontentbench.direct_commands import (
    DirectCommandSpec,
    direct_command_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = REPO_ROOT / "experiments/LibContentBench"

SIZES: tuple[str, ...] = ("small", "medium", "large")

# Pinned MinIO image for reproducible torchdata-fsspec-s3-ddp runs (matches the
# OpenTuner integration; avoids the floating ``:latest`` tag used in the size
# command catalog).
PINNED_MINIO_IMAGE = "quay.io/minio/minio:RELEASE.2024-01-16T16-07-38Z"


def visible_cpu_count() -> int:
    """Upper bound for thread/worker integer knobs (matches OpenTuner specs)."""
    return max(1, min(os.cpu_count() or 1, 64))


CPU = visible_cpu_count()


@dataclass(frozen=True)
class Tunable:
    """A single whitelist-authorized knob exposed to SMAC3.

    ``field_name`` is the key consumed by ``build_direct_command``'s payload.
    ``whitelist_name`` is the token that must appear in the workflow's
    ``tunable_whitelist.txt`` for this knob to be considered authorized.
    """

    field_name: str
    whitelist_name: str
    kind: str  # "integer" | "float" | "categorical"
    value_type: str = "str"  # "int" | "str" | "bool" | "float"
    min_value: int | float | None = None
    max_value: int | float | None = None
    options: tuple[str, ...] = ()

    def encode_default(self, baseline: Any) -> Any:
        """Convert a baseline payload value into a ConfigSpace default value.

        Integers are clamped into ``[min_value, max_value]`` so that baselines
        derived for a 32-core host (e.g. ``24`` workers) stay valid on smaller
        machines where the upper bound is lower.
        """
        if self.kind == "integer":
            value = int(baseline)
            assert self.min_value is not None
            assert self.max_value is not None
            lo, hi = int(self.min_value), int(self.max_value)
            return max(lo, min(hi, value))
        if self.kind == "float":
            value = float(baseline)
            assert self.min_value is not None
            assert self.max_value is not None
            lo, hi = float(self.min_value), float(self.max_value)
            return max(lo, min(hi, value))
        if self.value_type == "bool":
            truthy = baseline is True or str(baseline).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            token = "true" if truthy else "false"
        else:
            token = str(baseline)
        if token not in self.options:
            raise ValueError(
                f"baseline {baseline!r} for {self.field_name} not in options {self.options}"
            )
        return token

    def coerce(self, raw: Any) -> Any:
        """Convert a ConfigSpace value into the typed payload value."""
        if self.kind == "integer":
            return int(raw)
        if self.kind == "float":
            return float(raw)
        if self.value_type == "bool":
            return str(raw).strip().lower() == "true"
        if self.value_type == "int":
            return int(raw)
        if self.value_type == "float":
            return float(raw)
        return str(raw)


@dataclass
class WorkflowSpec:
    workflow_id: str
    primary_metric: str
    metric_mode: str  # "maximize" | "minimize"
    whitelist_path: Path
    size_payloads: dict[str, dict[str, Any]]  # size -> fixed workload-shape fields
    baseline_runtime: dict[str, Any]  # tunable field_name -> baseline value
    tunables: tuple[Tunable, ...] = field(default_factory=tuple)

    @property
    def direct_spec(self) -> DirectCommandSpec:
        return direct_command_spec(self.workflow_id)

    def load_whitelist(self) -> set[str]:
        entries: set[str] = set()
        for raw_line in self.whitelist_path.read_text(encoding="utf-8").splitlines():
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            entries.add(value)
        return entries

    def default_timeout_seconds(self, size: str) -> int:
        """Outer subprocess safety timeout for a size tier.

        Workflows that expose their own ``--timeout-seconds`` self-limit; we add
        a margin so the inner limit fires first. Workflows without that flag get
        a flat default.
        """
        inner = self.size_payloads[size].get("timeout_seconds")
        if isinstance(inner, int):
            return max(600, inner + 180)
        return 600


def _enum(field_name: str, whitelist_name: str, *options: str, value_type: str = "str") -> Tunable:
    return Tunable(
        field_name=field_name,
        whitelist_name=whitelist_name,
        kind="categorical",
        value_type=value_type,
        options=tuple(options),
    )


def _int(
    field_name: str, whitelist_name: str, minimum: int, maximum: int
) -> Tunable:
    return Tunable(
        field_name=field_name,
        whitelist_name=whitelist_name,
        kind="integer",
        value_type="int",
        min_value=minimum,
        max_value=maximum,
    )


def _float(
    field_name: str, whitelist_name: str, minimum: float, maximum: float
) -> Tunable:
    return Tunable(
        field_name=field_name,
        whitelist_name=whitelist_name,
        kind="float",
        value_type="float",
        min_value=minimum,
        max_value=maximum,
    )


SUPPORTED_WORKFLOWS: dict[str, WorkflowSpec] = {
    "nestedcpu-blas": WorkflowSpec(
        workflow_id="nestedcpu-blas",
        primary_metric="throughput",
        metric_mode="maximize",
        whitelist_path=SUITE_ROOT / "nestedcpu-blas/tunable_whitelist.txt",
        size_payloads={
            "small": {"matrix_size": 320, "task_count": 800, "repetitions": 6},
            "medium": {"matrix_size": 320, "task_count": 1600, "repetitions": 6},
            "large": {"matrix_size": 320, "task_count": 3200, "repetitions": 6},
        },
        baseline_runtime={
            "outer_jobs": 6,
            "omp_threads": 6,
            "openblas_threads": 6,
            "mkl_threads": 1,
            "blis_threads": 1,
            "numexpr_threads": 1,
            "backend_prefer": "processes",
        },
        tunables=(
            _int("outer_jobs", "--joblib-n-jobs", 1, CPU),
            _int("omp_threads", "OMP_NUM_THREADS", 1, CPU),
            _int("openblas_threads", "OPENBLAS_NUM_THREADS", 1, CPU),
            _int("mkl_threads", "MKL_NUM_THREADS", 1, CPU),
            _int("blis_threads", "BLIS_NUM_THREADS", 1, CPU),
            _int("numexpr_threads", "NUMEXPR_NUM_THREADS", 1, CPU),
            _enum("backend_prefer", "--backend-prefer", "processes", "threads"),
        ),
    ),
    "arrow-dask-duckdb-memspill": WorkflowSpec(
        workflow_id="arrow-dask-duckdb-memspill",
        primary_metric="throughput",
        metric_mode="maximize",
        whitelist_path=SUITE_ROOT / "arrow-dask-duckdb-memspill/tunable_whitelist.txt",
        size_payloads={
            "small": {"num_files": 10, "rows_per_file": 20000, "row_group_size": 5000},
            "medium": {"num_files": 12, "rows_per_file": 100000, "row_group_size": 10000},
            "large": {"num_files": 12, "rows_per_file": 250000, "row_group_size": 25000},
        },
        baseline_runtime={
            "n_workers": 24,
            "threads_per_worker": 24,
            "worker_memory_limit": "256MB",
            "arrow_cpu_count": 1,
            "arrow_io_threads": 1,
            "duckdb_memory_limit": "1GB",
            "duckdb_threads": 1,
            "path_layout": "shared",
        },
        tunables=(
            _int("n_workers", "BENCHMARK_N_WORKERS", 1, CPU),
            _int("threads_per_worker", "BENCHMARK_THREADS_PER_WORKER", 1, CPU),
            _enum(
                "worker_memory_limit",
                "BENCHMARK_WORKER_MEMORY_LIMIT",
                "128MB",
                "256MB",
                "512MB",
                "1GB",
            ),
            _int("arrow_cpu_count", "BENCHMARK_ARROW_CPU_COUNT", 1, CPU),
            _int("arrow_io_threads", "BENCHMARK_ARROW_IO_THREADS", 1, CPU),
            _enum(
                "duckdb_memory_limit",
                "BENCHMARK_DUCKDB_MEMORY_LIMIT",
                "128MB",
                "256MB",
                "512MB",
                "1GB",
            ),
            _int("duckdb_threads", "BENCHMARK_DUCKDB_THREADS", 1, CPU),
            _enum("path_layout", "BENCHMARK_PATH_LAYOUT", "shared", "isolated"),
        ),
    ),
    "hf-datasets-tokenizers-dataloader": WorkflowSpec(
        workflow_id="hf-datasets-tokenizers-dataloader",
        primary_metric="throughput",
        metric_mode="maximize",
        whitelist_path=SUITE_ROOT / "hf-datasets-tokenizers-dataloader/tunable_whitelist.txt",
        size_payloads={
            "small": {
                "dataset_rows": 4096,
                "tokens_per_row": 32,
                "max_length": 64,
                "tokenizer_vocab_size": 1024,
                "hidden_dim": 128,
                "num_classes": 32,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.001,
                "tokenize_batch_size": 128,
                "timeout_seconds": 600,
            },
            "medium": {
                "dataset_rows": 8192,
                "tokens_per_row": 48,
                "max_length": 96,
                "tokenizer_vocab_size": 2048,
                "hidden_dim": 256,
                "num_classes": 32,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.001,
                "tokenize_batch_size": 256,
                "timeout_seconds": 600,
            },
            "large": {
                "dataset_rows": 16384,
                "tokens_per_row": 64,
                "max_length": 128,
                "tokenizer_vocab_size": 4096,
                "hidden_dim": 512,
                "num_classes": 64,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.001,
                "tokenize_batch_size": 256,
                "timeout_seconds": 900,
            },
        },
        baseline_runtime={
            "torch_num_threads": 24,
            "datasets_num_proc": 24,
            "tokenizers_parallelism": False,
            "num_workers": 24,
            "prefetch_factor": 1,
            "persistent_workers": True,
        },
        tunables=(
            _int("torch_num_threads", "BENCHMARK_TORCH_NUM_THREADS", 1, CPU),
            _int("datasets_num_proc", "BENCHMARK_DATASETS_NUM_PROC", 1, CPU),
            _enum(
                "tokenizers_parallelism",
                "BENCHMARK_TOKENIZERS_PARALLELISM",
                "false",
                "true",
                value_type="bool",
            ),
            _int("num_workers", "BENCHMARK_NUM_WORKERS", 0, CPU),
            _int("prefetch_factor", "BENCHMARK_PREFETCH_FACTOR", 1, CPU),
            _enum(
                "persistent_workers",
                "BENCHMARK_PERSISTENT_WORKERS",
                "false",
                "true",
                value_type="bool",
            ),
        ),
    ),
    "torchdata-fsspec-s3-ddp": WorkflowSpec(
        workflow_id="torchdata-fsspec-s3-ddp",
        primary_metric="samples_per_sec",
        metric_mode="maximize",
        whitelist_path=SUITE_ROOT / "torchdata-fsspec-s3-ddp/tunable_whitelist.txt",
        size_payloads={
            "small": {
                "world_size": 2,
                "samples_per_rank": 1024,
                "shards_per_rank": 4,
                "input_dim": 256,
                "hidden_dim": 512,
                "num_classes": 32,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.01,
                "timeout_seconds": 420,
                "minio_image": PINNED_MINIO_IMAGE,
            },
            "medium": {
                "world_size": 2,
                "samples_per_rank": 2048,
                "shards_per_rank": 8,
                "input_dim": 256,
                "hidden_dim": 512,
                "num_classes": 32,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.01,
                "timeout_seconds": 480,
                "minio_image": PINNED_MINIO_IMAGE,
            },
            "large": {
                "world_size": 2,
                "samples_per_rank": 4096,
                "shards_per_rank": 8,
                "input_dim": 512,
                "hidden_dim": 1024,
                "num_classes": 32,
                "batch_size": 32,
                "epochs": 2,
                "learning_rate": 0.01,
                "timeout_seconds": 600,
                "minio_image": PINNED_MINIO_IMAGE,
            },
        },
        baseline_runtime={
            "torch_num_threads": 24,
            "s3_block_size": 65536,
            "s3_cache_type": "none",
            "s3_use_listings_cache": True,
            "ddp_bucket_cap_mb": 1,
        },
        tunables=(
            _int("torch_num_threads", "BENCHMARK_TORCH_NUM_THREADS", 1, CPU),
            _int("s3_block_size", "BENCHMARK_S3_BLOCK_SIZE", 4096, 1048576),
            _enum(
                "s3_cache_type",
                "BENCHMARK_S3_CACHE_TYPE",
                "none",
                "bytes",
                "readahead",
                "mmap",
            ),
            _enum(
                "s3_use_listings_cache",
                "BENCHMARK_S3_USE_LISTINGS_CACHE",
                "false",
                "true",
                value_type="bool",
            ),
            _int("ddp_bucket_cap_mb", "BENCHMARK_DDP_BUCKET_CAP_MB", 1, 64),
        ),
    ),
    "dali-nvjpeg-pytorch-cudnn": WorkflowSpec(
        workflow_id="dali-nvjpeg-pytorch-cudnn",
        primary_metric="score",
        metric_mode="maximize",
        whitelist_path=SUITE_ROOT / "dali-nvjpeg-pytorch-cudnn/tunable_whitelist.txt",
        size_payloads={
            "small": {
                "timeout_seconds": 300,
                "warmup_seconds": 10.0,
                "measure_seconds": 25.0,
                "cuda_device": "0",
                "dataset_images": 1500,
                "image_min_size": 640,
                "image_max_size": 768,
                "jpeg_quality": 90,
                "num_classes": 16,
                "resize_size": 224,
                "crop_size": 224,
                "dali_output_dtype": "float16",
                "dali_decode_backend": "mixed",
                "torch_model": "custom_cnn",
                "torch_precision": "tf32",
                "torch_channels_last": True,
                "torch_cudnn_benchmark": True,
            },
            "medium": {
                "timeout_seconds": 600,
                "warmup_seconds": 10.0,
                "measure_seconds": 45.0,
                "cuda_device": "0",
                "dataset_images": 5000,
                "image_min_size": 768,
                "image_max_size": 1024,
                "jpeg_quality": 90,
                "num_classes": 16,
                "resize_size": 224,
                "crop_size": 224,
                "dali_output_dtype": "float16",
                "dali_decode_backend": "mixed",
                "torch_model": "custom_cnn",
                "torch_precision": "tf32",
                "torch_channels_last": True,
                "torch_cudnn_benchmark": True,
            },
            "large": {
                "timeout_seconds": 900,
                "warmup_seconds": 15.0,
                "measure_seconds": 75.0,
                "cuda_device": "0",
                "dataset_images": 10000,
                "image_min_size": 1024,
                "image_max_size": 1280,
                "jpeg_quality": 90,
                "num_classes": 16,
                "resize_size": 224,
                "crop_size": 224,
                "dali_output_dtype": "float16",
                "dali_decode_backend": "mixed",
                "torch_model": "custom_cnn",
                "torch_precision": "tf32",
                "torch_channels_last": True,
                "torch_cudnn_benchmark": True,
            },
        },
        baseline_runtime={
            "dali_batch_size": 16,
            "dali_prefetch_queue_depth": 1,
            "dali_hw_decoder_load": 0.65,
            "dali_num_threads": 1,
            "dali_mps_share": 50,
            "torch_batch_size": 16,
            "torch_mps_share": 50,
        },
        tunables=(
            _int("dali_batch_size", "BENCHMARK_DALI_BATCH_SIZE", 1, 512),
            _int("dali_prefetch_queue_depth", "BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH", 1, CPU),
            _float("dali_hw_decoder_load", "BENCHMARK_DALI_HW_DECODER_LOAD", 0.0, 1.0),
            _int("dali_num_threads", "BENCHMARK_DALI_NUM_THREADS", 1, CPU),
            _int("dali_mps_share", "BENCHMARK_DALI_MPS_SHARE", 1, 100),
            _int("torch_batch_size", "BENCHMARK_TORCH_BATCH_SIZE", 1, 512),
            _int("torch_mps_share", "BENCHMARK_TORCH_MPS_SHARE", 1, 100),
        ),
    ),
}


def supported_workflow_ids() -> tuple[str, ...]:
    return tuple(SUPPORTED_WORKFLOWS)


def get_workflow(workflow_id: str) -> WorkflowSpec:
    try:
        return SUPPORTED_WORKFLOWS[workflow_id]
    except KeyError as exc:
        supported = ", ".join(supported_workflow_ids())
        raise KeyError(
            f"Unsupported SMAC3 workflow: {workflow_id!r}. Supported: {supported}"
        ) from exc


def baseline_config_values(spec: WorkflowSpec) -> dict[str, Any]:
    """The baseline (seed) point as a ConfigSpace-valued dict.

    Pure (no ConfigSpace import) so it is usable for host-safe dry runs.
    """
    return {
        tunable.field_name: tunable.encode_default(spec.baseline_runtime[tunable.field_name])
        for tunable in spec.tunables
    }


def build_trial_payload(
    spec: WorkflowSpec,
    size: str,
    config_values: dict[str, Any],
    per_trial: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the full direct-command payload for one trial.

    payload = fixed workload-shape fields (per size)
            + coerced tunable knobs (from the SMAC config)
            + per-trial output/isolation fields (paths)
    """
    if size not in spec.size_payloads:
        raise KeyError(f"Unknown size {size!r} for {spec.workflow_id}")
    payload: dict[str, Any] = dict(spec.size_payloads[size])
    for tunable in spec.tunables:
        payload[tunable.field_name] = tunable.coerce(config_values[tunable.field_name])
    payload.update(per_trial)
    return payload


def _validate_workflow_specs() -> None:
    for workflow_id, spec in SUPPORTED_WORKFLOWS.items():
        if spec.direct_spec.benchmark_id != workflow_id:
            raise ValueError(f"Direct command spec mismatch for {workflow_id}")
        if spec.metric_mode not in {"maximize", "minimize"}:
            raise ValueError(f"{workflow_id} has invalid metric_mode {spec.metric_mode!r}")
        if not spec.whitelist_path.exists():
            raise FileNotFoundError(
                f"Missing tunable whitelist for {workflow_id}: {spec.whitelist_path}"
            )
        whitelist = spec.load_whitelist()

        direct_field_names = {f.field_name for f in spec.direct_spec.cli_fields}
        direct_field_names.update(f.field_name for f in spec.direct_spec.env_fields)
        direct_field_names.update(spec.direct_spec.defaults.keys())
        if "output_json" not in direct_field_names:
            raise ValueError(f"{workflow_id} direct spec must materialize output_json")

        tunable_fields = {t.field_name for t in spec.tunables}
        if set(spec.baseline_runtime) != tunable_fields:
            raise ValueError(
                f"{workflow_id} baseline_runtime keys must equal tunable fields; "
                f"got {sorted(spec.baseline_runtime)} vs {sorted(tunable_fields)}"
            )

        for size in SIZES:
            if size not in spec.size_payloads:
                raise ValueError(f"{workflow_id} missing size payload for {size}")

        for tunable in spec.tunables:
            if tunable.field_name not in direct_field_names:
                raise ValueError(
                    f"Unknown field {tunable.field_name!r} for workflow {workflow_id}"
                )
            if tunable.whitelist_name not in whitelist:
                raise ValueError(
                    f"Workflow {workflow_id} tunable {tunable.field_name!r} "
                    f"({tunable.whitelist_name}) is not authorized by {spec.whitelist_path}"
                )
            # Baseline must encode into a valid ConfigSpace default.
            tunable.encode_default(spec.baseline_runtime[tunable.field_name])


_validate_workflow_specs()
