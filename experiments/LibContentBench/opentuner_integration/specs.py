from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.LibContentBench.libcontentbench.direct_commands import (
    DirectCommandSpec,
    direct_command_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = REPO_ROOT / "experiments/LibContentBench"
VISIBLE_CPU_COUNT = max(1, min(os.cpu_count() or 1, 64))


@dataclass(frozen=True)
class OpenTunerParameterSpec:
    parameter_name: str
    field_name: str
    parameter_kind: str
    min_value: int | None = None
    max_value: int | None = None
    options: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    value_type: str = "str"

    def whitelist_names(self) -> tuple[str, ...]:
        return (self.parameter_name, *self.aliases)


@dataclass(frozen=True)
class OpenTunerWorkflowSpec:
    workflow_id: str
    direct_spec: DirectCommandSpec
    whitelist_path: Path
    fixed_workload_payload: dict[str, Any]
    tunables: tuple[OpenTunerParameterSpec, ...]

    def load_whitelist(self) -> tuple[str, ...]:
        entries: list[str] = []
        for raw_line in self.whitelist_path.read_text(encoding="utf-8").splitlines():
            value = raw_line.strip()
            if not value:
                continue
            if "|" in value:
                message = "unsupported bundled whitelist entry"
                raise ValueError(f"{self.whitelist_path} contains an {message}: {value!r}")
            entries.append(value)
        return tuple(entries)

    def whitelist_set(self) -> set[str]:
        return set(self.load_whitelist())


def _enum_parameter(
    field_name: str, *options: str, aliases: tuple[str, ...] = (), value_type: str = "str"
) -> OpenTunerParameterSpec:
    return OpenTunerParameterSpec(
        parameter_name=field_name,
        field_name=field_name,
        parameter_kind="enum",
        options=tuple(options),
        aliases=aliases,
        value_type=value_type,
    )


def _integer_parameter(
    parameter_name: str,
    field_name: str,
    minimum: int,
    maximum: int,
    *,
    aliases: tuple[str, ...] = (),
) -> OpenTunerParameterSpec:
    return OpenTunerParameterSpec(
        parameter_name=parameter_name,
        field_name=field_name,
        parameter_kind="integer",
        min_value=minimum,
        max_value=maximum,
        aliases=aliases,
        value_type="int",
    )


SUPPORTED_WORKFLOWS: dict[str, OpenTunerWorkflowSpec] = {
    "nestedcpu-blas": OpenTunerWorkflowSpec(
        workflow_id="nestedcpu-blas",
        direct_spec=direct_command_spec("nestedcpu-blas"),
        whitelist_path=SUITE_ROOT / "nestedcpu-blas/tunable_whitelist.txt",
        fixed_workload_payload={
            "matrix_size": 320,
            "task_count": 800,
            "repetitions": 6,
            "outer_jobs": 6,
            "omp_threads": 6,
            "openblas_threads": 6,
            "mkl_threads": 1,
            "blis_threads": 1,
            "numexpr_threads": 1,
            "backend_prefer": "processes",
        },
        tunables=(
            _integer_parameter(
                "--joblib-n-jobs", "outer_jobs", 1, VISIBLE_CPU_COUNT, aliases=("--outer-jobs",)
            ),
            _integer_parameter("OMP_NUM_THREADS", "omp_threads", 1, VISIBLE_CPU_COUNT),
            _integer_parameter("OPENBLAS_NUM_THREADS", "openblas_threads", 1, VISIBLE_CPU_COUNT),
            _integer_parameter("MKL_NUM_THREADS", "mkl_threads", 1, VISIBLE_CPU_COUNT),
            _integer_parameter("BLIS_NUM_THREADS", "blis_threads", 1, VISIBLE_CPU_COUNT),
            _integer_parameter("NUMEXPR_NUM_THREADS", "numexpr_threads", 1, VISIBLE_CPU_COUNT),
            OpenTunerParameterSpec(
                parameter_name="--backend-prefer",
                field_name="backend_prefer",
                parameter_kind="enum",
                options=("processes", "threads"),
            ),
        ),
    ),
    "arrow-dask-duckdb-memspill": OpenTunerWorkflowSpec(
        workflow_id="arrow-dask-duckdb-memspill",
        direct_spec=direct_command_spec("arrow-dask-duckdb-memspill"),
        whitelist_path=SUITE_ROOT / "arrow-dask-duckdb-memspill/tunable_whitelist.txt",
        fixed_workload_payload={
            "runtime_root": "/var/tmp/libcontentbench/arrow-dask-duckdb-memspill-size-small",
            "num_files": 10,
            "rows_per_file": 20000,
            "row_group_size": 5000,
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
            _integer_parameter("BENCHMARK_N_WORKERS", "n_workers", 1, VISIBLE_CPU_COUNT),
            _integer_parameter(
                "BENCHMARK_THREADS_PER_WORKER", "threads_per_worker", 1, VISIBLE_CPU_COUNT
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_WORKER_MEMORY_LIMIT",
                field_name="worker_memory_limit",
                parameter_kind="enum",
                options=("128MB", "256MB", "512MB", "1GB"),
            ),
            _integer_parameter(
                "BENCHMARK_ARROW_CPU_COUNT", "arrow_cpu_count", 1, VISIBLE_CPU_COUNT
            ),
            _integer_parameter(
                "BENCHMARK_ARROW_IO_THREADS", "arrow_io_threads", 1, VISIBLE_CPU_COUNT
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_DUCKDB_MEMORY_LIMIT",
                field_name="duckdb_memory_limit",
                parameter_kind="enum",
                options=("128MB", "256MB", "512MB", "1GB", "2GB"),
            ),
            _integer_parameter("BENCHMARK_DUCKDB_THREADS", "duckdb_threads", 1, VISIBLE_CPU_COUNT),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_PATH_LAYOUT",
                field_name="path_layout",
                parameter_kind="enum",
                options=("shared", "isolated"),
            ),
        ),
    ),
    "hf-datasets-tokenizers-dataloader": OpenTunerWorkflowSpec(
        workflow_id="hf-datasets-tokenizers-dataloader",
        direct_spec=direct_command_spec("hf-datasets-tokenizers-dataloader"),
        whitelist_path=SUITE_ROOT / "hf-datasets-tokenizers-dataloader/tunable_whitelist.txt",
        fixed_workload_payload={
            "runtime_root": "/var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader-size-small",
            "dataset_rows": 4096,
            "tokens_per_row": 32,
            "max_length": 64,
            "tokenizer_vocab_size": 1024,
            "hidden_dim": 128,
            "num_classes": 32,
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 0.001,
            "torch_num_threads": 24,
            "datasets_num_proc": 24,
            "tokenize_batch_size": 128,
            "tokenizers_parallelism": False,
            "num_workers": 24,
            "prefetch_factor": 1,
            "persistent_workers": True,
            "timeout_seconds": 600,
        },
        tunables=(
            _integer_parameter(
                "BENCHMARK_TORCH_NUM_THREADS", "torch_num_threads", 1, VISIBLE_CPU_COUNT
            ),
            _integer_parameter(
                "BENCHMARK_DATASETS_NUM_PROC", "datasets_num_proc", 1, VISIBLE_CPU_COUNT
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_TOKENIZERS_PARALLELISM",
                field_name="tokenizers_parallelism",
                parameter_kind="enum",
                options=("false", "true"),
                value_type="bool",
            ),
            _integer_parameter("BENCHMARK_NUM_WORKERS", "num_workers", 0, VISIBLE_CPU_COUNT),
            _integer_parameter("BENCHMARK_PREFETCH_FACTOR", "prefetch_factor", 1, 8),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_PERSISTENT_WORKERS",
                field_name="persistent_workers",
                parameter_kind="enum",
                options=("false", "true"),
                value_type="bool",
            ),
        ),
    ),
    "torchdata-fsspec-s3-ddp": OpenTunerWorkflowSpec(
        workflow_id="torchdata-fsspec-s3-ddp",
        direct_spec=direct_command_spec("torchdata-fsspec-s3-ddp"),
        whitelist_path=SUITE_ROOT / "torchdata-fsspec-s3-ddp/tunable_whitelist.txt",
        fixed_workload_payload={
            "runtime_root": "/var/tmp/libcontentbench/torchdata-fsspec-s3-ddp-size-small",
            "world_size": 2,
            "samples_per_rank": 1024,
            "shards_per_rank": 4,
            "input_dim": 256,
            "hidden_dim": 512,
            "num_classes": 32,
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 0.01,
            "torch_num_threads": 24,
            "s3_block_size": 65536,
            "s3_cache_type": "none",
            "s3_use_listings_cache": True,
            "ddp_bucket_cap_mb": 1,
            "timeout_seconds": 420,
            "minio_image": "quay.io/minio/minio:latest",
        },
        tunables=(
            _integer_parameter(
                "BENCHMARK_TORCH_NUM_THREADS", "torch_num_threads", 1, VISIBLE_CPU_COUNT
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_S3_BLOCK_SIZE",
                field_name="s3_block_size",
                parameter_kind="enum",
                options=("4096", "16384", "65536", "262144", "1048576"),
                value_type="int",
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_S3_CACHE_TYPE",
                field_name="s3_cache_type",
                parameter_kind="enum",
                options=("none", "bytes", "readahead", "mmap"),
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_S3_USE_LISTINGS_CACHE",
                field_name="s3_use_listings_cache",
                parameter_kind="enum",
                options=("false", "true"),
                value_type="bool",
            ),
            _integer_parameter("BENCHMARK_DDP_BUCKET_CAP_MB", "ddp_bucket_cap_mb", 1, 64),
        ),
    ),
    "dali-nvjpeg-pytorch-cudnn": OpenTunerWorkflowSpec(
        workflow_id="dali-nvjpeg-pytorch-cudnn",
        direct_spec=direct_command_spec("dali-nvjpeg-pytorch-cudnn"),
        whitelist_path=SUITE_ROOT / "dali-nvjpeg-pytorch-cudnn/tunable_whitelist.txt",
        fixed_workload_payload={
            "runtime_root": "/var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn-opentuner",
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
            "dali_batch_size": 96,
            "dali_prefetch_queue_depth": 2,
            "dali_hw_decoder_load": 0.65,
            "dali_decode_backend": "mixed",
            "dali_num_threads": 4,
            "dali_mps_share": 50,
            "torch_mps_share": 50,
            "torch_batch_size": 256,
            "torch_model": "custom_cnn",
            "torch_precision": "tf32",
            "torch_channels_last": True,
            "torch_cudnn_benchmark": True,
        },
        tunables=(
            _integer_parameter("BENCHMARK_DALI_BATCH_SIZE", "dali_batch_size", 16, 256),
            _integer_parameter(
                "BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH", "dali_prefetch_queue_depth", 1, 4
            ),
            OpenTunerParameterSpec(
                parameter_name="BENCHMARK_DALI_HW_DECODER_LOAD",
                field_name="dali_hw_decoder_load",
                parameter_kind="enum",
                options=("0.25", "0.5", "0.65", "0.75", "0.9"),
                value_type="float",
            ),
            _integer_parameter("BENCHMARK_DALI_NUM_THREADS", "dali_num_threads", 1, 16),
            _integer_parameter("BENCHMARK_DALI_MPS_SHARE", "dali_mps_share", 1, 100),
            _integer_parameter("BENCHMARK_TORCH_BATCH_SIZE", "torch_batch_size", 16, 256),
            _integer_parameter("BENCHMARK_TORCH_MPS_SHARE", "torch_mps_share", 1, 100),
        ),
    ),
}


def supported_workflow_ids() -> tuple[str, ...]:
    return tuple(SUPPORTED_WORKFLOWS)


def load_workflow_spec(workflow_id: str) -> OpenTunerWorkflowSpec:
    try:
        return SUPPORTED_WORKFLOWS[workflow_id]
    except KeyError as exc:
        raise KeyError(f"Unsupported LibContentBench OpenTuner workflow: {workflow_id}") from exc


def _validate_workflow_specs() -> None:
    for workflow_id, workflow_spec in SUPPORTED_WORKFLOWS.items():
        if workflow_spec.direct_spec.benchmark_id != workflow_id:
            raise ValueError(f"Direct command spec mismatch for {workflow_id}")
        if not workflow_spec.whitelist_path.exists():
            raise FileNotFoundError(
                f"Missing tunable whitelist for {workflow_id}: {workflow_spec.whitelist_path}"
            )
        whitelist = workflow_spec.whitelist_set()
        direct_field_names = {field.field_name for field in workflow_spec.direct_spec.cli_fields}
        direct_field_names.update(
            field.field_name for field in workflow_spec.direct_spec.env_fields
        )
        direct_field_names.update(workflow_spec.direct_spec.defaults.keys())
        if "output_json" not in direct_field_names:
            raise ValueError(f"{workflow_id} direct spec must materialize output_json")
        for tunable in workflow_spec.tunables:
            if tunable.field_name not in direct_field_names:
                raise ValueError(f"Unknown field {tunable.field_name!r} for workflow {workflow_id}")
            if not any(name in whitelist for name in tunable.whitelist_names()):
                message = "is not authorized by"
                raise ValueError(
                    f"Workflow {workflow_id} tunable {tunable.parameter_name!r} "
                    f"{message} {workflow_spec.whitelist_path}"
                )


_validate_workflow_specs()
