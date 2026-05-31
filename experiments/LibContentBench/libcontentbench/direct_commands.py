from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = REPO_ROOT / "experiments/LibContentBench"


@dataclass(frozen=True)
class EnvVarSpec:
    field_name: str
    env_name: str


@dataclass(frozen=True)
class CliFieldSpec:
    field_name: str
    option: str


@dataclass(frozen=True)
class BoolCliFieldSpec:
    field_name: str
    positive_option: str
    negative_option: str


@dataclass(frozen=True)
class RepeatedCliFieldSpec:
    field_name: str
    option: str


@dataclass(frozen=True)
class DirectCommandSpec:
    benchmark_id: str
    workspace_name: str
    runner_name: str
    cli_fields: tuple[CliFieldSpec, ...]
    env_fields: tuple[EnvVarSpec, ...] = ()
    bool_cli_fields: tuple[BoolCliFieldSpec, ...] = ()
    repeated_cli_fields: tuple[RepeatedCliFieldSpec, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def workspace_root(self) -> Path:
        return SUITE_ROOT / self.workspace_name

    @property
    def python_relpath(self) -> Path:
        return (
            Path("experiments/LibContentBench") / self.workspace_name / "artifact/venv/bin/python"
        )

    @property
    def runner_relpath(self) -> Path:
        return Path("experiments/LibContentBench") / self.workspace_name / self.runner_name


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _default_repo_relative_output_json(spec: DirectCommandSpec) -> str:
    return str(
        Path("experiments/LibContentBench") / spec.workspace_name / "work/direct_result.json"
    )


def _default_repo_relative_case_output_dir(spec: DirectCommandSpec) -> str:
    return str(
        Path("experiments/LibContentBench") / spec.workspace_name / "work/direct_case_output"
    )


def _canonical_command_payload(
    spec: DirectCommandSpec, source: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(spec.defaults)
    payload.update(source)
    payload.setdefault("benchmark", spec.benchmark_id)
    payload["output_json"] = _default_repo_relative_output_json(spec)
    if any(field.field_name == "case_output_dir" for field in spec.cli_fields):
        payload["case_output_dir"] = _default_repo_relative_case_output_dir(spec)
    return payload


def _field_value(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in payload:
        raise KeyError(f"Missing field {field_name!r} for direct command synthesis")
    return payload[field_name]


def _build_env_tokens(spec: DirectCommandSpec, payload: Mapping[str, Any]) -> list[str]:
    env_tokens = [
        f"{field.env_name}={_stringify(_field_value(payload, field.field_name))}"
        for field in spec.env_fields
    ]
    if not env_tokens:
        return []
    return ["env", *env_tokens]


def _build_cli_tokens(spec: DirectCommandSpec, payload: Mapping[str, Any]) -> list[str]:
    tokens = [str(spec.python_relpath), str(spec.runner_relpath)]
    for cli_field in spec.cli_fields:
        tokens.extend([cli_field.option, _stringify(_field_value(payload, cli_field.field_name))])
    for bool_field in spec.bool_cli_fields:
        value = _field_value(payload, bool_field.field_name)
        if not isinstance(value, bool):
            value = bool(value)
        tokens.append(bool_field.positive_option if value else bool_field.negative_option)
    for repeated_field in spec.repeated_cli_fields:
        values = _field_value(payload, repeated_field.field_name)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(
                f"Repeated field {repeated_field.field_name!r} must be a sequence of values"
            )
        for value in values:
            tokens.extend([repeated_field.option, _stringify(value)])
    return tokens


DIRECT_COMMAND_SPECS: dict[str, DirectCommandSpec] = {
    "nestedcpu-blas": DirectCommandSpec(
        benchmark_id="nestedcpu-blas",
        workspace_name="nestedcpu-blas",
        runner_name="run_nestedcpu_blas_case.py",
        env_fields=(
            EnvVarSpec("omp_threads", "OMP_NUM_THREADS"),
            EnvVarSpec("openblas_threads", "OPENBLAS_NUM_THREADS"),
            EnvVarSpec("mkl_threads", "MKL_NUM_THREADS"),
            EnvVarSpec("blis_threads", "BLIS_NUM_THREADS"),
            EnvVarSpec("numexpr_threads", "NUMEXPR_NUM_THREADS"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("matrix_size", "--matrix-size"),
            CliFieldSpec("task_count", "--task-count"),
            CliFieldSpec("repetitions", "--repetitions"),
            CliFieldSpec("outer_jobs", "--joblib-n-jobs"),
            CliFieldSpec("backend_prefer", "--backend-prefer"),
        ),
        defaults={
            "backend_prefer": "processes",
            "omp_threads": 6,
            "openblas_threads": 6,
            "mkl_threads": 6,
            "blis_threads": 6,
            "numexpr_threads": 6,
        },
    ),
    "arrow-dask-duckdb-memspill": DirectCommandSpec(
        benchmark_id="arrow-dask-duckdb-memspill",
        workspace_name="arrow-dask-duckdb-memspill",
        runner_name="run_arrow_dask_duckdb_case.py",
        env_fields=(
            EnvVarSpec("n_workers", "BENCHMARK_N_WORKERS"),
            EnvVarSpec("threads_per_worker", "BENCHMARK_THREADS_PER_WORKER"),
            EnvVarSpec("worker_memory_limit", "BENCHMARK_WORKER_MEMORY_LIMIT"),
            EnvVarSpec("arrow_cpu_count", "BENCHMARK_ARROW_CPU_COUNT"),
            EnvVarSpec("arrow_io_threads", "BENCHMARK_ARROW_IO_THREADS"),
            EnvVarSpec("duckdb_memory_limit", "BENCHMARK_DUCKDB_MEMORY_LIMIT"),
            EnvVarSpec("duckdb_threads", "BENCHMARK_DUCKDB_THREADS"),
            EnvVarSpec("path_layout", "BENCHMARK_PATH_LAYOUT"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("num_files", "--num-files"),
            CliFieldSpec("rows_per_file", "--rows-per-file"),
            CliFieldSpec("row_group_size", "--row-group-size"),
        ),
    ),
    "hdf5-mpiio-write": DirectCommandSpec(
        benchmark_id="hdf5-mpiio-write",
        workspace_name="hdf5-mpiio-write",
        runner_name="run_hdf5_mpiio_case.py",
        env_fields=(
            EnvVarSpec("collective_io", "BENCHMARK_COLLECTIVE_IO"),
            EnvVarSpec("cb_nodes", "BENCHMARK_CB_NODES"),
            EnvVarSpec("cb_buffer_size", "BENCHMARK_CB_BUFFER_SIZE"),
            EnvVarSpec("romio_cb_write", "BENCHMARK_ROMIO_CB_WRITE"),
            EnvVarSpec("romio_ds_write", "BENCHMARK_ROMIO_DS_WRITE"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("mpi_ranks", "--mpi-ranks"),
            CliFieldSpec("write_mb_per_rank", "--write-mb-per-rank"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
        ),
    ),
    "horovod-tfdata-network": DirectCommandSpec(
        benchmark_id="horovod-tfdata-network",
        workspace_name="horovod-tfdata-network",
        runner_name="run_horovod_tfdata_case.py",
        env_fields=(
            EnvVarSpec("tfdata_target_workers", "BENCHMARK_TFDATA_TARGET_WORKERS"),
            EnvVarSpec(
                "tfdata_max_outstanding_requests", "BENCHMARK_TFDATA_MAX_OUTSTANDING_REQUESTS"
            ),
            EnvVarSpec("tfdata_compression", "BENCHMARK_TFDATA_COMPRESSION"),
            EnvVarSpec("horovod_fusion_threshold_mb", "BENCHMARK_HOROVOD_FUSION_THRESHOLD_MB"),
            EnvVarSpec("horovod_cycle_time_ms", "BENCHMARK_HOROVOD_CYCLE_TIME_MS"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("case_output_dir", "--case-output-dir"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
            CliFieldSpec("batch_size", "--batch-size"),
            CliFieldSpec("model_width", "--model-width"),
            CliFieldSpec("tensor_size", "--tensor-size"),
            CliFieldSpec("steps", "--steps"),
            CliFieldSpec("learning_rate", "--learning-rate"),
            CliFieldSpec("hvd_processes", "--hvd-processes"),
            CliFieldSpec("tfdata_worker_count", "--tfdata-worker-count"),
            CliFieldSpec("tfdata_processing_mode", "--tfdata-processing-mode"),
        ),
        repeated_cli_fields=(RepeatedCliFieldSpec("nic_names", "--nic-name"),),
        defaults={"nic_names": ["lo"]},
    ),
    "torchdata-checkpoint-io": DirectCommandSpec(
        benchmark_id="torchdata-checkpoint-io",
        workspace_name="torchdata-checkpoint-io",
        runner_name="run_torchdata_checkpoint_io_case.py",
        env_fields=(
            EnvVarSpec("torch_num_threads", "BENCHMARK_TORCH_NUM_THREADS"),
            EnvVarSpec("num_workers", "BENCHMARK_NUM_WORKERS"),
            EnvVarSpec("prefetch_factor", "BENCHMARK_PREFETCH_FACTOR"),
            EnvVarSpec("persistent_workers", "BENCHMARK_PERSISTENT_WORKERS"),
            EnvVarSpec("checkpoint_every_steps", "BENCHMARK_CHECKPOINT_EVERY_STEPS"),
            EnvVarSpec("log_every_steps", "BENCHMARK_LOG_EVERY_STEPS"),
            EnvVarSpec("writer_max_queue", "BENCHMARK_WRITER_MAX_QUEUE"),
            EnvVarSpec("writer_flush_secs", "BENCHMARK_WRITER_FLUSH_SECS"),
            EnvVarSpec("flush_writer_on_checkpoint", "BENCHMARK_FLUSH_WRITER_ON_CHECKPOINT"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("dataset_shards", "--dataset-shards"),
            CliFieldSpec("samples_per_shard", "--samples-per-shard"),
            CliFieldSpec("input_dim", "--input-dim"),
            CliFieldSpec("hidden_dim", "--hidden-dim"),
            CliFieldSpec("num_classes", "--num-classes"),
            CliFieldSpec("epochs", "--epochs"),
            CliFieldSpec("learning_rate", "--learning-rate"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
        ),
    ),
    "pytorch-h5py-hdf5-dataloader": DirectCommandSpec(
        benchmark_id="pytorch-h5py-hdf5-dataloader",
        workspace_name="pytorch-h5py-hdf5-dataloader",
        runner_name="run_pytorch_h5py_hdf5_case.py",
        env_fields=(
            EnvVarSpec("torch_num_threads", "BENCHMARK_TORCH_NUM_THREADS"),
            EnvVarSpec("num_workers", "BENCHMARK_NUM_WORKERS"),
            EnvVarSpec("prefetch_factor", "BENCHMARK_PREFETCH_FACTOR"),
            EnvVarSpec("persistent_workers", "BENCHMARK_PERSISTENT_WORKERS"),
            EnvVarSpec("hdf5_rdcc_nbytes", "BENCHMARK_HDF5_RDCC_NBYTES"),
            EnvVarSpec("hdf5_rdcc_nslots", "BENCHMARK_HDF5_RDCC_NSLOTS"),
            EnvVarSpec("hdf5_rdcc_w0", "BENCHMARK_HDF5_RDCC_W0"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("dataset_rows", "--dataset-rows"),
            CliFieldSpec("feature_dim", "--feature-dim"),
            CliFieldSpec("hidden_dim", "--hidden-dim"),
            CliFieldSpec("num_classes", "--num-classes"),
            CliFieldSpec("batch_size", "--batch-size"),
            CliFieldSpec("epochs", "--epochs"),
            CliFieldSpec("learning_rate", "--learning-rate"),
            CliFieldSpec("hdf5_chunk_rows", "--hdf5-chunk-rows"),
            CliFieldSpec("hdf5_libver", "--hdf5-libver"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
        ),
    ),
    "hf-datasets-tokenizers-dataloader": DirectCommandSpec(
        benchmark_id="hf-datasets-tokenizers-dataloader",
        workspace_name="hf-datasets-tokenizers-dataloader",
        runner_name="run_hf_datasets_tokenizers_case.py",
        env_fields=(
            EnvVarSpec("torch_num_threads", "BENCHMARK_TORCH_NUM_THREADS"),
            EnvVarSpec("datasets_num_proc", "BENCHMARK_DATASETS_NUM_PROC"),
            EnvVarSpec("tokenizers_parallelism", "BENCHMARK_TOKENIZERS_PARALLELISM"),
            EnvVarSpec("num_workers", "BENCHMARK_NUM_WORKERS"),
            EnvVarSpec("prefetch_factor", "BENCHMARK_PREFETCH_FACTOR"),
            EnvVarSpec("persistent_workers", "BENCHMARK_PERSISTENT_WORKERS"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("dataset_rows", "--dataset-rows"),
            CliFieldSpec("tokens_per_row", "--tokens-per-row"),
            CliFieldSpec("max_length", "--max-length"),
            CliFieldSpec("tokenizer_vocab_size", "--tokenizer-vocab-size"),
            CliFieldSpec("hidden_dim", "--hidden-dim"),
            CliFieldSpec("num_classes", "--num-classes"),
            CliFieldSpec("batch_size", "--batch-size"),
            CliFieldSpec("epochs", "--epochs"),
            CliFieldSpec("learning_rate", "--learning-rate"),
            CliFieldSpec("tokenize_batch_size", "--tokenize-batch-size"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
        ),
    ),
    "dask-zarr-blosc": DirectCommandSpec(
        benchmark_id="dask-zarr-blosc",
        workspace_name="dask-zarr-blosc",
        runner_name="run_dask_zarr_blosc_case.py",
        env_fields=(
            EnvVarSpec("n_workers", "BENCHMARK_DASK_N_WORKERS"),
            EnvVarSpec("threads_per_worker", "BENCHMARK_DASK_THREADS_PER_WORKER"),
            EnvVarSpec("blosc_nthreads", "BENCHMARK_BLOSC_NTHREADS"),
            EnvVarSpec("blosc_cname", "BENCHMARK_BLOSC_CNAME"),
            EnvVarSpec("blosc_clevel", "BENCHMARK_BLOSC_CLEVEL"),
            EnvVarSpec("blosc_shuffle", "BENCHMARK_BLOSC_SHUFFLE"),
            EnvVarSpec("zarr_dimension_separator", "BENCHMARK_ZARR_DIMENSION_SEPARATOR"),
            EnvVarSpec("zarr_order", "BENCHMARK_ZARR_ORDER"),
            EnvVarSpec("zarr_write_empty_chunks", "BENCHMARK_ZARR_WRITE_EMPTY_CHUNKS"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("array_rows", "--array-rows"),
            CliFieldSpec("array_cols", "--array-cols"),
            CliFieldSpec("chunk_rows", "--chunk-rows"),
            CliFieldSpec("chunk_cols", "--chunk-cols"),
            CliFieldSpec("analysis_passes", "--analysis-passes"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
        ),
        defaults={
            "zarr_dimension_separator": ".",
            "zarr_order": "C",
            "zarr_write_empty_chunks": False,
        },
    ),
    "torchdata-fsspec-s3-ddp": DirectCommandSpec(
        benchmark_id="torchdata-fsspec-s3-ddp",
        workspace_name="torchdata-fsspec-s3-ddp",
        runner_name="run_torchdata_fsspec_s3_ddp_case.py",
        env_fields=(
            EnvVarSpec("torch_num_threads", "BENCHMARK_TORCH_NUM_THREADS"),
            EnvVarSpec("s3_block_size", "BENCHMARK_S3_BLOCK_SIZE"),
            EnvVarSpec("s3_cache_type", "BENCHMARK_S3_CACHE_TYPE"),
            EnvVarSpec("s3_use_listings_cache", "BENCHMARK_S3_USE_LISTINGS_CACHE"),
            EnvVarSpec("ddp_bucket_cap_mb", "BENCHMARK_DDP_BUCKET_CAP_MB"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("case_output_dir", "--case-output-dir"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("world_size", "--world-size"),
            CliFieldSpec("samples_per_rank", "--samples-per-rank"),
            CliFieldSpec("shards_per_rank", "--shards-per-rank"),
            CliFieldSpec("input_dim", "--input-dim"),
            CliFieldSpec("hidden_dim", "--hidden-dim"),
            CliFieldSpec("num_classes", "--num-classes"),
            CliFieldSpec("batch_size", "--batch-size"),
            CliFieldSpec("epochs", "--epochs"),
            CliFieldSpec("learning_rate", "--learning-rate"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
            CliFieldSpec("minio_image", "--minio-image"),
        ),
    ),
    "adios2-bp5-sst": DirectCommandSpec(
        benchmark_id="adios2-bp5-sst",
        workspace_name="adios2-bp5-sst",
        runner_name="run_adios2_bp5_sst_case.py",
        env_fields=(
            EnvVarSpec("adios_engine", "BENCHMARK_ADIOS_ENGINE"),
            EnvVarSpec("aggregation_type", "BENCHMARK_AGGREGATION_TYPE"),
            EnvVarSpec("num_subfiles", "BENCHMARK_NUM_SUBFILES"),
            EnvVarSpec("async_write", "BENCHMARK_ASYNC_WRITE"),
            EnvVarSpec("stats_level", "BENCHMARK_STATS_LEVEL"),
            EnvVarSpec("buffer_vtype", "BENCHMARK_BUFFER_VTYPE"),
            EnvVarSpec("buffer_chunk_size", "BENCHMARK_BUFFER_CHUNK_SIZE"),
            EnvVarSpec("read_threads", "BENCHMARK_READ_THREADS"),
            EnvVarSpec("metadata_threads", "BENCHMARK_METADATA_THREADS"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("case_output_dir", "--case-output-dir"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("steps", "--steps"),
            CliFieldSpec("rows", "--rows"),
            CliFieldSpec("cols", "--cols"),
            CliFieldSpec("variables", "--variables"),
            CliFieldSpec("readback_passes", "--readback-passes"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
        ),
    ),
    "raydata-ddp-network": DirectCommandSpec(
        benchmark_id="raydata-ddp-network",
        workspace_name="raydata-ddp-network",
        runner_name="run_raydata_ddp_network_case.py",
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("case_output_dir", "--case-output-dir"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
            CliFieldSpec("total_samples", "--total-samples"),
            CliFieldSpec("feature_dim", "--feature-dim"),
            CliFieldSpec("hidden_dim", "--hidden-dim"),
            CliFieldSpec("epochs", "--epochs"),
            CliFieldSpec("batch_size", "--batch-size"),
            CliFieldSpec("learning_rate", "--learning-rate"),
            CliFieldSpec("ray_num_workers", "--ray-num-workers"),
            CliFieldSpec("cpus_per_worker", "--cpus-per-worker"),
            CliFieldSpec("ddp_backend", "--ddp-backend"),
            CliFieldSpec("ray_num_blocks", "--ray-num-blocks"),
            CliFieldSpec("repartition_blocks", "--repartition-blocks"),
            CliFieldSpec("map_batch_size", "--map-batch-size"),
            CliFieldSpec("prefetch_batches", "--prefetch-batches"),
            CliFieldSpec("local_shuffle_buffer_size", "--local-shuffle-buffer-size"),
            CliFieldSpec("bucket_cap_mb", "--bucket-cap-mb"),
            CliFieldSpec("max_steps", "--max-steps"),
            CliFieldSpec("benchmark", "--benchmark"),
        ),
        bool_cli_fields=(
            BoolCliFieldSpec("use_gpu", "--use-gpu", "--no-use-gpu"),
            BoolCliFieldSpec(
                "materialize_dataset", "--materialize-dataset", "--no-materialize-dataset"
            ),
        ),
        repeated_cli_fields=(RepeatedCliFieldSpec("nic_names", "--nic-name"),),
        defaults={
            "benchmark": "raydata-ddp-network",
            "nic_names": ["lo", "enp7s0f0", "enp7s0f1"],
            "use_gpu": False,
            "materialize_dataset": False,
        },
    ),
    "dali-nvjpeg-pytorch-cudnn": DirectCommandSpec(
        benchmark_id="dali-nvjpeg-pytorch-cudnn",
        workspace_name="dali-nvjpeg-pytorch-cudnn",
        runner_name="run_dali_nvjpeg_pytorch_cudnn_case.py",
        env_fields=(
            EnvVarSpec("dali_batch_size", "BENCHMARK_DALI_BATCH_SIZE"),
            EnvVarSpec("dali_prefetch_queue_depth", "BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH"),
            EnvVarSpec("dali_hw_decoder_load", "BENCHMARK_DALI_HW_DECODER_LOAD"),
            EnvVarSpec("dali_num_threads", "BENCHMARK_DALI_NUM_THREADS"),
            EnvVarSpec("dali_mps_share", "BENCHMARK_DALI_MPS_SHARE"),
            EnvVarSpec("torch_batch_size", "BENCHMARK_TORCH_BATCH_SIZE"),
            EnvVarSpec("torch_mps_share", "BENCHMARK_TORCH_MPS_SHARE"),
        ),
        cli_fields=(
            CliFieldSpec("output_json", "--output-json"),
            CliFieldSpec("case_output_dir", "--case-output-dir"),
            CliFieldSpec("runtime_root", "--runtime-root"),
            CliFieldSpec("timeout_seconds", "--timeout-seconds"),
            CliFieldSpec("warmup_seconds", "--warmup-seconds"),
            CliFieldSpec("measure_seconds", "--measure-seconds"),
            CliFieldSpec("cuda_device", "--cuda-device"),
            CliFieldSpec("dataset_images", "--dataset-images"),
            CliFieldSpec("image_min_size", "--image-min-size"),
            CliFieldSpec("image_max_size", "--image-max-size"),
            CliFieldSpec("jpeg_quality", "--jpeg-quality"),
            CliFieldSpec("num_classes", "--num-classes"),
            CliFieldSpec("resize_size", "--resize-size"),
            CliFieldSpec("crop_size", "--crop-size"),
            CliFieldSpec("dali_output_dtype", "--dali-output-dtype"),
            CliFieldSpec("dali_decode_backend", "--dali-decode-backend"),
            CliFieldSpec("torch_model", "--torch-model"),
            CliFieldSpec("torch_precision", "--torch-precision"),
            CliFieldSpec("benchmark", "--benchmark"),
        ),
        bool_cli_fields=(
            BoolCliFieldSpec(
                "torch_channels_last", "--torch-channels-last", "--no-torch-channels-last"
            ),
            BoolCliFieldSpec(
                "torch_cudnn_benchmark", "--torch-cudnn-benchmark", "--no-torch-cudnn-benchmark"
            ),
        ),
        defaults={"benchmark": "dali-nvjpeg-pytorch-cudnn"},
    ),
}


def direct_command_specs() -> dict[str, DirectCommandSpec]:
    return dict(DIRECT_COMMAND_SPECS)


def direct_command_spec(benchmark_id: str) -> DirectCommandSpec:
    try:
        return DIRECT_COMMAND_SPECS[benchmark_id]
    except KeyError as exc:
        raise KeyError(f"Unknown LibContentBench benchmark id: {benchmark_id}") from exc


def build_direct_command(benchmark_id: str, source: Mapping[str, Any]) -> str:
    spec = direct_command_spec(benchmark_id)
    payload = _canonical_command_payload(spec, source)
    return shlex.join([*_build_env_tokens(spec, payload), *_build_cli_tokens(spec, payload)])


def build_direct_command_from_row(benchmark_id: str, row: Mapping[str, Any]) -> str:
    config_snapshot = row.get("config_snapshot")
    if isinstance(config_snapshot, Mapping):
        source = config_snapshot
    else:
        source = row
    return build_direct_command(benchmark_id, source)


def rewrite_syspilot_command_value(command: str, inner_command: str) -> str:
    tokens = shlex.split(command)
    outer_option_tokens = {
        "--quick-mode",
        "--primary-objective",
        "--primary-objective-mode",
        "--contract",
        "--manual",
        "--memory-high",
        "--max-iterations",
        "--ui-mode",
        "--workdir",
    }
    for index, token in enumerate(tokens):
        if token == "--command":
            if index + 1 >= len(tokens):
                raise ValueError("--command is missing its value")
            tokens[index + 1] = inner_command
            return shlex.join(tokens)
        if token.startswith("--command="):
            tokens[index] = f"--command={inner_command}"
            trailing_command_end = len(tokens)
            for trailing_index in range(index + 1, len(tokens)):
                if tokens[trailing_index] in outer_option_tokens:
                    trailing_command_end = trailing_index
                    break
            del tokens[index + 1 : trailing_command_end]
            return shlex.join(tokens)
    raise ValueError("No --command argument found to rewrite")


def strip_removed_metadata_options(command: str) -> str:
    tokens = shlex.split(command)
    if not tokens:
        return command
    stripped: list[str] = []
    skip_next = False
    removed_options = {"--case-id", "--label", "--mode"}
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in removed_options:
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in removed_options):
            continue
        stripped.append(token)
    return shlex.join(stripped)
