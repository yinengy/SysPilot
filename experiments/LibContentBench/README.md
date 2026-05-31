# LibContentBench

LibContentBench is a synthetic multi-library contention benchmark suite for SysPilot.

The suite is intentionally synthetic: workloads use generated data and bounded runtimes, but each benchmark is built from real libraries and real configuration knobs. The goal is to isolate the claim that library-aware tuning can coordinate hidden per-library controls more effectively than application-only tuning.

## Benchmarks

- `nestedcpu-blas/` — CPU and scheduler contention from joblib outer parallelism plus inner BLAS threading.
- `arrow-dask-duckdb-memspill/` — host-memory and spill-path contention across Dask, PyArrow, and DuckDB.
- `hdf5-mpiio-write/` — HPC parallel I/O workspace with native HDF5 probing plus MPI-IO fallback evidence when parallel HDF5 is unavailable.
- `horovod-tfdata-network/` — optional network-contention workspace with readiness-first blocked reporting.
- `torchdata-checkpoint-io/` — optional local checkpoint/log write contention workspace using PyTorch DataLoader plus real `torch.save` and TensorBoard writes.
- `pytorch-h5py-hdf5-dataloader/` — local HDF5 chunk-cache contention workspace using PyTorch DataLoader plus real `h5py` raw-data chunk cache controls.
- `hf-datasets-tokenizers-dataloader/` — local text-processing contention workspace using Hugging Face `datasets`, Hugging Face `tokenizers`, and PyTorch DataLoader.
- `dask-zarr-blosc/` — local chunked-array contention workspace using Dask workers, Zarr stores, and real Blosc compression settings.
- `torchdata-fsspec-s3-ddp/` — optional single-host object-store input workspace using PyTorch DataPipes, `fsspec`/`s3fs`, local MinIO, and PyTorch DDP with loopback-scope caveats.
- `adios2-bp5-sst/` — local ADIOS2 BP5 file-backed HPC I/O workspace with documented SST readiness/probe scope.
- `raydata-ddp-network/` — optional single-host-first Ray Data plus PyTorch DDP workspace with explicit loopback/shared-memory caveats.
- `dali-nvjpeg-pytorch-cudnn/` — optional single-GPU CUDA MPS contention workspace using NVIDIA DALI/nvJPEG-style preprocessing plus PyTorch/cuDNN inference.

## Common interface

Suite-level orchestration is now prepare/report only. From this suite root, use:

```bash
python3 run_suite.py prepare
python3 run_suite.py summarize
```

For runnable benchmark commands, use the direct worker command blocks in `summary_repro.md` or the benchmark-local README.

For future LPCL work that maps benchmark-local `BENCHMARK_*` env overrides into original library contracts, follow `LPCL_BENCHMARK_ENV_MAPPING_SKILL.md`.

For OpenTuner-based fixed-workload search over the supported direct workers, see [`OPENTUNER.md`](OPENTUNER.md).

## Output contract

Each benchmark workspace owns:

- `prepare.sh`
- `build_report.py`
- `summary.md`
- `result_table.csv`
- `comparison_report.json`
- `acceptance_report.json`
- `runs/` with per-case `result.json`, command, env, stdout/stderr, and metrics artifacts

Direct benchmark runners that are used for SysPilot tuning print metric-only JSON to stdout
(`wall_time_sec`, throughput/rate, latency percentile, or benchmark score fields). Full
reproducibility details such as benchmark identity, config/env snapshots, resource signals,
and profiling summaries stay in the `--output-json` artifact instead of terminal stdout.

The suite-level `summary.md` also includes a human-readable per-benchmark section with status, baseline command/config, best archived command/config, and the recorded performance numbers. The direct-only reproduction surface lives in `summary_repro.md`, which keeps runnable benchmark commands stable even if old `runs/` artifacts are cleaned up.

## Evidence interpretation notes

- Benchmark-local `build_report.py` outputs are cumulative over archived `runs/*/case_*/result.json` artifacts. This preserves historical failures and retries for reproducibility, so `result_table.csv` and `comparison_report.json` may include both older failed attempts and newer successful reruns for the same logical case.
- For a pristine paper/demo snapshot, regenerate reports from a clean run directory set instead of assuming the checked-in report tables represent only the latest execution wave.
- On the current validation host, `arrow-dask-duckdb-memspill` remains runnable and useful as a coordination benchmark, but the successful archived runs did not show nonzero DuckDB temp or Dask local spill bytes. Treat its current `no_improvement` result as host-specific evidence that library-aware knobs did not help this workload here, not as proof that the workload exercised strong spill-path pressure.
- `torchdata-checkpoint-io` intentionally uses the standard PyTorch `DataLoader` path as its default runnable implementation rather than requiring the separate `torchdata` package, because the suite cares about real documented input-pipeline knobs more than a specific package name.
- `torchdata-fsspec-s3-ddp` uses the PyTorch-provided DataPipe API (`torch.utils.data.datapipes.iter`) because the currently installed `torchdata 0.11` package no longer exposes `torchdata.datapipes`; its object-store path uses a local MinIO container and should be interpreted as single-host/loopback S3-compatible evidence.
- `adios2-bp5-sst` runs BP5 by default and records SST as readiness/probe metadata only; do not present SST throughput claims until a bounded coupled streaming reader/writer case is added and validated.
- `raydata-ddp-network` is packaged as a credible single-host distributed benchmark on this host. Even when physical NIC counters change, its checked-in reports still limit the claim to single-host evidence rather than overclaiming inter-machine network contention.
- `dali-nvjpeg-pytorch-cudnn` is packaged as a runnable GPU benchmark but checked in with scaffold evidence only; do not claim measured DALI/cuDNN throughput until it is run on a CUDA+DALI host with MPS available.

## Main claim

Library-level information helps because it exposes hidden knobs from individual libraries that are not visible at the application level. Those knobs can coordinate libraries that share CPU, memory, I/O, or network resources. LibContentBench is designed to isolate that effect with real libraries and configuration-only tuning.
