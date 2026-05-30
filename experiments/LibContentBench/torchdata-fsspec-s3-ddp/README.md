# TorchData-FSSpec-S3-DDP

This optional LibContentBench workspace packages a real single-host benchmark built from:

- TorchData-style PyTorch DataPipe `IterableWrapper` pipelines
- `fsspec` / `s3fs` reads against a local MinIO S3-compatible object store
- PyTorch `DistributedDataParallel` on CPU with the Gloo backend

The workload uses synthetic numeric records, but the contention comes from real object-store reads, real DataPipe iteration, and real DDP gradient synchronization. Current `torchdata` releases no longer expose `torchdata.datapipes`, so the runner uses the PyTorch-provided `torch.utils.data.datapipes.iter` API while still recording the installed `torchdata` package version when present.

## Honest scope

- The packaged path is **single-host first** and uses a **local MinIO container** so the object-store traffic is credible but bounded.
- Because the object store is local to this host, network activity may stay on **loopback**. This workspace records that explicitly and does **not** overclaim multi-machine network contention.
- If Docker, MinIO image pulls, or PyTorch distributed dependencies are unavailable, the workspace records an explicit blocker instead of silently skipping.

## Layout

- `prepare.sh` — creates the local virtualenv, installs runtime dependencies, and records readiness metadata.
- archived case catalogs and reports — preserved for historical baseline/application/library-aware/oracle evidence.
- `run_torchdata_fsspec_s3_ddp_case.py` — single case runner.
- `build_report.py` — rebuilds `result_table.csv`, `comparison_report.json`, `acceptance_report.json`, and `summary.md` from archived run artifacts.
- `work/knob_visibility.json` — explicit split between application-only and library-aware tuning surfaces.

## Knob visibility

- `application_only` may change only coarse model-shape knobs like hidden width.
- `library_aware` may additionally change Torch CPU thread count, `s3fs` block/cache behavior, and DDP `bucket_cap_mb`.
- `oracle` is a bounded local sanity subset, not an exhaustive search.

## Usage

```bash
bash prepare.sh
artifact/venv/bin/python build_report.py
```

Run the packaged direct worker commands from `../summary_repro.md` when you want baseline or best archived repro commands.

## Fixed-baseline SysPilot tuning surface

To let `syspilot tune` keep the packaged baseline workload fixed while changing only real library knobs, the worker reads these optional environment overrides:

- `BENCHMARK_TORCH_NUM_THREADS`
- `BENCHMARK_S3_BLOCK_SIZE`
- `BENCHMARK_S3_CACHE_TYPE`
- `BENCHMARK_S3_USE_LISTINGS_CACHE`
- `BENCHMARK_DDP_BUCKET_CAP_MB`

These map directly onto documented PyTorch CPU runtime, `s3fs.S3FileSystem`, and DDP controls. Workload-shape fields like samples per rank, shard count, input dimension, batch size, and epoch count stay fixed unless you intentionally change the direct CLI workload shape.
