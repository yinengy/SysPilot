# LibContentBench

LibContentBench is a synthetic multi-library contention benchmark suite for SysPilot.

The suite is intentionally synthetic: workloads use generated data and bounded runtimes, but each benchmark is built from real libraries and real configuration knobs. The goal is to isolate the claim that library-aware tuning can coordinate hidden per-library controls more effectively than application-only tuning.

## Benchmarks

- `nestedcpu-blas/` — CPU and scheduler contention from joblib outer parallelism plus inner BLAS threading.
- `arrow-dask-duckdb-memspill/` — host-memory and spill-path contention across Dask, PyArrow, and DuckDB.
- `hdf5-mpiio-write/` — HPC parallel I/O workspace with native HDF5 probing plus MPI-IO fallback evidence when parallel HDF5 is unavailable.
- `horovod-tfdata-network/` — optional network-contention workspace with readiness-first blocked reporting.
- `torchdata-checkpoint-io/` — optional local checkpoint/log write contention workspace using PyTorch DataLoader plus real `torch.save` and TensorBoard writes.
- `raydata-ddp-network/` — optional single-host-first Ray Data plus PyTorch DDP workspace with explicit loopback/shared-memory caveats.

## Common interface

From this suite root, use:

```bash
python3 run_suite.py prepare
python3 run_suite.py smoke
python3 run_suite.py run-baseline
python3 run_suite.py run-application-only
python3 run_suite.py run-library-aware
python3 run_suite.py run-oracle
python3 run_suite.py summarize
```

Use `--include-optional` to include the optional benchmark workspaces in suite execution.

## Output contract

Each benchmark workspace owns:

- `prepare.sh`
- `matrix_smoke.yaml`
- `matrix.yaml`
- `run_matrix.py`
- `build_report.py`
- `summary.md`
- `result_table.csv`
- `comparison_report.json`
- `acceptance_report.json`
- `runs/` with per-case `result.json`, command, env, stdout/stderr, and metrics artifacts

The suite-level `summary.md` also includes a human-readable per-benchmark section with status, baseline command/config, best archived command/config, and the recorded performance numbers. Those commands are derived from the current benchmark `matrix.yaml` case IDs so they remain reproducible even if old `runs/` artifacts are cleaned up.

## Evidence interpretation notes

- Benchmark-local `build_report.py` outputs are cumulative over archived `runs/*/case_*/result.json` artifacts. This preserves historical failures and retries for reproducibility, so `result_table.csv` and `comparison_report.json` may include both older failed attempts and newer successful reruns for the same logical case.
- For a pristine paper/demo snapshot, regenerate reports from a clean run directory set instead of assuming the checked-in report tables represent only the latest execution wave.
- On the current validation host, `arrow-dask-duckdb-memspill` remains runnable and useful as a coordination benchmark, but the successful archived runs did not show nonzero DuckDB temp or Dask local spill bytes. Treat its current `no_improvement` result as host-specific evidence that library-aware knobs did not help this workload here, not as proof that the workload exercised strong spill-path pressure.
- `torchdata-checkpoint-io` intentionally uses the standard PyTorch `DataLoader` path as its default runnable implementation rather than requiring the separate `torchdata` package, because the suite cares about real documented input-pipeline knobs more than a specific package name.
- `raydata-ddp-network` is packaged as a credible single-host distributed benchmark on this host. Even when physical NIC counters change, its checked-in reports still limit the claim to single-host evidence rather than overclaiming inter-machine network contention.

## Main claim

Library-level information helps because it exposes hidden knobs from individual libraries that are not visible at the application level. Those knobs can coordinate libraries that share CPU, memory, I/O, or network resources. LibContentBench is designed to isolate that effect with real libraries and configuration-only tuning.
