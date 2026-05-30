# ArrowDask-DuckDB-MemSpill

This benchmark creates host-memory and scratch-path contention across three real libraries:

- Dask distributed worker concurrency
- PyArrow parquet scan thread pools
- DuckDB query memory and temp-directory usage

The workload is synthetic, but contention comes from real Parquet reads, real Dask workers, and real DuckDB operators.

## Usage

```bash
bash prepare.sh
artifact/venv/bin/python build_report.py
```

Run the packaged direct worker commands from `../summary_repro.md` when you want baseline or best archived repro commands.

## Fixed-baseline SysPilot tuning surface

To let `syspilot tune` keep the packaged baseline workload fixed while changing real library knobs, the worker accepts these optional environment overrides:

- `BENCHMARK_N_WORKERS`
- `BENCHMARK_THREADS_PER_WORKER`
- `BENCHMARK_WORKER_MEMORY_LIMIT`
- `BENCHMARK_ARROW_CPU_COUNT`
- `BENCHMARK_ARROW_IO_THREADS`
- `BENCHMARK_DUCKDB_MEMORY_LIMIT`
- `BENCHMARK_DUCKDB_THREADS`
- `BENCHMARK_PATH_LAYOUT`

These cover the benchmark's Dask worker concurrency, PyArrow scan thread pools, DuckDB execution/memory settings, and the shared-vs-isolated scratch-path layout. Synthetic workload-shape fields like file count, rows per file, and row-group size stay fixed unless you intentionally choose a different direct workload shape.

For SysPilot tuning, pass the original `dask`, `duckdb`, and `pyarrow` LPCL contracts. Their controls intentionally use the `BENCHMARK_*` environment-variable names above so recommendations can change the benchmark's real fixed-workload override surface directly.
