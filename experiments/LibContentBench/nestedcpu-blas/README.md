# NestedCPU-BLAS

This benchmark creates CPU oversubscription from two independent layers of parallelism:

- outer task parallelism from `joblib`
- inner native threading from NumPy's BLAS backend

It is synthetic on purpose: the data is random and the output has no semantic meaning, but the contention comes from real joblib scheduling and real BLAS thread pools.

## Layout

- `prepare.sh` — creates a local virtualenv and records threadpool/backend metadata.
- archived case catalogs and reports — preserved for historical baseline/application/library-aware/oracle evidence.
- `run_nestedcpu_blas_case.py` — direct benchmark runner with explicit CLI workload flags and real BLAS env controls.
- `build_report.py` — rebuilds `result_table.csv`, `comparison_report.json`, `acceptance_report.json`, and `summary.md` from archived run artifacts.
- `work/knob_visibility.json` — explicit visibility split between application-only and library-aware tuning.

## Modes

- `baseline` — bad nested oversubscription.
- `application_only` — only coarse outer concurrency/workload knobs vary.
- `library_aware` — outer concurrency and BLAS thread knobs are coordinated.
- `oracle` — bounded sanity grid.

## Usage

```bash
bash prepare.sh
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 BLIS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
  artifact/venv/bin/python run_nestedcpu_blas_case.py \
  --case-id baseline_direct \
  --label "baseline direct" \
  --mode baseline \
  --output-json work/direct_result.json \
  --matrix-size 320 \
  --task-count 8 \
  --repetitions 6 \
  --joblib-n-jobs 6 \
  --backend-prefer processes
artifact/venv/bin/python build_report.py
```

For packaged baseline/best direct reruns, use `../summary_repro.md`.

## Fixed-baseline SysPilot tuning surface

To let `syspilot tune` keep the packaged bad baseline workload fixed while changing only the nested parallelism controls, use the worker directly instead of `run_matrix.py`:

- `--joblib-n-jobs` / `--outer-jobs` for `joblib.Parallel(n_jobs=...)`
- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `BLIS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`

These are the real joblib and BLAS/OpenMP controls that the benchmark uses at runtime. The runnable benchmark surface for SysPilot is the direct worker command plus real thread-control env vars. Workload-shape fields like matrix size, task count, and repetition count stay fixed unless you intentionally change the direct CLI flags.
