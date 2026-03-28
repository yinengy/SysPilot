# OpenMP Runtime Tuning

## Overview
OpenMP controls CPU thread parallelism for libraries such as xgboost. Warning: setting OMP_NUM_THREADS above physical cores can reduce throughput.

## Controls
- Environment variable `OMP_NUM_THREADS` defaults to the runtime thread count.
- Environment variable `OMP_PROC_BIND=close` helps reduce cross-socket movement on NUMA hosts.
- CLI flag `--threads` should stay aligned with OpenMP threads when an application exposes both knobs.

## Diagnostics
- Symptom: throughput drops while CPU utilization stays saturated.
- Command: `perf stat -p <pid>`
- Command: `numactl --show`

## Validation
- Run the workload with 4, 8, and 16 threads.
- Keep the smallest thread count that preserves throughput without oversubscription.
