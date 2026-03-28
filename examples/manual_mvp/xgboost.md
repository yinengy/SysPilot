# XGBoost CPU Inference And Training

## Overview
XGBoost relies on thread pools and memory bandwidth. Default predictor settings can be stable, but CPU-only workloads still need thread and NUMA validation.

## Controls
- Config key `nthread` defaults to the process thread count.
- Config key `predictor=cpu_predictor` keeps CPU inference explicit.
- Warning: aggressive thread counts can conflict with OpenMP oversubscription.

## Diagnostics
- Symptom: low throughput with high context switching.
- Command: `pidstat -t -p <pid> 1 5`
- Command: `perf stat -p <pid>`

## Validation
- Compare requests/sec before and after changing `nthread`.
- Record model latency p95 and CPU migration counters.
