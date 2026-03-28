# XGBoost Training Worker

This workload trains boosted trees with xgboost on shared CPU hosts.

## Shared Libraries
- Depends on xgboost and openmp.

## Goal
- Improve training throughput while keeping memory stable.

## Warning
- Combining `nthread=32` with `OMP_NUM_THREADS=32` can oversubscribe the host.
