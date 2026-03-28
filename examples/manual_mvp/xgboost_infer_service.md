# XGBoost Inference Service

This workload serves xgboost predictions on a latency-sensitive API.

## Shared Libraries
- Depends on xgboost and openmp.

## Goal
- Reduce p95 latency without sacrificing steady-state throughput.

## Warning
- Large thread pools can increase tail latency during concurrent requests.
