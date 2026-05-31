# LibContentBench OpenTuner harness

`experiments/LibContentBench/opentuner_integration/cli.py` runs OpenTuner against exactly five fixed-workload direct workers. `experiments/LibContentBench/run_opentuner.py` remains a thin compatibility entry point for the same CLI.

- `nestedcpu-blas`
- `arrow-dask-duckdb-memspill`
- `hf-datasets-tokenizers-dataloader`
- `torchdata-fsspec-s3-ddp`
- `dali-nvjpeg-pytorch-cudnn`

The static workflow metadata lives in `opentuner_integration/specs.py`. That module imports only stdlib plus `libcontentbench.direct_commands`, so static inspection of supported workflows and whitelisted tunables does not require importing OpenTuner. Runtime OpenTuner imports live in `opentuner_integration/harness.py` and `opentuner_integration/cli.py`.

## Fairness rules

- The harness uses `libcontentbench/direct_commands.py` as the source of truth for command materialization.
- Each workflow loads its checked-in `tunable_whitelist.txt` and exposes only whitelist-authorized runtime/library controls.
- Workload-shape CLI flags stay fixed. OpenTuner does **not** tune matrix size, dataset rows, file counts, image counts, warmup/measure windows, shard counts, or similar workload-definition fields.
- NestedCPU alias handling is normalized onto runtime controls only:
  - `--joblib-n-jobs` is the canonical OpenTuner parameter for `outer_jobs`
  - `--outer-jobs` is treated as a whitelist alias for the same field
  - BLAS/OpenMP env vars are exposed separately so the seeded small workload keeps `OMP_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=6`, `MKL_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`

## Fixed workloads

The harness uses checked-in fixed workload payloads derived from `docs/libcontentbench_selected_benchmark_workflow_details.md`, which selects the workflow-size catalog's small tier for every supported benchmark except DALI/cuDNN and the medium tier for DALI/cuDNN:

- `nestedcpu-blas`: small fixed shape from `workflow_size_commands.md`
- `arrow-dask-duckdb-memspill`: small fixed shape from `workflow_size_commands.md`
- `hf-datasets-tokenizers-dataloader`: small fixed shape from `workflow_size_commands.md`
- `torchdata-fsspec-s3-ddp`: small fixed shape from `workflow_size_commands.md`
- `dali-nvjpeg-pytorch-cudnn`: medium fixed shape from `workflow_size_commands.md`

Only the whitelist-approved runtime/library controls vary across trials.

## Metric contract

The harness first parses metric-only JSON from direct-runner stdout. Use a metric key such as `wall_time_sec`:

```json
{"wall_time_sec": 12.34, "throughput": 99.1}
```

If stdout does not contain JSON, the harness falls back to the runner's `--output-json` artifact. The harness now keeps two separate values:

- `metric_value`: the raw runner metric
- `objective_value`: the value optimized through OpenTuner's objective field

`Result.time` is always the measured trial wall-clock time so OpenTuner's `cost time=...` display stays meaningful. By default, `--metric-direction auto` maximizes obvious throughput/rate metrics (`throughput`, `*_per_sec`, names containing `throughput`) and minimizes wall-time/latency metrics such as `wall_time_sec` through the separate objective value. Use `--tuning-goal minimize|maximize` to set the goal explicitly; it overrides `--metric-direction`.

Failing or timed-out trials return `Result(time=float('inf'))`.

## Per-trial artifacts

Each OpenTuner trial writes a `trials/trial_XXXX/trial_artifact.json` file under `--output-dir` with:

- the materialized command
- OpenTuner config values
- the full runner payload
- stdout/stderr
- return code / timeout state
- parsed metric value
- resolved metric direction and OpenTuner objective value
- independent `trial_wall_time_sec` measured around the `call_program(...)` subprocess launch
- result state
- the `output.json` path

The CLI also prints one line per completed trial with the metric value and prints a final summary with every trial's metric, the trial where the best metric was found, wall time until that trial completed, and total tuning wall time.

## Usage

Use the agent Python environment:

```bash
/home/yinengy/miniconda3/envs/agent/bin/python \
  experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow nestedcpu-blas \
  --output-dir /tmp/lcb-opentuner/nestedcpu \
  --metric throughput \
  --tuning-goal maximize \
  --test-limit 8
```

List workflows:

```bash
/home/yinengy/miniconda3/envs/agent/bin/python \
  experiments/LibContentBench/opentuner_integration/cli.py \
  --list-workflows
```

The older `experiments/LibContentBench/run_opentuner.py` path is kept as a compatibility shim.

## Optional-workflow caveats

- `torchdata-fsspec-s3-ddp` still depends on Docker/MinIO and PyTorch DDP at runtime; tests for this harness mock subprocess execution and do not launch containers. The harness follows the selected small workflow's `quay.io/minio/minio:latest` image instead of Docker Hub's `minio/minio:latest`.
- `dali-nvjpeg-pytorch-cudnn` still depends on CUDA, DALI, and MPS for real runs; tests mock subprocess execution and do not import CUDA libraries or execute GPU work.
- Static spec inspection and unit tests stay host-safe and do not require OpenTuner to import until the runtime harness or CLI actually runs.
