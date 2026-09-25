# LibContentBench OpenTuner harness

`experiments/LibContentBench/opentuner_integration/cli.py` runs OpenTuner against exactly five fixed-workload direct workers. `experiments/LibContentBench/run_opentuner.py` remains a thin compatibility entry point for the same CLI.

- `nestedcpu-blas`
- `arrow-dask-duckdb-memspill`
- `hf-datasets-tokenizers-dataloader`
- `torchdata-fsspec-s3-ddp`
- `dali-nvjpeg-pytorch-cudnn`

The static workflow metadata lives in `opentuner_integration/specs.py`. That module imports only stdlib plus `libcontentbench.direct_commands`, so static inspection of supported workflows and whitelisted tunables does not require importing OpenTuner. Runtime OpenTuner imports live in `opentuner_integration/harness.py` and `opentuner_integration/cli.py`.

The driver is pinned to exactly `opentuner==0.8.8` in `opentuner_integration/requirements.txt`. The CLI reads installed distribution metadata and exits before constructing the OpenTuner parser or starting a run when the version is missing or different. With no explicit `--technique`, OpenTuner 0.8.8 uses `AUCBanditMetaTechniqueA`; the trace records the exact per-request sub-technique from `DesiredResult.requestor` rather than inferring it from that default.

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

For successful trials, `Result.time` is the measured trial wall-clock time so OpenTuner's `cost time=...` display stays meaningful. By default, `--metric-direction auto` maximizes obvious throughput/rate metrics (`throughput`, `*_per_sec`, names containing `throughput`) and minimizes wall-time/latency metrics such as `wall_time_sec` through the separate objective value. Use `--tuning-goal minimize|maximize` to set the goal explicitly; it overrides `--metric-direction`.

Failed non-baseline trials remain searchable evidence but are always dominated: they return finite `accuracy=-sys.float_info.max` and `time=sys.float_info.max`. Their repository artifact preserves the specific `ERROR`, `TIMEOUT`, or `NO_METRIC` state, while `NO_METRIC` maps to OpenTuner's supported ORM `ERROR` state. Finite sentinels prevent failed Arrow/DALI candidates from becoming incumbents and keep JSON artifacts free of infinities and NaNs.

The baseline is identified by the canonical shared configuration fingerprint, not by trial number. If that baseline returns `ERROR`, `TIMEOUT`, or `NO_METRIC`, its trial artifact is written and the run raises `BaselineTrialFailed`. The CLI still finalizes the partial optimizer trace from the aborted database before surfacing the failure. Repository summaries and `best_config.json` independently select only `state == OK` artifacts; they do not trust OpenTuner's incumbent state.

## Per-trial artifacts

Each OpenTuner trial writes a `trials/trial_XXXX/trial_artifact.json` file under `--output-dir` with:

- the materialized command
- OpenTuner config values
- the full runner payload
- stdout/stderr
- return code / timeout state
- parsed metric value
- resolved metric direction and OpenTuner objective value
- canonical config fingerprint and baseline identity
- DesiredResult id, generation, requestor, and configuration id/hash when available
- independent `trial_wall_time_sec` measured around the `call_program(...)` subprocess launch
- result state
- finite OpenTuner result accuracy/time values
- the `output.json` path

The CLI also prints one line per completed trial. Successful tuning runs print a final summary with every trial's metric, the trial where the best metric was found, wall time until that trial completed, and total tuning wall time. A failed baseline aborts after evidence and trace finalization, so it surfaces the typed failure instead of printing a success-style final summary.

## Optimizer trace artifacts

Every completed or partially aborted run writes three compact deterministic artifacts under `--output-dir`:

- `optimizer_trace.jsonl`: one row per OpenTuner `DesiredResult`, ordered by `DesiredResult.id`. Rows include `schema_version=1`, optimizer/version identity, request index/id, generation, exact requestor, request state, config and canonical fingerprint, OpenTuner configuration id/hash, duplicate linkage, execution index, result reuse, result id/state/accuracy/time, and the persisted new-best transition.
- `trace_manifest.json`: exact version/preflight status, tuning-run state, configured/default technique information, baseline fingerprint, request ordering rule, and artifact inventory.
- `trace_summary.json`: request/execution/duplicate/reuse/incomplete counts, requestor and result-state counts, new-best transition count, and the best successful result selected independently with `result_state == OK`.

OpenTuner stores duplicate configuration requests as separate `DesiredResult` rows that may point to one reused `Result`. The trace retains every request, links it to the first request index for that configuration, marks only the measurement-owning request as executed, and marks later rows as reused. Technique attribution therefore comes from the database's exact requestor field even when no additional benchmark process ran.

Trace extraction is version-gated and uses the installed OpenTuner 0.8.8 ORM models against the run's SQLite database after the run closes. If the expected model columns are unavailable, the CLI raises an explicit `TraceAvailabilityError`; it never silently emits a trace without technique data.

## NestedCPU-BLAS deep-trace study mode

The shared normalized case-study ledger is opt-in and applies only to
`nestedcpu-blas`:

```bash
/home/yyan/miniconda3/envs/agent/bin/python \
  experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow nestedcpu-blas \
  --output-dir /tmp/lcb-opentuner/nestedcpu-study \
  --metric throughput \
  --tuning-goal maximize \
  --study-ledger /tmp/lcb-opentuner/nestedcpu-study/case_study.jsonl
```

Study mode keeps OpenTuner's native `DesiredResult` generation and SQLite result
reuse. The seeded baseline runs in the same tuning session but is excluded from
the 30 unique non-baseline execution target. The driver observes at most 60
native requests, stops after 30 unique non-baseline launches when possible, and
records every request in stable ORM id order with its persisted requestor. A
duplicate request reuses the first OpenTuner `Result` without relaunching the
benchmark. Failed, timed-out, and missing-metric first launches still count once
and retain finite dominated values. If 60 requests do not produce 30 unique
launches, the ledger ends with `proposal_cap_reached_with_shortfall`.

`--study-proposal-cap` and `--study-unique-execution-target` are bounded-smoke
overrides and require `--study-ledger`. Without `--study-ledger`, the existing
OpenTuner budget, artifacts, and behavior are unchanged.

## Usage

Use the agent Python environment:

```bash
/home/yyan/miniconda3/envs/agent/bin/python \
  experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow nestedcpu-blas \
  --output-dir /tmp/lcb-opentuner/nestedcpu \
  --metric throughput \
  --tuning-goal maximize \
  --test-limit 8
```

List workflows:

```bash
/home/yyan/miniconda3/envs/agent/bin/python \
  experiments/LibContentBench/opentuner_integration/cli.py \
  --list-workflows
```

The older `experiments/LibContentBench/run_opentuner.py` path is kept as a compatibility shim.

Each request uses one benchmark measurement. There are no in-loop repeats or variance aggregation, matching the existing fixed-budget comparison protocol. Duplicate requests can reuse an earlier result and therefore consume an OpenTuner request without launching another measurement; the trace exposes both request count and execution count so reports do not conflate them.

## Optional-workflow caveats

- `torchdata-fsspec-s3-ddp` still depends on Docker/MinIO and PyTorch DDP at runtime; tests for this harness mock subprocess execution and do not launch containers. The harness follows the selected small workflow's `quay.io/minio/minio:latest` image instead of Docker Hub's `minio/minio:latest`.
- `dali-nvjpeg-pytorch-cudnn` still depends on CUDA, DALI, and MPS for real runs; tests mock subprocess execution and do not import CUDA libraries or execute GPU work.
- Static spec inspection and unit tests stay host-safe and do not require OpenTuner to import until the runtime harness or CLI actually runs.
- Exact requestor and duplicate/reuse provenance are available only for the pinned OpenTuner 0.8.8 SQLite ORM schema. Other versions or non-SQLite database URLs fail explicitly instead of producing a reduced trace.
- `was_new_best` is OpenTuner's persisted result transition. Repository best selection remains separately state-filtered because optimizer-internal history is not the authority for report publication.
