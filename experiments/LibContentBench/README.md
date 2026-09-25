# LibContentBench

LibContentBench is a multi-library contention benchmark suite for CONCORD.

The paper evaluates nine workloads: five original fixed-workload contention
benchmarks and four real-data applications. Every workload uses real libraries
and real configuration knobs; synthetic inputs keep repeated comparisons
affordable, while the real-data workloads test whether the same contention
reasoning transfers to representative applications.

## Benchmarks

- `nestedcpu-blas/` — CPU and scheduler contention from joblib outer parallelism plus inner BLAS threading.
- `arrow-dask-duckdb-memspill/` — host-memory and spill-path contention across Dask, PyArrow, and DuckDB.
- `hf-datasets-tokenizers-dataloader/` — local text-processing contention workspace using Hugging Face `datasets`, Hugging Face `tokenizers`, and PyTorch DataLoader.
- `torchdata-fsspec-s3-ddp/` — optional single-host object-store input workspace using PyTorch DataPipes, `fsspec`/`s3fs`, local MinIO, and PyTorch DDP with loopback-scope caveats.
- `dali-nvjpeg-pytorch-cudnn/` — optional single-GPU CUDA MPS contention workspace using NVIDIA DALI/nvJPEG-style preprocessing plus PyTorch/cuDNN inference.
- `forest-cover-model-selection/` — Dask-ML and XGBoost model selection on UCI Covertype.
- `movie-recommendation-training/` — implicit-feedback ALS training on MovieLens20M.
- `scientific-claim-retrieval/` — SentenceTransformers and FAISS retrieval on SciFact.
- `sift-visual-codebook-construction/` — Spark MLlib codebook construction on SIFT1M descriptors.

## Common interface

Suite-level orchestration is now prepare/report only. From this suite root, use:

```bash
python3 run_suite.py prepare
python3 run_suite.py summarize
```

For runnable benchmark commands, use the command blocks in this README or the
benchmark-local README.

For future LPCL work that maps benchmark-local `BENCHMARK_*` env overrides into original library contracts, follow `LPCL_BENCHMARK_ENV_MAPPING_SKILL.md`.

For OpenTuner-based fixed-workload search over the supported direct workers, see [`OPENTUNER.md`](OPENTUNER.md).

## Output contract

Each benchmark workspace owns:

- `prepare.sh`
- `build_report.py`
- `summary.md`
- `result_table.csv`
- `comparison_report.json`
- `acceptance_report.json`
- `runs/` with per-case `result.json`, command, env, stdout/stderr, and metrics artifacts

Direct benchmark runners that are used for SysPilot tuning print metric-only JSON to stdout
(`wall_time_sec`, throughput/rate, latency percentile, or benchmark score fields). Full
reproducibility details such as benchmark identity, config/env snapshots, resource signals,
and profiling summaries stay in the `--output-json` artifact instead of terminal stdout.

Generated suite reports are runtime artifacts. The reproducibility surface for
the paper is this README plus the benchmark-local source and setup files.

## Evidence interpretation notes

- Benchmark-local `build_report.py` outputs are cumulative over archived `runs/*/case_*/result.json` artifacts. This preserves historical failures and retries for reproducibility, so `result_table.csv` and `comparison_report.json` may include both older failed attempts and newer successful reruns for the same logical case.
- For a pristine paper/demo snapshot, regenerate reports from a clean run directory set instead of assuming the checked-in report tables represent only the latest execution wave.
- On the current validation host, `arrow-dask-duckdb-memspill` remains runnable and useful as a coordination benchmark, but the successful archived runs did not show nonzero DuckDB temp or Dask local spill bytes. Treat its current `no_improvement` result as host-specific evidence that library-aware knobs did not help this workload here, not as proof that the workload exercised strong spill-path pressure.
- `torchdata-fsspec-s3-ddp` uses the PyTorch-provided DataPipe API (`torch.utils.data.datapipes.iter`) because the currently installed `torchdata 0.11` package no longer exposes `torchdata.datapipes`; its object-store path uses a local MinIO container and should be interpreted as single-host/loopback S3-compatible evidence.
- `dali-nvjpeg-pytorch-cudnn` is packaged as a runnable GPU benchmark but checked in with scaffold evidence only; do not claim measured DALI/cuDNN throughput until it is run on a CUDA+DALI host with MPS available.

## Main claim

Library-level information helps because it exposes hidden knobs from individual libraries that are not visible at the application level. Those knobs can coordinate libraries that share CPU, memory, I/O, or network resources. LibContentBench is designed to isolate that effect with real libraries and configuration-only tuning.

Run `prepare.sh` once in each
workload directory first. The fixed workload arguments must remain unchanged
when comparing CONCORD, SMAC3, and OpenTuner; only the listed runtime/library
controls are tunable. The four real-data application commands follow below.

### Nested CPU and BLAS

```bash
cd /home/yyan/SysPilot
bash experiments/LibContentBench/nestedcpu-blas/prepare.sh
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    BLIS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
  experiments/LibContentBench/nestedcpu-blas/artifact/venv/bin/python \
  experiments/LibContentBench/nestedcpu-blas/run_nestedcpu_blas_case.py \
  --case-id bad_outer6_inner6 --label 'bad baseline outer6 inner6' \
  --mode baseline --output-json experiments/LibContentBench/nestedcpu-blas/work/direct_result.json \
  --matrix-size 320 --task-count 8 --repetitions 6 \
  --joblib-n-jobs 6 --backend-prefer processes
```

### Arrow, Dask, and DuckDB

```bash
cd /home/yyan/SysPilot
bash experiments/LibContentBench/arrow-dask-duckdb-memspill/prepare.sh
env BENCHMARK_N_WORKERS=6 BENCHMARK_THREADS_PER_WORKER=2 \
    BENCHMARK_WORKER_MEMORY_LIMIT=256MB BENCHMARK_ARROW_CPU_COUNT=8 \
    BENCHMARK_ARROW_IO_THREADS=8 BENCHMARK_DUCKDB_MEMORY_LIMIT=1GB \
    BENCHMARK_DUCKDB_THREADS=8 BENCHMARK_PATH_LAYOUT=shared \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/artifact/venv/bin/python \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/run_arrow_dask_duckdb_case.py \
  --case-id wide_bad_shared_w6_t2_arrow8_duck8_mem256 \
  --label 'wide bad baseline shared heavy oversubscription' --mode baseline \
  --output-json experiments/LibContentBench/arrow-dask-duckdb-memspill/work/direct_result.json \
  --runtime-root /var/tmp/libcontentbench/arrow-dask-duckdb-memspill \
  --num-files 10 --rows-per-file 20000 --row-group-size 5000
```

### Hugging Face data processing and tokenization

```bash
cd /home/yyan/SysPilot
bash experiments/LibContentBench/hf-datasets-tokenizers-dataloader/prepare.sh
env BENCHMARK_TORCH_NUM_THREADS=16 BENCHMARK_DATASETS_NUM_PROC=4 \
    BENCHMARK_TOKENIZERS_PARALLELISM=false BENCHMARK_NUM_WORKERS=8 \
    BENCHMARK_PREFETCH_FACTOR=8 BENCHMARK_PERSISTENT_WORKERS=true \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/artifact/venv/bin/python \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/run_hf_datasets_tokenizers_case.py \
  --case-id search_bad_map8_tokpar_on_workers16_threads24 \
  --label 'search bad map8 tokpar on workers16 threads24' --mode baseline \
  --output-json experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/direct_result.json \
  --runtime-root /var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader \
  --dataset-rows 4096 --tokens-per-row 32 --max-length 64 \
  --tokenizer-vocab-size 1024 --hidden-dim 128 --num-classes 32 \
  --batch-size 32 --epochs 2 --learning-rate 0.001 \
  --tokenize-batch-size 128 --timeout-seconds 600
```

### TorchData, FSSpec, and S3-compatible input

```bash
cd /home/yyan/SysPilot
bash experiments/LibContentBench/torchdata-fsspec-s3-ddp/prepare.sh
env BENCHMARK_TORCH_NUM_THREADS=16 BENCHMARK_S3_BLOCK_SIZE=65536 \
    BENCHMARK_S3_CACHE_TYPE=none BENCHMARK_S3_USE_LISTINGS_CACHE=true \
    BENCHMARK_DDP_BUCKET_CAP_MB=1 \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/artifact/venv/bin/python \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/run_torchdata_fsspec_s3_ddp_case.py \
  --case-id search_bad_threads24_block4k_none_bucket1 \
  --label 'search bad threads24 block4k none bucket1' --mode baseline \
  --output-json experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/direct_result.json \
  --case-output-dir experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/direct_case_output \
  --runtime-root /var/tmp/libcontentbench/torchdata-fsspec-s3-ddp \
  --world-size 2 --samples-per-rank 1024 --shards-per-rank 4 \
  --input-dim 256 --hidden-dim 512 --num-classes 32 --batch-size 32 \
  --epochs 2 --learning-rate 0.01 --timeout-seconds 420 \
  --minio-image quay.io/minio/minio:latest
```

### DALI and PyTorch/cuDNN

This workload requires a CUDA+DALI host for a performance run. On a host with
the required GPU stack, the baseline is:

```bash
cd /home/yyan/SysPilot
bash experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/prepare.sh
experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/artifact/venv/bin/python \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/run_dali_nvjpeg_pytorch_cudnn_case.py \
  --case-id baseline_dali_b64_prefetch2_hw90_mps20_80 \
  --label 'baseline DALI b64 prefetch2 hw90 MPS20/80' --mode baseline \
  --output-json experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/direct_result.json \
  --case-output-dir experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/direct_case_output \
  --runtime-root /var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn \
  --warmup-seconds 10 --measure-seconds 45 --dataset-images 5000 \
  --image-min-size 768 --image-max-size 1024 --jpeg-quality 90 \
  --num-classes 16 --resize-size 256 --crop-size 224 \
  --dali-output-dtype float16 --dali-decode-backend mixed \
  --torch-model custom_cnn --torch-precision tf32
```

Each baseline runner writes the metric JSON to `--output-json` and prints the
primary metric on stdout. The original workloads use `throughput`, except the
DALI workload, which uses the combined `score` metric.

### Forest-Cover Model Selection

```bash
cd experiments/LibContentBench/forest-cover-model-selection
WORK=/var/tmp/libcontentbench/forest-cover-model-selection
python run.py prepare --data-dir "$WORK/data"

# Baseline
DASK_CV_WORKERS=0 DASK_XGB_THREADS=0 DASK_SCHEDULER=threads \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

# CONCORD-selected configuration
DASK_CV_WORKERS=26 DASK_XGB_THREADS=6 DASK_SCHEDULER=threads \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/concord" \
  --reference "$WORK/baseline"
```

### Movie Recommendation Training

```bash
cd experiments/LibContentBench/movie-recommendation-training
WORK=/var/tmp/libcontentbench/movie-recommendation-training
python run.py prepare --data-dir "$WORK/data"

# Baseline
REC_THREADS=0 OPENBLAS_NUM_THREADS=48 OMP_PROC_BIND=false OMP_PLACES=cores \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

# CONCORD-selected configuration
REC_THREADS=24 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/concord" \
  --reference "$WORK/baseline"
```

### Scientific Claim Retrieval

```bash
cd experiments/LibContentBench/scientific-claim-retrieval
WORK=/var/tmp/libcontentbench/scientific-claim-retrieval
python run.py prepare --data-dir "$WORK/data"

# Baseline
ST_PROCESSES=4 ST_INTRA_THREADS=0 ST_TOKENIZER_THREADS=0 \
ST_BATCH_SIZE=32 ST_CHUNK_SIZE=0 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

# CONCORD-selected configuration
ST_PROCESSES=2 ST_INTRA_THREADS=0 ST_TOKENIZER_THREADS=0 \
ST_BATCH_SIZE=18 ST_CHUNK_SIZE=384 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/concord" \
  --reference "$WORK/baseline"
```

### SIFT Visual Codebook Construction

```bash
cd experiments/LibContentBench/sift-visual-codebook-construction
WORK=/var/tmp/libcontentbench/sift-visual-codebook-construction
python run.py prepare --data-dir "$WORK/data"

# Baseline
SPARK_TASK_CPUS=0 SPARK_BLAS_THREADS=0 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

# CONCORD-selected configuration
SPARK_TASK_CPUS=3 SPARK_BLAS_THREADS=1 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/concord" \
  --reference "$WORK/baseline"
```

Each tuned run checks its numerical or application output against the baseline
and prints `wall_time_sec`. Run the baseline and tuned commands serially, then
compare `baseline wall time / tuned wall time`.

These four `run.py` commands are direct workload reproductions. They do not
invoke an optimizer. The optimizer entry points for the suite are the existing
drivers below.

## Optimizer entry points

The generic OpenTuner and SMAC3 drivers are still used for the original
LibContentBench workflows. They materialize each trial through the shared
direct-command layer, keep the workload shape fixed, and write per-trial traces
under the requested output directory. Keep one trial at a time (`--parallelism
1`, `--pipelining 0`, and `--n-workers 1`) because concurrent trials would
contend with one another.

The paper comparison used the following commands. SMAC3 uses its dedicated
`smac3/.venv` environment; OpenTuner uses the agent environment. Set
`PYTHONHASHSEED=0` for every run. Each command writes its trace to a separate
directory under `/var/tmp`.

```bash
cd /home/yyan/SysPilot
OT=/home/yyan/miniconda3/envs/agent/bin/python
SMAC=experiments/LibContentBench/smac3/.venv/bin/python

# SMAC3: --budget includes the baseline seed.
PYTHONHASHSEED=0 $SMAC experiments/LibContentBench/smac3/cli.py \
  --workflow nestedcpu-blas --size small --budget 30 \
  --metric throughput --metric-mode maximize --output-dir /var/tmp/libcontentbench/smac3-nestedcpu-b30 \
  --seed 0 --n-workers 1
PYTHONHASHSEED=0 $SMAC experiments/LibContentBench/smac3/cli.py \
  --workflow arrow-dask-duckdb-memspill --size small --budget 30 \
  --metric throughput --metric-mode maximize --output-dir /var/tmp/libcontentbench/smac3-arrow-b30 \
  --seed 0 --n-workers 1
PYTHONHASHSEED=0 $SMAC experiments/LibContentBench/smac3/cli.py \
  --workflow hf-datasets-tokenizers-dataloader --size small --budget 30 \
  --metric throughput --metric-mode maximize --output-dir /var/tmp/libcontentbench/smac3-hf-b30 \
  --seed 0 --n-workers 1
PYTHONHASHSEED=0 $SMAC experiments/LibContentBench/smac3/cli.py \
  --workflow torchdata-fsspec-s3-ddp --size small --budget 30 \
  --metric samples_per_sec --metric-mode maximize --output-dir /var/tmp/libcontentbench/smac3-s3-b30 \
  --seed 0 --n-workers 1
PYTHONHASHSEED=0 $SMAC experiments/LibContentBench/smac3/cli.py \
  --workflow dali-nvjpeg-pytorch-cudnn --size medium --budget 30 \
  --metric score --metric-mode maximize --output-dir /var/tmp/libcontentbench/smac3-dali-b30 \
  --seed 0 --n-workers 1

# OpenTuner: --test-limit is the request budget; keep trials serial.
PYTHONHASHSEED=0 $OT experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow nestedcpu-blas --output-dir /var/tmp/libcontentbench/opentuner-nestedcpu-b30 \
  --metric throughput --tuning-goal maximize --test-limit 30 --parallelism 1
PYTHONHASHSEED=0 $OT experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow arrow-dask-duckdb-memspill --output-dir /var/tmp/libcontentbench/opentuner-arrow-b30 \
  --metric throughput --tuning-goal maximize --test-limit 30 --parallelism 1
PYTHONHASHSEED=0 $OT experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow hf-datasets-tokenizers-dataloader --output-dir /var/tmp/libcontentbench/opentuner-hf-b30 \
  --metric throughput --tuning-goal maximize --test-limit 30 --parallelism 1
PYTHONHASHSEED=0 $OT experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow torchdata-fsspec-s3-ddp --output-dir /var/tmp/libcontentbench/opentuner-s3-b30 \
  --metric samples_per_sec --tuning-goal maximize --test-limit 30 --parallelism 1
PYTHONHASHSEED=0 $OT experiments/LibContentBench/opentuner_integration/cli.py \
  --workflow dali-nvjpeg-pytorch-cudnn --output-dir /var/tmp/libcontentbench/opentuner-dali-b30 \
  --metric score --tuning-goal maximize --test-limit 30 --parallelism 1
```

The CONCORD contract and manual commands use the same baseline workers and
fixed workload flags, with `--fixed-budget 30`; the exact contract names,
objective flags, and whitelist paths are recorded in
[`docs/fixed_budget_compare_exp_prompt.md`](../../docs/fixed_budget_compare_exp_prompt.md).
For every workload, the contract run uses the library contracts listed there;
the manual run adds `--manual <workload-name> --no-profile`.

The active files for these paths are
`opentuner_integration/cli.py`, `opentuner_integration/harness.py`,
`opentuner_integration/specs.py`, and `opentuner_integration/trace.py` for
OpenTuner, and `smac3/cli.py`, `smac3/optimize.py`, `smac3/runner.py`,
`smac3/workflows.py`, and `smac3/trace_callback.py` for SMAC3. The setup and
workflow details are in [`OPENTUNER.md`](OPENTUNER.md) and
[`smac3/README.md`](smac3/README.md). `run_opentuner.py` remains only as a
compatibility wrapper around the OpenTuner CLI.

## Real-application comparison commands

The 30-candidate real-application comparison used the existing driver
`experiments/real_apps_20260917/expansion_20260919/run_search.py`. It registers
the frozen application binding, invokes the existing CONCORD, OpenTuner, or
SMAC3 pipeline, and writes each method's complete trace under the driver's
`runs/` directory. For the Movie Recommendation Training result, the original
comparison was launched by:

```bash
cd experiments/real_apps_20260917/expansion_20260919
PY=/home/yyan/miniconda3/envs/agent/bin/python
$PY run_recommendation_comparison.py
```

The lower-level `run_search.py` commands below are the method-specific form of
that script. They require the frozen application workspace, including its
`inputs/<application>_space.json`, whitelist, worker, contracts, and prepared
datasets. Those files are intentionally outside the clean four-directory
package and are ignored by the repository. The current checkout retains the
complete `recommendation20m` binding; commands for the other historical
bindings are shown for reference and need their original frozen inputs restored
before they can be rerun.

Run from the driver directory with the application-specific timeout shown below:

```bash
cd experiments/real_apps_20260917/expansion_20260919
PY=/home/yyan/miniconda3/envs/agent/bin/python

# Forest-Cover Model Selection (application=dask, timeout=180 s)
$PY run_search.py concord   --application dask             --label forest-concord-b30 --budget 30 --timeout 180
$PY run_search.py smac3     --application dask             --label forest-smac3-b30   --budget 30 --timeout 180
$PY run_search.py opentuner --application dask             --label forest-opentuner-b30 --budget 30 --timeout 180

# Movie Recommendation Training (application=recommendation20m, timeout=300 s)
$PY run_search.py concord   --application recommendation20m --label movie-concord-b30 --budget 30 --timeout 300
$PY run_search.py smac3     --application recommendation20m --label movie-smac3-b30   --budget 30 --timeout 300
$PY run_search.py opentuner --application recommendation20m --label movie-opentuner-b30 --budget 30 --timeout 300

# Scientific Claim Retrieval (application=sentence_mp, timeout=300 s)
$PY run_search.py concord   --application sentence_mp --label retrieval-concord-b30 --budget 30 --timeout 300
$PY run_search.py smac3     --application sentence_mp --label retrieval-smac3-b30   --budget 30 --timeout 300
$PY run_search.py opentuner --application sentence_mp --label retrieval-opentuner-b30 --budget 30 --timeout 300

# SIFT Visual Codebook Construction (application=spark_sift, timeout=1800 s)
$PY run_search.py concord   --application spark_sift --label sift-concord-b30 --budget 30 --timeout 1800
$PY run_search.py smac3     --application spark_sift --label sift-smac3-b30   --budget 30 --timeout 1800
$PY run_search.py opentuner --application spark_sift --label sift-opentuner-b30 --budget 30 --timeout 1800
```

Here `--budget 30` means 30 candidate executions; the driver adds the baseline
for SMAC3 and keeps the same baseline-excluded accounting for the other methods.
All trials are serial. The driver uses `smac==2.4.0` and `opentuner==0.8.8`;
their setup and standalone CLI details are documented in [`smac3/README.md`](smac3/README.md)
and [`OPENTUNER.md`](OPENTUNER.md). These comparison commands require the
frozen datasets, application environments, contracts, and model files prepared
by the original experiment workspace.
