# LibContentBench workflow-size command catalog

These commands intentionally vary **workflow/input size** for selected LibContentBench
workspaces. They are for scale sweeps, capacity checks, and choosing a more meaningful
benchmark size before fixed-workload tuning.

They are **not** fixed-baseline tuning comparisons by themselves. For fair tuning claims,
pick one size tier first, keep its workload-shape flags fixed, and then vary only the
library/runtime controls documented by that benchmark.

Assumptions:

- Run commands from the repository root.
- Run each benchmark's `prepare.sh` first so `artifact/venv/bin/python` exists.
- Output files are written under each workspace's `work/size_sweeps/` directory.
- The library/runtime knobs are left at each runner's direct-run defaults unless explicitly
  shown. The fields varied here are workload-shape fields.

## Size tiers

| Benchmark | Small | Medium | Large |
|---|---:|---:|---:|
| `nestedcpu-blas` | `320 × 800 tasks × 6 reps` | `320 × 1600 tasks × 6 reps` | `320 × 3200 tasks × 6 reps` |
| `arrow-dask-duckdb-memspill` | `10 × 20k rows` | `12 × 100k rows` | `12 × 250k rows` |
| `hf-datasets-tokenizers-dataloader` | `4096 rows, 32 tokens/row` | `8192 rows, 48 tokens/row` | `16384 rows, 64 tokens/row` |
| `torchdata-fsspec-s3-ddp` | `2 ranks × 1024 samples/rank` | `2 ranks × 2048 samples/rank` | `2 ranks × 4096 samples/rank` |
| `dali-nvjpeg-pytorch-cudnn` | `1.5k JPEGs, 640-768px, crop224, 10s+25s` | `5k JPEGs, 768-1024px, crop224, 10s+45s` | `10k JPEGs, 1024-1280px, crop224, 15s+75s` |

## `nestedcpu-blas`

Workload-size flags: `--matrix-size`, `--task-count`, `--repetitions`.
The `--joblib-n-jobs` and BLAS thread env vars are runtime/concurrency knobs; these commands
keep the current direct baseline concurrency shape (`6` outer jobs, `6` OpenMP/OpenBLAS
threads, and single-threaded MKL/BLIS/NumExpr) so only the task volume changes across tiers.

### Small

```bash
mkdir -p experiments/LibContentBench/nestedcpu-blas/work/size_sweeps && \
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  experiments/LibContentBench/nestedcpu-blas/artifact/venv/bin/python \
  experiments/LibContentBench/nestedcpu-blas/run_nestedcpu_blas_case.py \
  --case-id size_small --label 'size small' --mode size_sweep \
  --output-json experiments/LibContentBench/nestedcpu-blas/work/size_sweeps/small.json \
  --matrix-size 320 --task-count 800 --repetitions 6 \
  --joblib-n-jobs 6 --backend-prefer processes
```

### Medium

```bash
mkdir -p experiments/LibContentBench/nestedcpu-blas/work/size_sweeps && \
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  experiments/LibContentBench/nestedcpu-blas/artifact/venv/bin/python \
  experiments/LibContentBench/nestedcpu-blas/run_nestedcpu_blas_case.py \
  --case-id size_medium --label 'size medium' --mode size_sweep \
  --output-json experiments/LibContentBench/nestedcpu-blas/work/size_sweeps/medium.json \
  --matrix-size 320 --task-count 1600 --repetitions 6 \
  --joblib-n-jobs 6 --backend-prefer processes
```

### Large

```bash
mkdir -p experiments/LibContentBench/nestedcpu-blas/work/size_sweeps && \
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  experiments/LibContentBench/nestedcpu-blas/artifact/venv/bin/python \
  experiments/LibContentBench/nestedcpu-blas/run_nestedcpu_blas_case.py \
  --case-id size_large --label 'size large' --mode size_sweep \
  --output-json experiments/LibContentBench/nestedcpu-blas/work/size_sweeps/large.json \
  --matrix-size 320 --task-count 3200 --repetitions 6 \
  --joblib-n-jobs 6 --backend-prefer processes
```

## `arrow-dask-duckdb-memspill`

Workload-size flags: `--num-files`, `--rows-per-file`, `--row-group-size`.
The commands keep the direct small-run runtime controls fixed (`24` Dask workers,
`24` threads/worker, single-threaded Arrow/DuckDB) while scaling the Parquet input volume.
The medium and large tiers stay below the earlier exploratory 9M/12M-row stress cases, which
were useful for stress testing but too large to use as a generic local default.

### Small

```bash
mkdir -p experiments/LibContentBench/arrow-dask-duckdb-memspill/work/size_sweeps && \
env BENCHMARK_N_WORKERS=24 BENCHMARK_THREADS_PER_WORKER=24 \
  BENCHMARK_WORKER_MEMORY_LIMIT=256MB BENCHMARK_ARROW_CPU_COUNT=1 \
  BENCHMARK_ARROW_IO_THREADS=1 BENCHMARK_DUCKDB_MEMORY_LIMIT=1GB \
  BENCHMARK_DUCKDB_THREADS=1 BENCHMARK_PATH_LAYOUT=shared \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/artifact/venv/bin/python \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/run_arrow_dask_duckdb_case.py \
  --case-id size_small --label 'size small' --mode size_sweep \
  --output-json experiments/LibContentBench/arrow-dask-duckdb-memspill/work/size_sweeps/small.json \
  --runtime-root /var/tmp/libcontentbench/arrow-dask-duckdb-memspill-size-small \
  --num-files 10 --rows-per-file 20000 --row-group-size 5000
```

### Medium

```bash
mkdir -p experiments/LibContentBench/arrow-dask-duckdb-memspill/work/size_sweeps && \
env BENCHMARK_N_WORKERS=24 BENCHMARK_THREADS_PER_WORKER=24 \
  BENCHMARK_WORKER_MEMORY_LIMIT=256MB BENCHMARK_ARROW_CPU_COUNT=1 \
  BENCHMARK_ARROW_IO_THREADS=1 BENCHMARK_DUCKDB_MEMORY_LIMIT=1GB \
  BENCHMARK_DUCKDB_THREADS=1 BENCHMARK_PATH_LAYOUT=shared \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/artifact/venv/bin/python \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/run_arrow_dask_duckdb_case.py \
  --case-id size_medium --label 'size medium' --mode size_sweep \
  --output-json experiments/LibContentBench/arrow-dask-duckdb-memspill/work/size_sweeps/medium.json \
  --runtime-root /var/tmp/libcontentbench/arrow-dask-duckdb-memspill-size-medium \
  --num-files 12 --rows-per-file 100000 --row-group-size 10000
```

### Large

```bash
mkdir -p experiments/LibContentBench/arrow-dask-duckdb-memspill/work/size_sweeps && \
env BENCHMARK_N_WORKERS=24 BENCHMARK_THREADS_PER_WORKER=24 \
  BENCHMARK_WORKER_MEMORY_LIMIT=256MB BENCHMARK_ARROW_CPU_COUNT=1 \
  BENCHMARK_ARROW_IO_THREADS=1 BENCHMARK_DUCKDB_MEMORY_LIMIT=1GB \
  BENCHMARK_DUCKDB_THREADS=1 BENCHMARK_PATH_LAYOUT=shared \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/artifact/venv/bin/python \
  experiments/LibContentBench/arrow-dask-duckdb-memspill/run_arrow_dask_duckdb_case.py \
  --case-id size_large --label 'size large' --mode size_sweep \
  --output-json experiments/LibContentBench/arrow-dask-duckdb-memspill/work/size_sweeps/large.json \
  --runtime-root /var/tmp/libcontentbench/arrow-dask-duckdb-memspill-size-large \
  --num-files 12 --rows-per-file 250000 --row-group-size 25000
```

## `hf-datasets-tokenizers-dataloader`

Workload-size flags: `--dataset-rows`, `--tokens-per-row`, `--max-length`,
`--tokenizer-vocab-size`, `--hidden-dim`, `--num-classes`, `--batch-size`,
`--epochs`, and `--tokenize-batch-size`. These commands keep the direct small-run
runtime controls fixed (`24` Torch threads, `24` dataset processes, and `24` DataLoader workers)
while scaling corpus/token/model size.

### Small

```bash
mkdir -p experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/size_sweeps && \
env BENCHMARK_TORCH_NUM_THREADS=24 BENCHMARK_DATASETS_NUM_PROC=24 \
  BENCHMARK_TOKENIZERS_PARALLELISM=false BENCHMARK_NUM_WORKERS=24 \
  BENCHMARK_PREFETCH_FACTOR=1 BENCHMARK_PERSISTENT_WORKERS=true \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/artifact/venv/bin/python \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/run_hf_datasets_tokenizers_case.py \
  --case-id size_small --label 'size small' --mode size_sweep \
  --output-json experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/size_sweeps/small.json \
  --runtime-root /var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader-size-small \
  --dataset-rows 4096 --tokens-per-row 32 --max-length 64 --tokenizer-vocab-size 1024 \
  --hidden-dim 128 --num-classes 32 --batch-size 32 --epochs 2 --learning-rate 0.001 \
  --tokenize-batch-size 128 --timeout-seconds 600
```

### Medium

```bash
mkdir -p experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/size_sweeps && \
env BENCHMARK_TORCH_NUM_THREADS=24 BENCHMARK_DATASETS_NUM_PROC=24 \
  BENCHMARK_TOKENIZERS_PARALLELISM=false BENCHMARK_NUM_WORKERS=24 \
  BENCHMARK_PREFETCH_FACTOR=1 BENCHMARK_PERSISTENT_WORKERS=true \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/artifact/venv/bin/python \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/run_hf_datasets_tokenizers_case.py \
  --case-id size_medium --label 'size medium' --mode size_sweep \
  --output-json experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/size_sweeps/medium.json \
  --runtime-root /var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader-size-medium \
  --dataset-rows 8192 --tokens-per-row 48 --max-length 96 --tokenizer-vocab-size 2048 \
  --hidden-dim 256 --num-classes 32 --batch-size 32 --epochs 2 --learning-rate 0.001 \
  --tokenize-batch-size 256 --timeout-seconds 600
```

### Large

```bash
mkdir -p experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/size_sweeps && \
env BENCHMARK_TORCH_NUM_THREADS=24 BENCHMARK_DATASETS_NUM_PROC=24 \
  BENCHMARK_TOKENIZERS_PARALLELISM=false BENCHMARK_NUM_WORKERS=24 \
  BENCHMARK_PREFETCH_FACTOR=1 BENCHMARK_PERSISTENT_WORKERS=true \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/artifact/venv/bin/python \
  experiments/LibContentBench/hf-datasets-tokenizers-dataloader/run_hf_datasets_tokenizers_case.py \
  --case-id size_large --label 'size large' --mode size_sweep \
  --output-json experiments/LibContentBench/hf-datasets-tokenizers-dataloader/work/size_sweeps/large.json \
  --runtime-root /var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader-size-large \
  --dataset-rows 16384 --tokens-per-row 64 --max-length 128 --tokenizer-vocab-size 4096 \
  --hidden-dim 512 --num-classes 64 --batch-size 32 --epochs 2 --learning-rate 0.001 \
  --tokenize-batch-size 256 --timeout-seconds 900
```

## `torchdata-fsspec-s3-ddp`

Workload-size flags: `--world-size`, `--samples-per-rank`, `--shards-per-rank`,
`--input-dim`, `--hidden-dim`, `--num-classes`, `--batch-size`, and `--epochs`.
The tiers keep `world-size=2` because this workspace is single-host-first and the runner uses
CPU DDP/Gloo with a local MinIO object store. The commands keep the direct small-run S3/DDP
runtime controls fixed while scaling samples, shards, and model/input dimensions.

### Small

```bash
mkdir -p experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/small && \
env BENCHMARK_TORCH_NUM_THREADS=24 BENCHMARK_S3_BLOCK_SIZE=65536 \
  BENCHMARK_S3_CACHE_TYPE=none BENCHMARK_S3_USE_LISTINGS_CACHE=true \
  BENCHMARK_DDP_BUCKET_CAP_MB=1 \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/artifact/venv/bin/python \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/run_torchdata_fsspec_s3_ddp_case.py \
  --case-id size_small --label 'size small' --mode size_sweep \
  --output-json experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/small.json \
  --case-output-dir experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/small \
  --runtime-root /var/tmp/libcontentbench/torchdata-fsspec-s3-ddp-size-small \
  --world-size 2 --samples-per-rank 1024 --shards-per-rank 4 \
  --input-dim 256 --hidden-dim 512 --num-classes 32 --batch-size 32 --epochs 2 \
  --learning-rate 0.01 --timeout-seconds 420 --minio-image quay.io/minio/minio:latest
```

### Medium

```bash
mkdir -p experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/medium && \
env BENCHMARK_TORCH_NUM_THREADS=24 BENCHMARK_S3_BLOCK_SIZE=65536 \
  BENCHMARK_S3_CACHE_TYPE=none BENCHMARK_S3_USE_LISTINGS_CACHE=true \
  BENCHMARK_DDP_BUCKET_CAP_MB=1 \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/artifact/venv/bin/python \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/run_torchdata_fsspec_s3_ddp_case.py \
  --case-id size_medium --label 'size medium' --mode size_sweep \
  --output-json experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/medium.json \
  --case-output-dir experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/medium \
  --runtime-root /var/tmp/libcontentbench/torchdata-fsspec-s3-ddp-size-medium \
  --world-size 2 --samples-per-rank 2048 --shards-per-rank 8 \
  --input-dim 256 --hidden-dim 512 --num-classes 32 --batch-size 32 --epochs 2 \
  --learning-rate 0.01 --timeout-seconds 480 --minio-image quay.io/minio/minio:latest
```

### Large

```bash
mkdir -p experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/large && \
env BENCHMARK_TORCH_NUM_THREADS=24 BENCHMARK_S3_BLOCK_SIZE=65536 \
  BENCHMARK_S3_CACHE_TYPE=none BENCHMARK_S3_USE_LISTINGS_CACHE=true \
  BENCHMARK_DDP_BUCKET_CAP_MB=1 \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/artifact/venv/bin/python \
  experiments/LibContentBench/torchdata-fsspec-s3-ddp/run_torchdata_fsspec_s3_ddp_case.py \
  --case-id size_large --label 'size large' --mode size_sweep \
  --output-json experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/large.json \
  --case-output-dir experiments/LibContentBench/torchdata-fsspec-s3-ddp/work/size_sweeps/large \
  --runtime-root /var/tmp/libcontentbench/torchdata-fsspec-s3-ddp-size-large \
  --world-size 2 --samples-per-rank 4096 --shards-per-rank 8 \
  --input-dim 512 --hidden-dim 1024 --num-classes 32 --batch-size 32 --epochs 2 \
  --learning-rate 0.01 --timeout-seconds 600 --minio-image quay.io/minio/minio:latest
```

## `dali-nvjpeg-pytorch-cudnn`

Workload-size flags: `--dataset-images`, `--image-min-size`, `--image-max-size`,
`--warmup-seconds`, `--measure-seconds`, and `--timeout-seconds`. These commands
intentionally keep the current medium run's library/runtime controls fixed through
`BENCHMARK_*` env vars (`DALI b96 prefetch2 hw0.65 threads4 MPS50/50, Torch batch256`)
and keep class count plus CNN spatial input (`resize/crop=224`) stable, so the size tiers
primarily change synthetic JPEG volume/resolution and measurement budget.

Real runs require CUDA, NVIDIA DALI, PyTorch, and CUDA MPS. For host-only command
validation, append `--scaffold-only`; scaffold results are not throughput evidence.

### Small

```bash
mkdir -p experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/small && \
env BENCHMARK_DALI_BATCH_SIZE=96 BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH=2 \
  BENCHMARK_DALI_HW_DECODER_LOAD=0.65 BENCHMARK_DALI_NUM_THREADS=4 \
  BENCHMARK_DALI_MPS_SHARE=50 BENCHMARK_TORCH_MPS_SHARE=50 \
  BENCHMARK_TORCH_BATCH_SIZE=256 \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/artifact/venv/bin/python \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/run_dali_nvjpeg_pytorch_cudnn_case.py \
  --case-id size_small --label 'size small' --mode size_sweep \
  --output-json experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/small.json \
  --case-output-dir experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/small \
  --runtime-root /var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn-size-small \
  --timeout-seconds 300 --warmup-seconds 10 --measure-seconds 25 \
  --dataset-images 1500 --image-min-size 640 --image-max-size 768 \
  --jpeg-quality 90 --num-classes 16 --resize-size 224 --crop-size 224 \
  --dali-output-dtype float16 --dali-decode-backend mixed \
  --torch-model custom_cnn --torch-precision tf32
```

### Medium

```bash
mkdir -p experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/medium && \
env BENCHMARK_DALI_BATCH_SIZE=96 BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH=2 \
  BENCHMARK_DALI_HW_DECODER_LOAD=0.65 BENCHMARK_DALI_NUM_THREADS=4 \
  BENCHMARK_DALI_MPS_SHARE=50 BENCHMARK_TORCH_MPS_SHARE=50 \
  BENCHMARK_TORCH_BATCH_SIZE=256 \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/artifact/venv/bin/python \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/run_dali_nvjpeg_pytorch_cudnn_case.py \
  --case-id size_medium --label 'size medium' --mode size_sweep \
  --output-json experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/medium.json \
  --case-output-dir experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/medium \
  --runtime-root /var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn-size-medium \
  --timeout-seconds 600 --warmup-seconds 10 --measure-seconds 45 \
  --dataset-images 5000 --image-min-size 768 --image-max-size 1024 \
  --jpeg-quality 90 --num-classes 16 --resize-size 224 --crop-size 224 \
  --dali-output-dtype float16 --dali-decode-backend mixed \
  --torch-model custom_cnn --torch-precision tf32
```

### Large

```bash
mkdir -p experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/large && \
env BENCHMARK_DALI_BATCH_SIZE=96 BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH=2 \
  BENCHMARK_DALI_HW_DECODER_LOAD=0.65 BENCHMARK_DALI_NUM_THREADS=4 \
  BENCHMARK_DALI_MPS_SHARE=50 BENCHMARK_TORCH_MPS_SHARE=50 \
  BENCHMARK_TORCH_BATCH_SIZE=256 \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/artifact/venv/bin/python \
  experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/run_dali_nvjpeg_pytorch_cudnn_case.py \
  --case-id size_large --label 'size large' --mode size_sweep \
  --output-json experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/large.json \
  --case-output-dir experiments/LibContentBench/dali-nvjpeg-pytorch-cudnn/work/size_sweeps/large \
  --runtime-root /var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn-size-large \
  --timeout-seconds 900 --warmup-seconds 15 --measure-seconds 75 \
  --dataset-images 10000 --image-min-size 1024 --image-max-size 1280 \
  --jpeg-quality 90 --num-classes 16 --resize-size 224 --crop-size 224 \
  --dali-output-dtype float16 --dali-decode-backend mixed \
  --torch-model custom_cnn --torch-precision tf32
```
