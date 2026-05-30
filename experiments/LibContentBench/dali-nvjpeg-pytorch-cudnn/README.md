# DALI-nvJPEG-PyTorch-cuDNN

This optional LibContentBench workspace packages a short-running single-GPU contention benchmark built from:

- NVIDIA DALI / nvJPEG-style GPU JPEG decode and preprocessing
- PyTorch/cuDNN convolution-heavy inference on synthetic tensors
- CUDA MPS active-thread partitioning for two independent GPU client processes

The benchmark uses synthetic JPEGs because the claim is about decode, GPU preprocessing, copy/memory pressure, and cuDNN contention rather than model accuracy. DALI writes no preprocessed outputs and PyTorch uses synthetic inputs only.

## Honest scope

- Real runs require CUDA, NVIDIA DALI, PyTorch, and `nvidia-cuda-mps-control` on the host.
- `--scaffold-only` validates CLI/env/report plumbing without importing GPU libraries or claiming performance.
- Fixed-baseline comparisons keep JPEG count/size range, resize/crop, output dtype, DALI decode backend, PyTorch model/spatial input shape, precision, and measurement windows fixed while exposing PyTorch batch size as a runtime tuning knob.
- The primary score is `min(DALI_pair/DALI_solo, Torch_pair/Torch_solo)`, so a run must protect both the preprocessing worker and the main cuDNN worker.

## Layout

- `prepare.sh` — creates the benchmark virtualenv, installs Python dependencies when possible, and records readiness metadata.
- `run_dali_nvjpeg_pytorch_cudnn_case.py` — direct benchmark runner and worker entrypoint.
- `build_report.py` — rebuilds `result_table.csv`, `comparison_report.json`, `acceptance_report.json`, and `summary.md`.
- `matrix.yaml` — bounded case catalog for baseline/application/library/oracle comparisons.
- `work/`, `runs/`, `env/`, `artifact/` — runtime metadata and archived evidence locations.

For runnable benchmark commands, use `../summary_repro.md` or the direct worker command surfaces in generated reports.

## Usage

```bash
bash prepare.sh
artifact/venv/bin/python run_dali_nvjpeg_pytorch_cudnn_case.py \
  --case-id baseline_dali_b64_prefetch2_hw90_mps20_80 \
  --label "baseline DALI b64 prefetch2 hw90 MPS20/80" \
  --mode baseline \
  --output-json work/direct_result.json \
  --case-output-dir work/direct_case_output \
  --runtime-root /var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn \
  --warmup-seconds 10 \
  --measure-seconds 45 \
  --dataset-images 5000 \
  --image-min-size 768 \
  --image-max-size 1024 \
  --jpeg-quality 90 \
  --num-classes 16 \
  --resize-size 256 \
  --crop-size 224 \
  --dali-output-dtype float16 \
  --dali-decode-backend mixed \
  --torch-model custom_cnn \
  --torch-precision tf32
```

For host-only validation, append `--scaffold-only`.

`prepare.sh` isolates the benchmark venv from any outer `PYTHONPATH` and chooses CUDA-family Python wheels from `nvidia-smi` by default:

- CUDA 11.x hosts: PyTorch `cu117` plus `nvidia-dali-cuda110`.
- CUDA 12.x hosts: PyTorch `cu121` plus `nvidia-dali-cuda120`.
- CUDA 13.x hosts: PyTorch `cu130` plus `nvidia-dali-cuda130`.

On CUDA 11.x hosts, setup pins `numpy<2` for PyTorch 2.0 compatibility and creates `artifact/driver_lib/libcuda.so` pointing at the system `libcuda.so.1` when needed. Worker subprocesses prepend that directory to `LD_LIBRARY_PATH` so cuDNN/DALI libraries can resolve the driver stub consistently.

CUDA MPS pipe/log directories use short hashed paths under `/tmp/lcb-dali-mps/` because MPS control sockets are Unix-domain sockets and long benchmark case IDs can exceed the socket path limit.

Override this only when the host driver/toolkit requires a different pairing:

```bash
BENCHMARK_CUDA_FAMILY=cu117 bash prepare.sh
BENCHMARK_TORCH_PIP_ARGS="torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117" \
BENCHMARK_DALI_PIP_ARGS="nvidia-dali-cuda110 --extra-index-url https://pypi.nvidia.com" \
  bash prepare.sh
```

## Fixed-baseline SysPilot tuning surface

The runner reads these optional environment overrides for library/runtime tuning:

- `BENCHMARK_DALI_BATCH_SIZE`
- `BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH`
- `BENCHMARK_DALI_HW_DECODER_LOAD`
- `BENCHMARK_DALI_NUM_THREADS`
- `BENCHMARK_DALI_MPS_SHARE`
- `BENCHMARK_TORCH_BATCH_SIZE`
- `BENCHMARK_TORCH_MPS_SHARE`

These map to DALI decode/preprocessing pressure, PyTorch/cuDNN batch pressure, and CUDA MPS partitioning. Workload-shape fields such as JPEG count, JPEG size range, resize/crop size, output dtype, DALI decode backend, PyTorch model, precision, and synthetic spatial input shape stay fixed unless intentionally changed through direct CLI flags.
