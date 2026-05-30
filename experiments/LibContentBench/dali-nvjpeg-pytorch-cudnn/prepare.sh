#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/artifact/venv"
RUNTIME_ROOT="/var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn"
DRIVER_LIB_DIR="$SCRIPT_DIR/artifact/driver_lib"

venv_python() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" "$@"
}

detect_cuda_family() {
  if [ "${BENCHMARK_CUDA_FAMILY:-auto}" != "auto" ]; then
    printf '%s\n' "$BENCHMARK_CUDA_FAMILY"
    return
  fi
  cuda_version=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_version=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ { for (i = 1; i < NF; i++) if ($i == "CUDA" && $(i + 1) == "Version:") { print $(i + 2); exit } }') || true
  fi
  cuda_major=${cuda_version%%.*}
  case "$cuda_major" in
    11) printf '%s\n' "cu117" ;;
    12) printf '%s\n' "cu120" ;;
    13) printf '%s\n' "cu130" ;;
    *) printf '%s\n' "cu120" ;;
  esac
}

CUDA_FAMILY=$(detect_cuda_family)
case "$CUDA_FAMILY" in
  cu117)
    NUMPY_PIP_ARGS=${BENCHMARK_NUMPY_PIP_ARGS:-"numpy<2"}
    TORCH_PIP_ARGS=${BENCHMARK_TORCH_PIP_ARGS:-"torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117"}
    DALI_PIP_ARGS=${BENCHMARK_DALI_PIP_ARGS:-"nvidia-dali-cuda110 --extra-index-url https://pypi.nvidia.com"}
    ;;
  cu120)
    NUMPY_PIP_ARGS=${BENCHMARK_NUMPY_PIP_ARGS:-"numpy"}
    TORCH_PIP_ARGS=${BENCHMARK_TORCH_PIP_ARGS:-"torch --index-url https://download.pytorch.org/whl/cu121"}
    DALI_PIP_ARGS=${BENCHMARK_DALI_PIP_ARGS:-"nvidia-dali-cuda120 --extra-index-url https://pypi.nvidia.com"}
    ;;
  cu130)
    NUMPY_PIP_ARGS=${BENCHMARK_NUMPY_PIP_ARGS:-"numpy"}
    TORCH_PIP_ARGS=${BENCHMARK_TORCH_PIP_ARGS:-"torch --index-url https://download.pytorch.org/whl/cu130"}
    DALI_PIP_ARGS=${BENCHMARK_DALI_PIP_ARGS:-"nvidia-dali-cuda130 --extra-index-url https://pypi.nvidia.com"}
    ;;
  *)
    echo "Unsupported BENCHMARK_CUDA_FAMILY=$CUDA_FAMILY; expected cu117, cu120, or cu130" >&2
    exit 2
    ;;
esac

mkdir -p "$SCRIPT_DIR/env" "$SCRIPT_DIR/work" "$SCRIPT_DIR/runs" "$SCRIPT_DIR/artifact" "$DRIVER_LIB_DIR" "$RUNTIME_ROOT"

LIBCUDA_TARGET=""
for candidate in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 /usr/lib/libcuda.so.1; do
  if [ -e "$candidate" ]; then
    LIBCUDA_TARGET="$candidate"
    break
  fi
done
if [ -n "$LIBCUDA_TARGET" ]; then
  ln -sf "$LIBCUDA_TARGET" "$DRIVER_LIB_DIR/libcuda.so"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
venv_python -m pip install --upgrade pip wheel >/dev/null

PILLOW_STATUS="missing"
NUMPY_STATUS="missing"
TORCH_STATUS="missing"
DALI_STATUS="missing"

if venv_python -m pip install pillow $NUMPY_PIP_ARGS >"$SCRIPT_DIR/artifact/install_core.log" 2>&1; then
  PILLOW_STATUS="installed"
  NUMPY_STATUS="installed"
else
  PILLOW_STATUS="pip_failed"
  NUMPY_STATUS="pip_failed"
fi

if venv_python -m pip install $TORCH_PIP_ARGS >"$SCRIPT_DIR/artifact/install_torch.log" 2>&1; then
  TORCH_STATUS="installed"
else
  TORCH_STATUS="pip_failed"
fi

venv_python -m pip uninstall -y nvidia-dali-cuda110 nvidia-dali-cuda120 nvidia-dali-cuda130 >"$SCRIPT_DIR/artifact/install_dali_cleanup.log" 2>&1 || true
if venv_python -m pip install $DALI_PIP_ARGS >"$SCRIPT_DIR/artifact/install_dali.log" 2>&1; then
  DALI_STATUS="installed"
else
  DALI_STATUS="pip_failed"
fi

PILLOW_STATUS="$PILLOW_STATUS" NUMPY_STATUS="$NUMPY_STATUS" TORCH_STATUS="$TORCH_STATUS" DALI_STATUS="$DALI_STATUS" CUDA_FAMILY="$CUDA_FAMILY" NUMPY_PIP_ARGS="$NUMPY_PIP_ARGS" TORCH_PIP_ARGS="$TORCH_PIP_ARGS" DALI_PIP_ARGS="$DALI_PIP_ARGS" DRIVER_LIB_DIR="$DRIVER_LIB_DIR" LIBCUDA_TARGET="$LIBCUDA_TARGET" venv_python - <<'PY'
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
from pathlib import Path


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


root = Path("env")
root.mkdir(exist_ok=True)
runtime_root = Path("/var/tmp/libcontentbench/dali-nvjpeg-pytorch-cudnn")
readiness = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "python_executable": os.sys.executable,
    "runtime_root": str(runtime_root),
    "cuda_family": os.environ["CUDA_FAMILY"],
    "numpy_pip_args": os.environ["NUMPY_PIP_ARGS"],
    "torch_pip_args": os.environ["TORCH_PIP_ARGS"],
    "dali_pip_args": os.environ["DALI_PIP_ARGS"],
    "driver_lib_dir": os.environ["DRIVER_LIB_DIR"],
    "libcuda_target": os.environ["LIBCUDA_TARGET"],
    "pillow_status": os.environ["PILLOW_STATUS"],
    "numpy_status": os.environ["NUMPY_STATUS"],
    "torch_status": os.environ["TORCH_STATUS"],
    "dali_status": os.environ["DALI_STATUS"],
    "pillow_version": module_version("PIL"),
    "numpy_version": module_version("numpy"),
    "torch_version": module_version("torch"),
    "dali_version": module_version("nvidia.dali"),
    "cuda_available": cuda_available(),
    "mps_control": shutil.which("nvidia-cuda-mps-control"),
}
readiness["ready"] = bool(readiness["torch_version"] and readiness["dali_version"] and readiness["cuda_available"] and readiness["mps_control"])
readiness["notes"] = "Real runs require CUDA, NVIDIA DALI, PyTorch, and nvidia-cuda-mps-control. Scaffold mode validates host-side plumbing without GPU."
(root / "python_executable.txt").write_text(os.sys.executable + "\n", encoding="utf-8")
(root / "readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(root / "environment_summary.txt").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path("work/runtime_plan.json").write_text(json.dumps({
    "runtime_root": str(runtime_root),
    "dataset_root": str(runtime_root / "dataset"),
    "mps_scope": "per-case CUDA MPS pipe/log directories",
    "scaffold_supported": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path("work/workload_lock.json").write_text(json.dumps({
    "fairness_rule": "dataset image count/size range, resize/crop, DALI output dtype, PyTorch model/input/precision stay fixed for fixed-baseline comparisons",
    "primary_score": "min(DALI_pair/DALI_solo, Torch_pair/Torch_solo)",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
