#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/artifact/venv"
RUNTIME_ROOT="/var/tmp/libcontentbench/torchdata-fsspec-s3-ddp"

mkdir -p "$SCRIPT_DIR/env" "$SCRIPT_DIR/work" "$SCRIPT_DIR/runs" "$SCRIPT_DIR/artifact" "$RUNTIME_ROOT"

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel >/dev/null

PSUTIL_STATUS="missing"
NUMPY_STATUS="missing"
TORCH_STATUS="missing"
TORCHDATA_STATUS="missing"
S3FS_STATUS="missing"
DOCKER_STATUS="missing"

if "$VENV_DIR/bin/python" -m pip install psutil >"$SCRIPT_DIR/artifact/install_psutil.log" 2>&1; then
  PSUTIL_STATUS="installed"
else
  PSUTIL_STATUS="pip_failed"
fi

if "$VENV_DIR/bin/python" -m pip install numpy >"$SCRIPT_DIR/artifact/install_numpy.log" 2>&1; then
  NUMPY_STATUS="installed"
else
  NUMPY_STATUS="pip_failed"
fi

if "$VENV_DIR/bin/python" -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch >"$SCRIPT_DIR/artifact/install_torch.log" 2>&1; then
  TORCH_STATUS="installed"
elif "$VENV_DIR/bin/python" -m pip install --force-reinstall torch >>"$SCRIPT_DIR/artifact/install_torch.log" 2>&1; then
  TORCH_STATUS="installed_non_cpu_wheel"
else
  TORCH_STATUS="pip_failed"
fi

if "$VENV_DIR/bin/python" -m pip install torchdata >"$SCRIPT_DIR/artifact/install_torchdata.log" 2>&1; then
  TORCHDATA_STATUS="installed"
else
  TORCHDATA_STATUS="pip_failed"
fi

if "$VENV_DIR/bin/python" -m pip install s3fs fsspec >"$SCRIPT_DIR/artifact/install_s3fs.log" 2>&1; then
  S3FS_STATUS="installed"
else
  S3FS_STATUS="pip_failed"
fi

if command -v docker >/dev/null 2>&1; then
  DOCKER_STATUS="available"
else
  DOCKER_STATUS="missing"
fi

PSUTIL_STATUS="$PSUTIL_STATUS" \
NUMPY_STATUS="$NUMPY_STATUS" \
TORCH_STATUS="$TORCH_STATUS" \
TORCHDATA_STATUS="$TORCHDATA_STATUS" \
S3FS_STATUS="$S3FS_STATUS" \
DOCKER_STATUS="$DOCKER_STATUS" \
"$VENV_DIR/bin/python" - <<'PY'
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
    return getattr(module, "__version__", None)


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


runtime_root = Path("/var/tmp/libcontentbench/torchdata-fsspec-s3-ddp")
root = Path("env")
root.mkdir(exist_ok=True)

torch_version = module_version("torch")
torchdata_version = module_version("torchdata")
datapipe_iter_available = module_available("torch.utils.data.datapipes.iter")
numpy_version = module_version("numpy")
psutil_version = module_version("psutil")
s3fs_version = module_version("s3fs")
fsspec_version = module_version("fsspec")

docker_path = shutil.which("docker")

ready = (
    torch_version is not None
    and datapipe_iter_available
    and numpy_version is not None
    and s3fs_version is not None
    and fsspec_version is not None
    and docker_path is not None
)

readiness = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "python_executable": os.sys.executable,
    "cpu_count": os.cpu_count(),
    "runtime_root": str(runtime_root),
    "torch_status": os.environ["TORCH_STATUS"],
    "torchdata_status": os.environ["TORCHDATA_STATUS"],
    "s3fs_status": os.environ["S3FS_STATUS"],
    "numpy_status": os.environ["NUMPY_STATUS"],
    "psutil_status": os.environ["PSUTIL_STATUS"],
    "docker_status": os.environ["DOCKER_STATUS"],
    "docker_path": docker_path,
    "torch_version": torch_version,
    "torchdata_version": torchdata_version,
    "datapipe_iter_module": "torch.utils.data.datapipes.iter" if datapipe_iter_available else None,
    "numpy_version": numpy_version,
    "psutil_version": psutil_version,
    "s3fs_version": s3fs_version,
    "fsspec_version": fsspec_version,
    "ready": ready,
    "notes": (
        "The packaged benchmark requires Docker for the local MinIO object store and uses CPU/Gloo single-host DDP. "
        "Successful prepare does not guarantee that MinIO image pulls will succeed at runtime."
    ),
}

(root / "python_executable.txt").write_text(os.sys.executable + "\n", encoding="utf-8")
(root / "environment_summary.txt").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(root / "readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")

Path("work/runtime_plan.json").write_text(
    json.dumps(
        {
            "runtime_root": str(runtime_root),
            "dataset_root": str(runtime_root / "datasets"),
            "case_root": str(runtime_root / "cases"),
            "minio_note": "Each case starts a bounded local MinIO container and should not overclaim multi-host S3 behavior.",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
Path("work/workload_lock.json").write_text(
    json.dumps(
        {
            "synthetic_data": True,
            "fairness_rule": "samples per rank, shard layout, record format, batch size, and epoch count stay fixed across comparison modes",
            "runtime_root": str(runtime_root),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
