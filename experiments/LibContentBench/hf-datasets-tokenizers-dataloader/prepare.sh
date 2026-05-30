#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/artifact/venv"
RUNTIME_ROOT="/var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader"

mkdir -p "$SCRIPT_DIR/env" "$SCRIPT_DIR/work" "$SCRIPT_DIR/runs" "$SCRIPT_DIR/artifact" "$RUNTIME_ROOT"

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel >/dev/null

PSUTIL_STATUS="missing"
NUMPY_STATUS="missing"
TORCH_STATUS="missing"
DATASETS_STATUS="missing"
TOKENIZERS_STATUS="missing"

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

if "$VENV_DIR/bin/python" -m pip install datasets >"$SCRIPT_DIR/artifact/install_datasets.log" 2>&1; then
  DATASETS_STATUS="installed"
else
  DATASETS_STATUS="pip_failed"
fi

if "$VENV_DIR/bin/python" -m pip install tokenizers >"$SCRIPT_DIR/artifact/install_tokenizers.log" 2>&1; then
  TOKENIZERS_STATUS="installed"
else
  TOKENIZERS_STATUS="pip_failed"
fi

PSUTIL_STATUS="$PSUTIL_STATUS" NUMPY_STATUS="$NUMPY_STATUS" TORCH_STATUS="$TORCH_STATUS" DATASETS_STATUS="$DATASETS_STATUS" TOKENIZERS_STATUS="$TOKENIZERS_STATUS" "$VENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


runtime_root = Path("/var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader")
root = Path("env")
root.mkdir(exist_ok=True)

torch_version = module_version("torch")
datasets_version = module_version("datasets")
tokenizers_version = module_version("tokenizers")
numpy_version = module_version("numpy")
psutil_version = module_version("psutil")

readiness = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "python_executable": os.sys.executable,
    "cpu_count": os.cpu_count(),
    "runtime_root": str(runtime_root),
    "torch_status": os.environ["TORCH_STATUS"],
    "datasets_status": os.environ["DATASETS_STATUS"],
    "tokenizers_status": os.environ["TOKENIZERS_STATUS"],
    "numpy_status": os.environ["NUMPY_STATUS"],
    "psutil_status": os.environ["PSUTIL_STATUS"],
    "torch_version": torch_version,
    "datasets_version": datasets_version,
    "tokenizers_version": tokenizers_version,
    "numpy_version": numpy_version,
    "psutil_version": psutil_version,
    "ready": (
        torch_version is not None
        and datasets_version is not None
        and tokenizers_version is not None
        and numpy_version is not None
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
            "tokenizer_root": str(runtime_root / "tokenizers"),
            "case_root": str(runtime_root / "cases"),
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
            "fairness_rule": "corpus rows, token length budget, tokenizer vocab target, batch size, and epoch count stay fixed across comparison modes",
            "runtime_root": str(runtime_root),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
