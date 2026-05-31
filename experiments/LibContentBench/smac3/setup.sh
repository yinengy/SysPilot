#!/bin/sh
# Reproducible 0->1 setup for SMAC3 tuning of LibContentBench workflows.
#
# This script:
#   1. creates an isolated SMAC3 driver venv at smac3/.venv and installs SMAC3;
#   2. runs prepare.sh for each target workflow so its artifact/venv exists;
#   3. prints a readiness summary.
#
# Usage:
#   sh setup.sh                      # driver venv + prepare all 4 workflows
#   sh setup.sh nestedcpu-blas       # driver venv + prepare only listed workflows
#   SKIP_DRIVER=1 sh setup.sh        # skip the SMAC3 driver venv
#   SKIP_PREPARE=1 sh setup.sh       # skip workflow prepare.sh (driver only)
#
# Notes:
#   - torchdata-fsspec-s3-ddp additionally needs Docker at *run* time (local MinIO).
#   - A failed workflow prepare does not abort the rest; failures are summarized.
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
SUITE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DRIVER_VENV="$SCRIPT_DIR/.venv"

ALL_WORKFLOWS="nestedcpu-blas arrow-dask-duckdb-memspill hf-datasets-tokenizers-dataloader torchdata-fsspec-s3-ddp dali-nvjpeg-pytorch-cudnn"
if [ "$#" -gt 0 ]; then
  WORKFLOWS="$*"
else
  WORKFLOWS="$ALL_WORKFLOWS"
fi

PYTHON="${PYTHON:-python3}"
FAILURES=""

echo "=========================================================="
echo " SMAC3 LibContentBench setup"
echo "   suite root : $SUITE_ROOT"
echo "   driver venv: $DRIVER_VENV"
echo "   workflows  : $WORKFLOWS"
echo "=========================================================="

# ---- 1. SMAC3 driver venv -------------------------------------------------
if [ "${SKIP_DRIVER:-0}" = "1" ]; then
  echo "[driver] skipped (SKIP_DRIVER=1)"
else
  echo "[driver] creating venv at $DRIVER_VENV"
  "$PYTHON" -m venv "$DRIVER_VENV"
  "$DRIVER_VENV/bin/python" -m pip install --upgrade pip wheel >/dev/null
  echo "[driver] installing SMAC3 (this can take a few minutes)"
  if "$DRIVER_VENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"; then
    echo "[driver] SMAC3 install OK"
  else
    echo "[driver] SMAC3 install FAILED."
    echo "[driver] If pyrfr failed to build, install SWIG and retry:"
    echo "[driver]     apt-get install -y swig   # or: $DRIVER_VENV/bin/python -m pip install swig"
    FAILURES="$FAILURES driver"
  fi
fi

# ---- 2. workflow prepare.sh (each in its own workspace dir) ---------------
if [ "${SKIP_PREPARE:-0}" = "1" ]; then
  echo "[prepare] skipped (SKIP_PREPARE=1)"
else
  for wf in $WORKFLOWS; do
    wf_dir="$SUITE_ROOT/$wf"
    if [ ! -f "$wf_dir/prepare.sh" ]; then
      echo "[prepare:$wf] MISSING prepare.sh at $wf_dir"
      FAILURES="$FAILURES $wf"
      continue
    fi
    echo "[prepare:$wf] running prepare.sh ..."
    # prepare.sh writes relative paths (env/, work/), so run it from its dir.
    if ( cd "$wf_dir" && sh prepare.sh ); then
      echo "[prepare:$wf] OK"
    else
      echo "[prepare:$wf] FAILED"
      FAILURES="$FAILURES $wf"
    fi
  done
fi

# ---- 3. readiness summary -------------------------------------------------
echo "=========================================================="
echo " Readiness summary"
echo "=========================================================="
if [ -x "$DRIVER_VENV/bin/python" ]; then
  SMAC_VER=$("$DRIVER_VENV/bin/python" -c "import smac; from smac import HyperparameterOptimizationFacade, Scenario; import importlib.metadata as m; print(m.version('smac'))" 2>/dev/null)
  if [ -n "$SMAC_VER" ]; then
    echo "  driver: smac $SMAC_VER  ($DRIVER_VENV/bin/python)"
  else
    echo "  driver: SMAC3 NOT importable"
  fi
else
  echo "  driver: no venv"
fi
for wf in $WORKFLOWS; do
  if [ -x "$SUITE_ROOT/$wf/artifact/venv/bin/python" ]; then
    echo "  $wf: venv READY"
  else
    echo "  $wf: venv MISSING"
  fi
done

if echo "$WORKFLOWS" | grep -q torchdata; then
  if command -v docker >/dev/null 2>&1; then
    echo "  docker: $(docker --version 2>/dev/null)  (needed by torchdata-fsspec-s3-ddp)"
  else
    echo "  docker: MISSING  (torchdata-fsspec-s3-ddp will fail at run time)"
  fi
fi

if echo "$WORKFLOWS" | grep -q dali; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)  (dali-nvjpeg-pytorch-cudnn also needs NVIDIA DALI + CUDA MPS)"
  else
    echo "  gpu: MISSING  (dali-nvjpeg-pytorch-cudnn will fail at run time; needs CUDA + DALI + MPS)"
  fi
fi

echo "=========================================================="
if [ -n "$FAILURES" ]; then
  echo " Completed WITH FAILURES:$FAILURES"
  echo " Re-run setup for the failed targets after addressing the cause."
  exit 1
fi
echo " Setup complete. Next:"
echo "   $DRIVER_VENV/bin/python $SCRIPT_DIR/cli.py --list-workflows"
echo "   $DRIVER_VENV/bin/python $SCRIPT_DIR/cli.py --workflow nestedcpu-blas --size small --dry-run"
echo "=========================================================="
