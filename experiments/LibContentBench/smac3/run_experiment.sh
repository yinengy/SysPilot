#!/bin/sh
# Single-command entry point for the full SMAC3 fair-comparison sweep.
#
# Runs every (workflow, size, budget) tuning job sequentially via the SMAC3
# driver venv, then aggregates a master_report.csv that is row-for-row
# comparable to SysPilot's fixed-budget results.
#
# Prereq: run ./setup.sh once first (creates smac3/.venv + workflow venvs).
#
# Usage:
#   sh run_experiment.sh                          # full matrix: 4 wf x 3 sizes x {10,20,30}
#   sh run_experiment.sh --sizes small --budgets 10        # quick subset
#   sh run_experiment.sh --workflows nestedcpu-blas,arrow-dask-duckdb-memspill
#   sh run_experiment.sh --dry-run                # print the plan only
#
# All extra args are passed straight through to experiment.py.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
DRIVER_VENV="$SCRIPT_DIR/.venv"
PYBIN="$DRIVER_VENV/bin/python"

if [ ! -x "$PYBIN" ]; then
  echo "error: SMAC3 driver venv not found at $DRIVER_VENV" >&2
  echo "       run 'sh $SCRIPT_DIR/setup.sh' first." >&2
  exit 1
fi

# PYTHONHASHSEED must be set before the interpreter starts for determinism;
# the orchestrator propagates it to every benchmark subprocess.
export PYTHONHASHSEED=0

exec "$PYBIN" "$SCRIPT_DIR/experiment.py" "$@"
