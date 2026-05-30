#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/artifact/venv"
RUNTIME_ROOT="/var/tmp/libcontentbench/arrow-dask-duckdb-memspill"

mkdir -p "$SCRIPT_DIR/env" "$SCRIPT_DIR/work" "$SCRIPT_DIR/runs" "$SCRIPT_DIR/artifact" "$RUNTIME_ROOT"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel >/dev/null
"$VENV_DIR/bin/python" -m pip install numpy pyarrow duckdb "dask[distributed]" psutil >/dev/null

"$VENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

root = Path("env")
root.mkdir(exist_ok=True)
runtime_root = Path("/var/tmp/libcontentbench/arrow-dask-duckdb-memspill")
(root / "python_executable.txt").write_text(os.sys.executable + "\n", encoding="utf-8")
(root / "environment_summary.txt").write_text(
    json.dumps(
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "runtime_root": str(runtime_root),
            "cpu_count": os.cpu_count(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
Path("work/runtime_plan.json").write_text(
    json.dumps(
        {
            "runtime_root": str(runtime_root),
            "dataset_root": str(runtime_root / "datasets"),
            "shared_root": str(runtime_root / "shared"),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
