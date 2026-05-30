#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/artifact/venv"

mkdir -p "$SCRIPT_DIR/env" "$SCRIPT_DIR/work" "$SCRIPT_DIR/runs" "$SCRIPT_DIR/artifact"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel >/dev/null
"$VENV_DIR/bin/python" -m pip install numpy joblib threadpoolctl psutil >/dev/null

"$VENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from threadpoolctl import threadpool_info

root = Path("env")
root.mkdir(exist_ok=True)
(root / "python_executable.txt").write_text(os.sys.executable + "\n", encoding="utf-8")
(root / "environment_summary.txt").write_text(
    json.dumps(
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(root / "threadpool_info.json").write_text(
    json.dumps(threadpool_info(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
Path("work/workload_lock.json").write_text(
    json.dumps(
        {
            "random_data": True,
            "fairness_rule": "matrix_size, repetitions, and task_count remain fixed across comparison modes",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
