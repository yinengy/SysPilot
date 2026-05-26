from __future__ import annotations

from typing import Any

from toolbox.probes.runtime.api._openblas_common import openblas_entries
from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    entries = openblas_entries(payload)
    for entry in entries:
        threading_layer = entry.get("threading_layer")
        if isinstance(threading_layer, str) and threading_layer.strip():
            return {
                "observed": True,
                "source": "runtime_output",
                "value": threading_layer.strip(),
                "excerpt": [threading_layer.strip()],
                "flat": {},
                "structured": {"openblas_parallel_model": threading_layer.strip(), "threadpools": entries[:4]},
            }

    return {
        "observed": False,
        "source": "runtime_output_missing",
        "value": None,
        "excerpt": [],
        "flat": {},
        "structured": {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
