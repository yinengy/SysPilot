from __future__ import annotations

from typing import Any

from toolbox.probes.runtime.api._openblas_common import openblas_entries, openblas_output_excerpt
from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    entries = openblas_entries(payload)
    for entry in entries:
        architecture = entry.get("architecture")
        if isinstance(architecture, str) and architecture.strip():
            return {
                "observed": True,
                "source": "runtime_output",
                "value": architecture.strip(),
                "excerpt": [architecture.strip()],
                "flat": {},
                "structured": {"openblas_corename": architecture.strip(), "threadpools": entries[:4]},
            }

    excerpt = openblas_output_excerpt(payload)
    return {
        "observed": bool(excerpt),
        "source": "runtime_output" if excerpt else "runtime_output_missing",
        "value": excerpt[0] if excerpt else None,
        "excerpt": excerpt[:8],
        "flat": {},
        "structured": {"openblas_corename": excerpt[0], "matches": excerpt[:8]} if excerpt else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
