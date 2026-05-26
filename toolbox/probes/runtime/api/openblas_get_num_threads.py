from __future__ import annotations

from typing import Any

from toolbox.probes.runtime.api._openblas_common import openblas_entries, openblas_excerpt
from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    entries = openblas_entries(payload)
    thread_counts = [entry.get("num_threads") for entry in entries if isinstance(entry.get("num_threads"), int)]
    value = max(thread_counts) if thread_counts else None
    excerpt = [openblas_excerpt(entries[0])] if entries else []
    return {
        "observed": value is not None,
        "source": "runtime_output" if value is not None else "runtime_output_missing",
        "value": value,
        "excerpt": excerpt,
        "flat": {"runtime.api.openblas_get_num_threads": value} if value is not None else {},
        "structured": {"openblas_num_threads": value, "threadpools": entries[:4]} if value is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
