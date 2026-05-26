from __future__ import annotations

from typing import Any

from toolbox.probes.runtime.api._openblas_common import openblas_processor_count
from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    value, source = openblas_processor_count(payload)
    return {
        "observed": value is not None,
        "source": source,
        "value": value,
        "excerpt": [],
        "flat": {"runtime.api.openblas_get_num_procs": value} if value is not None else {},
        "structured": {"openblas_num_procs": value} if value is not None else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
