from __future__ import annotations

from typing import Any

from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    if isinstance(runtime_output, dict):
        for key in ("batch_queue_wait_ms", "input_batch_queue_wait_ms", "queue_wait_ms"):
            value = runtime_output.get(key)
            if isinstance(value, (int, float)):
                numeric = float(value)
                return {
                    "observed": True,
                    "source": f"runtime_output.{key}",
                    "value": numeric,
                    "flat": {"runtime.pipeline.batch_queue_wait_time": numeric},
                    "structured": {"batch_queue_wait_time_ms": numeric},
                }
    return {
        "observed": False,
        "source": "runtime_output_missing",
        "value": None,
        "flat": {},
        "structured": {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
