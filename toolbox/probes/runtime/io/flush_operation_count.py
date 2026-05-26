from __future__ import annotations

from typing import Any

from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    if isinstance(runtime_output, dict):
        for key in ("flush_operation_count", "writer_flush_count", "flush_count"):
            value = runtime_output.get(key)
            if isinstance(value, int):
                return {
                    "observed": True,
                    "source": f"runtime_output.{key}",
                    "value": value,
                    "flat": {"runtime.io.flush_operation_count": value},
                    "structured": {"flush_operation_count": value},
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
