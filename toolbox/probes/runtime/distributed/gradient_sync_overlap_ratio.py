from __future__ import annotations

from typing import Any

from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    if isinstance(runtime_output, dict):
        for key in ("gradient_sync_overlap_ratio", "ddp_gradient_sync_overlap_ratio"):
            value = runtime_output.get(key)
            if isinstance(value, (int, float)):
                numeric = float(value)
                return {
                    "observed": True,
                    "source": f"runtime_output.{key}",
                    "value": numeric,
                    "flat": {"runtime.distributed.gradient_sync_overlap_ratio": numeric},
                    "structured": {"gradient_sync_overlap_ratio": numeric},
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
