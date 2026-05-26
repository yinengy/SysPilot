from __future__ import annotations

from typing import Any

from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    if not isinstance(runtime_output, dict):
        runtime_output = {}
    config_snapshot = runtime_output.get("config_snapshot", {})
    if not isinstance(config_snapshot, dict):
        config_snapshot = {}

    for key, source in (
        ("inflight_request_count", "runtime_output.inflight_request_count"),
        ("tfdata_request_depth", "runtime_output.tfdata_request_depth"),
        ("tfdata_max_outstanding_requests", "runtime_output.tfdata_max_outstanding_requests"),
    ):
        value = runtime_output.get(key)
        if isinstance(value, (int, float)):
            numeric = float(value)
            return {
                "observed": True,
                "source": source,
                "value": numeric,
                "flat": {"io.queue.inflight_requests": numeric},
                "structured": {"inflight_requests": numeric},
            }

    value = config_snapshot.get("tfdata_max_outstanding_requests")
    if isinstance(value, (int, float)):
        numeric = float(value)
        return {
            "observed": True,
            "source": "runtime_output.config_snapshot.tfdata_max_outstanding_requests",
            "value": numeric,
            "flat": {"io.queue.inflight_requests": numeric},
            "structured": {
                "inflight_requests": numeric,
                "approximation": "configured_request_cap_proxy",
            },
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
