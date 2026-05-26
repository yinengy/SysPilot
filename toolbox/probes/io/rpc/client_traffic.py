from __future__ import annotations

from typing import Any

from toolbox.shared.probe_payload import emit_result, load_payload


def collect(payload: dict[str, Any]) -> dict[str, object]:
    runtime_output = payload.get("runtime_output", {})
    if not isinstance(runtime_output, dict):
        runtime_output = {}
    network_delta = runtime_output.get("network_delta", {})
    if not isinstance(network_delta, dict):
        network_delta = {}

    total_bytes = 0.0
    for value in network_delta.values():
        if not isinstance(value, dict):
            continue
        sent = value.get("bytes_sent")
        recv = value.get("bytes_recv")
        if isinstance(sent, (int, float)) and isinstance(recv, (int, float)):
            total_bytes += float(sent) + float(recv)

    if total_bytes > 0:
        return {
            "observed": True,
            "source": "runtime_output.network_delta",
            "value": total_bytes,
            "flat": {"io.rpc.client_traffic": total_bytes},
            "structured": {
                "client_traffic_bytes": total_bytes,
                "approximation": "aggregate_single_host_network_proxy",
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
