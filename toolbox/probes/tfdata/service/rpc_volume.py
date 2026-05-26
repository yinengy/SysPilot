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
    loopback_bytes = 0.0
    physical_bytes = 0.0
    for name, value in network_delta.items():
        if not isinstance(value, dict):
            continue
        bytes_sent = value.get("bytes_sent")
        bytes_recv = value.get("bytes_recv")
        if not isinstance(bytes_sent, (int, float)) or not isinstance(bytes_recv, (int, float)):
            continue
        delta_bytes = float(bytes_sent) + float(bytes_recv)
        total_bytes += delta_bytes
        if str(name) == "lo":
            loopback_bytes += delta_bytes
        else:
            physical_bytes += delta_bytes

    if total_bytes > 0:
        return {
            "observed": True,
            "source": "runtime_output.network_delta",
            "value": total_bytes,
            "flat": {"tfdata.service.rpc_volume": total_bytes},
            "structured": {
                "rpc_volume_bytes": total_bytes,
                "loopback_bytes": loopback_bytes,
                "physical_bytes": physical_bytes,
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
