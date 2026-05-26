from __future__ import annotations

from toolbox.probes.io.queue.inflight_requests import collect
from toolbox.shared.probe_payload import emit_result, load_payload


if __name__ == "__main__":
    emit_result(collect(load_payload()))
