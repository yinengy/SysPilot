from __future__ import annotations

from toolbox.probes.runtime.training.evaluation_history import collect
from toolbox.shared.probe_payload import emit_result, load_payload


if __name__ == "__main__":
    emit_result(collect(load_payload()))
