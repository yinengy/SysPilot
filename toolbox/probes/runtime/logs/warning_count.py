from __future__ import annotations

import re

from toolbox.shared.probe_payload import combined_output, emit_result, load_payload


_WARNING_PATTERN = re.compile(r"\bwarning\b", re.IGNORECASE)


def collect(payload: dict[str, object]) -> dict[str, object]:
    output = combined_output(payload)
    matches = [line.strip() for line in output.splitlines() if _WARNING_PATTERN.search(line)]
    return {
        "observed": bool(matches),
        "source": "runtime_output" if matches else "runtime_output_missing",
        "value": len(matches),
        "excerpt": matches[:8],
        "flat": {},
        "structured": {"warning_count": len(matches), "matches": matches[:8]},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
