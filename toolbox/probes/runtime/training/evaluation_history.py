from __future__ import annotations

import re

from toolbox.shared.probe_payload import combined_output, emit_result, load_payload


_EVAL_PATTERN = re.compile(
    r"\[[0-9]+\].*(train|eval|validation|test)[^\n]*[:=]\s*[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)


def collect(payload: dict[str, object]) -> dict[str, object]:
    output = combined_output(payload)
    matches = [line.strip() for line in output.splitlines() if _EVAL_PATTERN.search(line)]
    return {
        "observed": bool(matches),
        "source": "runtime_output" if matches else "runtime_output_missing",
        "value": matches[:8] if matches else None,
        "excerpt": matches[:8],
        "flat": {},
        "structured": {"evaluation_lines": matches[:8], "evaluation_count": len(matches)} if matches else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
