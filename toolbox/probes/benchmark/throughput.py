from __future__ import annotations

import re

from toolbox.shared.probe_payload import combined_output, emit_result, excerpt_lines, load_payload, numeric_from_text


_THROUGHPUT_VALUE_PATTERNS = [
    re.compile(r"throughput[^0-9+-]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(samples/s|ops/s|items/s|it/s|steps/s)", re.IGNORECASE),
]


def _extract_throughput_value(line: str) -> float | int | None:
    for pattern in _THROUGHPUT_VALUE_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        raw = match.group(1)
        return float(raw) if "." in raw else int(raw)
    return numeric_from_text(line)


def collect(payload: dict[str, object]) -> dict[str, object]:
    output = combined_output(payload)
    excerpt = excerpt_lines(
        output,
        ["throughput", "samples/s", "ops/s", "items/s", "it/s", "steps/s"],
    )
    value = _extract_throughput_value(excerpt[0]) if excerpt else None
    return {
        "observed": bool(excerpt),
        "source": "runtime_output" if excerpt else "runtime_output_missing",
        "value": value if value is not None else (excerpt[:8] if excerpt else None),
        "excerpt": excerpt[:8],
        "flat": {},
        "structured": {"matches": excerpt[:8]} if excerpt else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
