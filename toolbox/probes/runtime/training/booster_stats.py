from __future__ import annotations

import re

from toolbox.shared.probe_payload import combined_output, emit_result, excerpt_lines, load_payload, numeric_from_text


_ROUND_PATTERNS = [
    re.compile(r"\[(\d+)\]\s*(train|eval|validation|test)", re.IGNORECASE),
    re.compile(r"(?:boost(?:ing|ed)?\s+rounds?|num_boosted_rounds|n_estimators)[^0-9+-]*([-+]?\d+)", re.IGNORECASE),
]


def _extract_round_stat(line: str) -> int | None:
    for pattern in _ROUND_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        value = match.group(1)
        if value.isdigit():
            return int(value)
    parsed = numeric_from_text(line)
    return int(parsed) if isinstance(parsed, int) else None


def collect(payload: dict[str, object]) -> dict[str, object]:
    output = combined_output(payload)
    excerpt = excerpt_lines(
        output,
        ["num_boosted_rounds", "boosted rounds", "boosting rounds", "n_estimators", "train", "validation"],
    )
    values = [value for value in (_extract_round_stat(line) for line in excerpt) if value is not None]
    round_count = max(values) if values else None
    return {
        "observed": bool(excerpt),
        "source": "runtime_output" if excerpt else "runtime_output_missing",
        "value": round_count if round_count is not None else (excerpt[:8] if excerpt else None),
        "excerpt": excerpt[:8],
        "flat": {},
        "structured": {"round_count": round_count, "matches": excerpt[:8]} if excerpt else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
