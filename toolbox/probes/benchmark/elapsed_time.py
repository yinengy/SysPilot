from __future__ import annotations

from toolbox.shared.probe_payload import combined_output, emit_result, excerpt_lines, load_payload, numeric_from_text


def collect(payload: dict[str, object]) -> dict[str, object]:
    output = combined_output(payload)
    excerpt = excerpt_lines(output, ["elapsed", "elapsed_sec", "wall time", "wall_time", "seconds"])
    value = numeric_from_text(excerpt[0]) if excerpt else None
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
