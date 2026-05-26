from __future__ import annotations

from toolbox.shared.probe_payload import combined_output, emit_result, excerpt_lines, json_rows_from_output, load_payload, numeric_from_text


def _completion_count_from_json_rows(payload: dict[str, object]) -> int | None:
    rows = json_rows_from_output(payload)
    if not rows:
        return None
    completed = 0
    saw_completion_field = False
    for row in rows:
        success = row.get("success")
        if isinstance(success, bool):
            saw_completion_field = True
            if success:
                completed += 1
            continue
        status = str(row.get("status", row.get("result_status", ""))).strip().lower()
        if status:
            saw_completion_field = True
            if status in {"ok", "success", "completed", "complete", "passed"}:
                completed += 1
    if saw_completion_field:
        return completed
    return len(rows)


def collect(payload: dict[str, object]) -> dict[str, object]:
    json_count = _completion_count_from_json_rows(payload)
    if json_count is not None:
        return {
            "observed": True,
            "source": "runtime_output.json.rows",
            "value": json_count,
            "excerpt": [],
            "flat": {"benchmark.completion_count": json_count},
            "structured": {"completion_count": json_count},
        }
    output = combined_output(payload)
    excerpt = excerpt_lines(output, ["completion", "completed", "totaltaskcompletes", "tasks complete", "workflowmetrics"])
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
