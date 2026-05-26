from __future__ import annotations

import json
import re
import sys
import csv
from pathlib import Path
from typing import Any


def load_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def emit_result(result: dict[str, Any]) -> None:
    print(json.dumps(result))


def combined_output(payload: dict[str, Any]) -> str:
    stdout = str(payload.get("stdout", "") or "")
    stderr = str(payload.get("stderr", "") or "")
    return stdout + (("\n" + stderr) if stderr else "")


def json_objects_from_output(payload: dict[str, Any]) -> list[dict[str, Any]]:
    text = combined_output(payload).strip()
    if not text:
        return []
    candidates = [text]
    candidates.extend(line.strip() for line in text.splitlines() if line.strip())
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def json_rows_from_output(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in json_objects_from_output(payload):
        raw_rows = item.get("rows")
        if isinstance(raw_rows, list):
            rows.extend(row for row in raw_rows if isinstance(row, dict))
            continue
        rows.append(item)
    return rows


def libcontentbench_metric_timeseries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_row in json_rows_from_output(payload):
        output_dir = output_row.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir.strip():
            continue
        csv_path = Path(output_dir) / "metrics_timeseries.csv"
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows.extend(dict(row) for row in reader)
        except OSError:
            continue
    return rows


def excerpt_lines(text: str, terms: list[str], *, limit: int = 8) -> list[str]:
    lowered_terms = [term.lower() for term in terms if term]
    matches: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(term in lowered for term in lowered_terms):
            matches.append(stripped)
        if len(matches) >= limit:
            break
    return matches


def primary_collect_profile(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("collect_profile_data", {})
    return data if isinstance(data, dict) else {}


def per_pid_runtime_maps(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("per_pid_runtime_maps", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): value
        for key, value in raw.items()
        if isinstance(value, dict)
    }


def primary_runtime_map(payload: dict[str, Any]) -> dict[str, Any]:
    runtime_maps = per_pid_runtime_maps(payload)
    primary_pid = payload.get("primary_pid")
    if primary_pid is not None:
        primary = runtime_maps.get(str(primary_pid))
        if isinstance(primary, dict):
            return primary
    return next(iter(runtime_maps.values()), {}) if runtime_maps else {}


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def parse_cpu_list_count(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    total = 0
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.strip().isdigit() and end.strip().isdigit():
                total += int(end) - int(start) + 1
        elif part.isdigit():
            total += 1
    return total or None


def parse_rss_kb(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def path_size_bytes(path: Path) -> int | None:
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.exists():
            return None
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return None


def numeric_from_text(text: str) -> float | int | None:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    raw = match.group(1)
    return float(raw) if "." in raw else int(raw)
