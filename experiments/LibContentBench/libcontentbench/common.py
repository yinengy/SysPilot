from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object JSON at {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


PERFORMANCE_OUTPUT_KEYS = (
    "wall_time_sec",
    "elapsed_sec",
    "throughput",
    "samples_per_sec",
    "steps_per_sec",
    "rows_per_sec",
    "records_per_sec",
    "items_per_sec",
    "tasks_per_sec",
    "images_per_sec",
    "latency_p50",
    "latency_p90",
    "latency_p95",
    "latency_p99",
    "p50_latency_sec",
    "p90_latency_sec",
    "p95_latency_sec",
    "p99_latency_sec",
    "score",
    "solo_dali_images_per_sec",
    "solo_torch_images_per_sec",
    "pair_dali_images_per_sec",
    "pair_torch_images_per_sec",
    "dali_norm",
    "torch_norm",
)


def performance_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only end-to-end performance metrics for direct-run stdout."""
    return {
        key: payload[key]
        for key in PERFORMANCE_OUTPUT_KEYS
        if key in payload and payload[key] is not None
    }


def require_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {str(key): item for key, item in value.items()}


def require_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return list(value)


def normalize_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    values = require_list(value, context="tags")
    result: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or not raw:
            raise ValueError("tags entries must be non-empty strings")
        result.append(raw)
    return result


def load_matrix(path: Path) -> dict[str, Any]:
    root = load_json_object(path)
    defaults = require_object(root.get("defaults", {}), context="defaults")
    cases = require_list(root.get("cases"), context="cases")
    if not cases:
        raise ValueError("matrix must define a non-empty cases list")

    normalized_cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(cases, start=1):
        case_obj = require_object(raw_case, context=f"case #{index}")
        case_id = case_obj.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"case #{index} must define a non-empty case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        merged = dict(defaults)
        merged.update(case_obj)
        merged["tags"] = normalize_tags(merged.get("tags"))
        normalized_cases.append(merged)

    return {
        "experiment_name": str(root.get("experiment_name", path.parent.name.replace("-", "_"))),
        "defaults": defaults,
        "cases": normalized_cases,
    }


def filter_cases_by_tag(cases: list[dict[str, Any]], tag: str | None) -> list[dict[str, Any]]:
    return filter_cases(cases, tag=tag, case_id=None)


def filter_cases(cases: list[dict[str, Any]], *, tag: str | None, case_id: str | None) -> list[dict[str, Any]]:
    filtered = list(cases)
    if tag is not None:
        filtered = [case for case in filtered if tag in case.get("tags", [])]
    if case_id is not None:
        filtered = [case for case in filtered if str(case.get("case_id", "")) == case_id]
    return filtered


def default_run_label(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{sanitize_label(prefix)}_{timestamp}"


def sanitize_label(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    result = "".join(cleaned).strip("-")
    return result or "case"


def case_output_dir(run_dir: Path, index: int, case_id: str) -> Path:
    return run_dir / f"case_{index:02d}_{sanitize_label(case_id)}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_command(path: Path, command: list[str]) -> None:
    path.write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def current_epoch_seconds() -> float:
    return time.time()


def merge_env(overrides: dict[str, Any] | None) -> dict[str, str]:
    env = dict(os.environ)
    if overrides:
        env.update({str(key): str(value) for key, value in overrides.items()})
    return env


def normalize_executable(value: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or "/" in value:
        lexical_absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
        return os.path.abspath(str(lexical_absolute))
    return value
