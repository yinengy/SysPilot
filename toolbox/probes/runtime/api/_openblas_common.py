from __future__ import annotations

from typing import Any

from toolbox.probes.threadpoolctl.threadpool_info import collect as collect_threadpool_info
from toolbox.shared.probe_payload import (
    excerpt_lines,
    parse_cpu_list_count,
    primary_runtime_map,
)


def openblas_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = collect_threadpool_info(payload)
    structured = result.get("structured", {})
    if not isinstance(structured, dict):
        return []
    raw_entries = structured.get("threadpools", [])
    if not isinstance(raw_entries, list):
        return []

    entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = " ".join(
            str(entry.get(key, "") or "")
            for key in ("internal_api", "user_api", "prefix", "filepath")
        ).lower()
        if "openblas" in fingerprint or entry.get("user_api") == "blas":
            entries.append(entry)
    return entries


def openblas_excerpt(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("internal_api", "") or entry.get("user_api", "") or "openblas").strip(),
        f"threads={entry.get('num_threads')}" if entry.get("num_threads") is not None else "",
        str(entry.get("threading_layer", "")).strip(),
        str(entry.get("architecture", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def openblas_processor_count(payload: dict[str, Any]) -> tuple[int | None, str]:
    runtime_map = primary_runtime_map(payload)
    allowed_cpus = runtime_map.get("Cpus_allowed_list") if isinstance(runtime_map, dict) else None
    allowed_count = parse_cpu_list_count(allowed_cpus)
    if allowed_count is not None:
        return allowed_count, "proc_status.Cpus_allowed_list"

    runtime_output = payload.get("runtime_output", {})
    if isinstance(runtime_output, dict):
        threads_counted = runtime_output.get("threads_counted")
        if isinstance(threads_counted, int):
            return threads_counted, "runtime_output.threads_counted"
    return None, "os_missing"


def openblas_output_excerpt(payload: dict[str, Any]) -> list[str]:
    runtime_output = payload.get("runtime_output", {})
    if isinstance(runtime_output, dict):
        stdout_excerpt = runtime_output.get("stdout_excerpt")
        if isinstance(stdout_excerpt, str) and stdout_excerpt.strip():
            matches = excerpt_lines(stdout_excerpt, ["openblas", "blas", "thread"], limit=8)
            if matches:
                return matches
    return []
