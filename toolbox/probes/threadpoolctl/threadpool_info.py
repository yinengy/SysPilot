from __future__ import annotations

import json
from json import JSONDecodeError, JSONDecoder
from typing import Any

from toolbox.shared.probe_payload import combined_output, emit_result, load_payload


def _looks_like_threadpool_entry(value: object) -> bool:
    return isinstance(value, dict) and any(
        key in value for key in ("num_threads", "internal_api", "user_api", "prefix", "filepath")
    )


def _coerce_threadpool_entries(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        nested = value.get("threadpools")
        if isinstance(nested, list):
            return [entry for entry in nested if _looks_like_threadpool_entry(entry)]
        return [value] if _looks_like_threadpool_entry(value) else []
    if isinstance(value, list):
        return [entry for entry in value if _looks_like_threadpool_entry(entry)]
    return []


def _extract_json_candidates(text: str) -> list[object]:
    decoder = JSONDecoder()
    candidates: list[object] = []
    stripped = text.strip()
    if not stripped:
        return candidates

    try:
        candidates.append(json.loads(stripped))
        return candidates
    except JSONDecodeError:
        pass

    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        candidates.append(parsed)
    return candidates


def _extract_threadpools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    texts: list[str] = []
    combined = combined_output(payload)
    if combined.strip():
        texts.append(combined)
    runtime_output = payload.get("runtime_output")
    if isinstance(runtime_output, dict):
        excerpt = runtime_output.get("stdout_excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            texts.append(excerpt)

    for text in texts:
        for candidate in _extract_json_candidates(text):
            entries = _coerce_threadpool_entries(candidate)
            if entries:
                return entries
    return []


def _thread_count(entry: dict[str, Any]) -> int | None:
    value = entry.get("num_threads")
    return value if isinstance(value, int) else None


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    entries = _extract_threadpools(payload)
    thread_counts = [count for count in (_thread_count(entry) for entry in entries) if count is not None]
    max_threads = max(thread_counts) if thread_counts else None
    total_threads = sum(thread_counts) if thread_counts else None
    excerpt = [
        " ".join(
            part
            for part in (
                str(entry.get("internal_api", "") or entry.get("user_api", "") or entry.get("prefix", "runtime")).strip(),
                f"threads={entry.get('num_threads')}" if entry.get("num_threads") is not None else "",
            )
            if part
        )
        for entry in entries[:8]
    ]
    flat = {
        "threadpoolctl.threadpool_count": len(entries),
        "threadpoolctl.max_num_threads": max_threads,
        "threadpoolctl.total_num_threads": total_threads,
        "native_threadpool.loaded_runtime_count": len(entries),
        "native_threadpool.num_threads_per_pool": max_threads,
        "native_threadpool.max_num_threads_per_pool": max_threads,
    }
    flat = {key: value for key, value in flat.items() if value is not None}
    return {
        "observed": bool(entries),
        "source": "runtime_output" if entries else "runtime_output_missing",
        "value": max_threads,
        "excerpt": excerpt,
        "flat": flat,
        "structured": {
            "threadpool_count": len(entries),
            "num_threads_per_pool": thread_counts,
            "threadpools": entries[:8],
        }
        if entries
        else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
