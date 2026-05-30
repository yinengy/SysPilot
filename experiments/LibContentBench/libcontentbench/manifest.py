from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import load_json_object, require_list, require_object


@dataclass(frozen=True)
class BenchmarkEntry:
    benchmark_id: str
    relative_path: str
    optional: bool
    primary_metric: str
    python_path: str

    @property
    def workspace_name(self) -> str:
        return self.relative_path


def load_manifest(path: Path) -> list[BenchmarkEntry]:
    root = load_json_object(path)
    benchmarks = require_list(root.get("benchmarks"), context="benchmarks")
    entries: list[BenchmarkEntry] = []
    for index, raw in enumerate(benchmarks, start=1):
        item = require_object(raw, context=f"benchmark #{index}")
        entries.append(
            BenchmarkEntry(
                benchmark_id=str(item["benchmark_id"]),
                relative_path=str(item["relative_path"]),
                optional=bool(item.get("optional", False)),
                primary_metric=str(item.get("primary_metric", "throughput")),
                python_path=str(item.get("python_path", "artifact/venv/bin/python")),
            )
        )
    return entries
