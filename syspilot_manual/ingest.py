from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Iterable, Literal, cast

import yaml

from syspilot_manual.schemas import IngestedSource


DOC_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm"}
CODE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
}


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "source"


def _source_id_from_path(path: Path) -> str:
    without_suffix = path.with_suffix("").as_posix()
    return _slugify(without_suffix)


def _html_to_text(content: str) -> str:
    converted = content
    for level in range(1, 7):
        converted = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            lambda match: (
                "\n" + ("#" * level) + " " + re.sub(r"<[^>]+>", " ", match.group(1)) + "\n"
            ),
            converted,
            flags=re.IGNORECASE | re.DOTALL,
        )
    converted = re.sub(r"<script[\s\S]*?</script>", "", converted, flags=re.IGNORECASE)
    converted = re.sub(r"<style[\s\S]*?</style>", "", converted, flags=re.IGNORECASE)
    converted = re.sub(r"<[^>]+>", " ", converted)
    converted = unescape(converted)
    converted = re.sub(r"\r\n?", "\n", converted)
    converted = re.sub(r"[ \t]+", " ", converted)
    converted = re.sub(r"\n{3,}", "\n\n", converted)
    return converted.strip()


def _read_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        return _html_to_text(raw)
    return raw


def _title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name


def ingest_file(
    path: Path,
    *,
    kind: Literal["doc", "code_hint", "metadata"] = "doc",
) -> IngestedSource:
    return IngestedSource(
        source_id=_source_id_from_path(path),
        path=str(path),
        kind=kind,
        title=_title_from_path(path),
        content=_read_text(path),
    )


def discover_files(root: Path, *, suffixes: Iterable[str]) -> list[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    if root.is_file():
        return [root] if root.suffix.lower() in allowed else []
    if not root.exists():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed],
        key=lambda path: str(path),
    )


def ingest_paths(
    paths: list[Path],
    *,
    kind: Literal["doc", "code_hint", "metadata"] = "doc",
) -> list[IngestedSource]:
    normalized_kind = cast(Literal["doc", "code_hint", "metadata"], kind)
    return [
        ingest_file(path, kind=normalized_kind)
        for path in sorted(paths, key=lambda path: str(path))
    ]


def load_metadata_source(path: Path) -> IngestedSource:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"raw": path.read_text(encoding="utf-8")}
    formatted = json.dumps(payload, indent=2, sort_keys=True)
    return IngestedSource(
        source_id=_source_id_from_path(path),
        path=str(path),
        kind="metadata",
        title=_title_from_path(path),
        content=formatted,
    )
