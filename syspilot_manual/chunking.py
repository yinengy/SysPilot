from __future__ import annotations

import re
from typing import Literal, cast

from syspilot_manual.schemas import CodeHint, IngestedSource, SourceChunk


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CODE_HINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("env_var", re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")),
    ("cli_flag", re.compile(r"(?<!\w)(--[a-z0-9][a-z0-9-]*|-{1}[A-Za-z])(?!\w)")),
    (
        "config_key",
        re.compile(r"(?:^|[\s\{\[\"'])(([a-z][a-z0-9_.-]{2,}))\s*(?:=|:)", re.MULTILINE),
    ),
    ("default", re.compile(r"\bdefault(?:s)?\s*(?:to|=|:)\s*([^\n.;]+)", re.IGNORECASE)),
    ("warning", re.compile(r"\b(?:warning|caution|note)\b[:\-]?\s*([^\n]+)", re.IGNORECASE)),
)


def extract_code_hints(text: str) -> list[CodeHint]:
    hints: list[CodeHint] = []
    seen: set[tuple[str, str]] = set()
    for hint_type, pattern in CODE_HINT_PATTERNS:
        for match in pattern.finditer(text):
            value = (match.group(1) or "").strip()
            if not value:
                continue
            key = (hint_type, value)
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                CodeHint(
                    hint_type=cast(
                        Literal["env_var", "cli_flag", "config_key", "default", "warning"],
                        hint_type,
                    ),
                    value=value,
                    snippet=match.group(0).strip(),
                )
            )
    return hints


def _split_large_text(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped] if stripped else []

    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]:
        addition = len(paragraph) + (2 if current else 0)
        if current and current_len + addition > max_chars:
            parts.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
            continue
        current.append(paragraph)
        current_len += addition

    if current:
        parts.append("\n\n".join(current))

    if not parts:
        return [stripped[i : i + max_chars].strip() for i in range(0, len(stripped), max_chars)]
    return parts


def chunk_source(source: IngestedSource, *, max_chars: int = 1600) -> list[SourceChunk]:
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    current_heading: list[str] = [source.title]
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((list(current_heading), text))

    for raw_line in source.content.splitlines():
        line = raw_line.rstrip()
        heading_match = HEADING_RE.match(line)
        if source.kind == "doc" and heading_match:
            flush()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            current_heading = list(heading_stack)
            current_lines = [line]
            continue
        if not current_lines and source.kind != "doc":
            current_heading = [source.title]
        current_lines.append(line)
    flush()

    if not sections:
        sections = [([source.title], source.content.strip())]

    chunks: list[SourceChunk] = []
    order = 0
    for heading_path, text in sections:
        for chunk_text in _split_large_text(text, max_chars=max_chars):
            chunks.append(
                SourceChunk(
                    source_id=source.source_id,
                    chunk_id=f"{source.source_id}:chunk_{order:03d}",
                    path=source.path,
                    kind=source.kind,
                    order=order,
                    heading_path=heading_path,
                    text=chunk_text,
                    code_hints=extract_code_hints(chunk_text) if source.kind == "code_hint" else [],
                )
            )
            order += 1
    return chunks
