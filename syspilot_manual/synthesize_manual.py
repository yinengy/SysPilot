from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Any

from syspilot_manual.llm_client import ManualLLMProtocol, extract_json_object
from syspilot_manual.schemas import EvidenceItem, LibraryManual, SourceArtifact


def _trim_exemplar(content: str, max_chars: int = 4000) -> str:
    stripped = content.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip()


def _default_style_exemplars() -> dict[str, str]:
    manual_dir = Path(__file__).resolve().parents[1] / "manual"
    exemplars: dict[str, str] = {}
    for name in ("numactl", "pytorch"):
        path = manual_dir / f"{name}.md"
        if path.exists():
            exemplars[name] = _trim_exemplar(path.read_text(encoding="utf-8"))
    return exemplars


def _synthesis_evidence_payload(evidence_items: list[EvidenceItem]) -> list[dict[str, object]]:
    per_kind_limits = {
        "feature": 8,
        "observability": 8,
        "intervention": 10,
        "validation": 8,
        "composition_hint": 8,
    }
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence_items:
        grouped[item.kind].append(item)

    selected: list[EvidenceItem] = []
    for kind in ("feature", "observability", "intervention", "validation", "composition_hint"):
        ranked = sorted(
            grouped.get(kind, []),
            key=lambda item: (
                -len(item.source_refs),
                -len(item.resource_domains),
                -len(item.keywords),
                item.title.lower(),
            ),
        )
        selected.extend(ranked[: per_kind_limits[kind]])

    return [
        {
            "evidence_id": item.evidence_id,
            "kind": item.kind,
            "title": item.title,
            "content": item.content,
            "keywords": item.keywords,
            "resource_domains": item.resource_domains,
            "source_quote": item.source_quote,
            "source_refs": item.source_refs,
            "confidence_note": item.confidence_note,
        }
        for item in selected
    ]


async def synthesize_library_manual(
    manual_id: str,
    library_name: str,
    source_artifacts: list[SourceArtifact],
    evidence_items: list[EvidenceItem],
    llm_client: ManualLLMProtocol,
    *,
    style_exemplars: dict[str, str] | None = None,
) -> LibraryManual:
    payload: dict[str, Any] = {
        "manual_id": manual_id,
        "library_name": library_name,
        "source_artifacts": [artifact.model_dump(mode="json") for artifact in source_artifacts],
        "evidence_items": _synthesis_evidence_payload(evidence_items),
        "style_exemplars": style_exemplars or _default_style_exemplars(),
        "schema": LibraryManual.model_json_schema(),
    }
    response_text = await llm_client.complete_prompt("synthesize_manual.txt", payload)
    parsed = extract_json_object(response_text)
    manual = LibraryManual.model_validate(parsed)
    return manual.model_copy(
        update={
            "manual_id": manual_id,
            "manual_type": "library",
            "library_name": library_name,
            "source_artifacts": source_artifacts,
        }
    )
