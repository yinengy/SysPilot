from __future__ import annotations

from syspilot_manual.schemas import EvidenceItem, RESOURCE_DOMAINS


def _merge_key(item: EvidenceItem) -> tuple[str, str, tuple[str, ...]]:
    return (
        item.kind,
        item.title.strip().lower(),
        tuple(keyword.lower() for keyword in item.keywords),
    )


def merge_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    merged: dict[tuple[str, str, tuple[str, ...]], EvidenceItem] = {}
    for item in items:
        key = _merge_key(item)
        if key not in merged:
            merged[key] = item
            continue

        existing = merged[key]
        merged[key] = existing.model_copy(
            update={
                "content": existing.content
                if len(existing.content) >= len(item.content)
                else item.content,
                "resource_domains": [
                    domain
                    for domain in RESOURCE_DOMAINS
                    if domain in set(existing.resource_domains) | set(item.resource_domains)
                ],
                "source_refs": sorted(set(existing.source_refs) | set(item.source_refs)),
                "source_quote": existing.source_quote
                if len(existing.source_quote) >= len(item.source_quote)
                else item.source_quote,
                "confidence_note": existing.confidence_note or item.confidence_note,
            }
        )

    return sorted(
        merged.values(), key=lambda item: (item.kind, item.title.lower(), item.evidence_id)
    )
