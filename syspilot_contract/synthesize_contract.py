from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from syspilot_contract.checker import assert_valid_contract
from syspilot_contract.schemas import LibraryContract, SourceArtifact
from syspilot_manual.llm_client import ManualLLMProtocol, extract_json_object
from syspilot_manual.schemas import EvidenceItem


def _clean_id_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _sanitize_contract_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Contract synthesis returned a non-object payload")
    sanitized = deepcopy(payload)

    def _filter_source_ref_items(section: str) -> None:
        raw_items = sanitized.get(section)
        if not isinstance(raw_items, list):
            return
        filtered_items: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            source_refs = _clean_id_list(item.get("source_refs"))
            if not source_refs:
                continue
            filtered_item = dict(item)
            filtered_item["source_refs"] = source_refs
            filtered_items.append(filtered_item)
        sanitized[section] = filtered_items

    for section in ("uses", "assumptions", "signals"):
        _filter_source_ref_items(section)

    raw_controls = sanitized.get("controls")
    if isinstance(raw_controls, list):
        filtered_controls: list[dict[str, Any]] = []
        for item in raw_controls:
            if not isinstance(item, dict):
                continue
            affects = _clean_id_list(item.get("affects"))
            source_refs = _clean_id_list(item.get("source_refs"))
            if not affects or not source_refs:
                continue
            filtered_item = dict(item)
            filtered_item["affects"] = affects
            filtered_item["source_refs"] = source_refs
            filtered_controls.append(filtered_item)
        sanitized["controls"] = filtered_controls

    raw_recipes = sanitized.get("recipes")
    if isinstance(raw_recipes, list):
        filtered_recipes: list[dict[str, Any]] = []
        for item in raw_recipes:
            if not isinstance(item, dict):
                continue
            collect = _clean_id_list(item.get("collect"))
            actions = _clean_id_list(item.get("actions"))
            source_refs = _clean_id_list(item.get("source_refs"))
            if not collect or not actions or not source_refs:
                continue
            filtered_item = dict(item)
            filtered_item["collect"] = collect
            filtered_item["actions"] = actions
            filtered_item["source_refs"] = source_refs
            filtered_recipes.append(filtered_item)
        sanitized["recipes"] = filtered_recipes
    return sanitized


def _selected_evidence_payload(evidence_items: list[EvidenceItem]) -> list[dict[str, object]]:
    per_kind_limits = {
        "feature": 10,
        "observability": 10,
        "intervention": 12,
        "validation": 10,
        "composition_hint": 10,
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


async def synthesize_library_contract(
    contract_name: str,
    source_artifacts: list[SourceArtifact],
    evidence_items: list[EvidenceItem],
    llm_client: ManualLLMProtocol,
) -> LibraryContract:
    payload: dict[str, Any] = {
        "contract": contract_name,
        "source_artifacts": [artifact.model_dump(mode="json") for artifact in source_artifacts],
        "evidence_items": _selected_evidence_payload(evidence_items),
        "schema": LibraryContract.model_json_schema(),
        "resource_vocab": [
            "cpu.core",
            "cpu.thread",
            "cpu.affinity",
            "gpu.device",
            "gpu.stream",
            "mem.capacity",
            "mem.bandwidth",
            "mem.allocator",
            "io.bandwidth",
            "io.queue",
            "net.bandwidth",
            "worker.pool",
            "cache.capacity",
        ],
    }
    response_text = await llm_client.complete_prompt("synthesize_contract.txt", payload)
    parsed = _sanitize_contract_payload(extract_json_object(response_text))
    contract = LibraryContract.model_validate(parsed)
    verified = assert_valid_contract(
        contract.model_copy(
            update={
                "contract": contract_name,
                "contract_type": "library",
                "version": "lpcl/v0",
                "source_artifacts": source_artifacts,
            }
        )
    )
    if not isinstance(verified, LibraryContract):
        raise TypeError("Expected a library contract after validation")
    return verified
