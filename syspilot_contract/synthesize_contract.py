from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from syspilot_contract.checker import check_contract, parse_contract_yaml
from syspilot_contract.schemas import (
    COMPONENT_KINDS,
    CONTRACT_VERSION,
    CONTROL_KINDS,
    CONTROL_TYPES,
    ContractVerificationResult,
    EFFECT_KINDS,
    HYPOTHESIS_KINDS,
    LibraryContract,
    PHASES,
    PREDICATE_KINDS,
    PROBE_KINDS,
    RESOURCE_TYPES,
    SCOPES,
    SHARE_MODES,
    SourceArtifact,
)
from syspilot_manual.llm_client import ManualLLMProtocol
from syspilot_manual.schemas import EvidenceItem


def _clean_id_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _normalize_condition(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = _clean_id_list(value)
        return "; ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _sanitize_contract_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Contract synthesis returned a non-object payload")
    sanitized = deepcopy(payload)

    if "type" not in sanitized and "contract_type" in sanitized:
        sanitized["type"] = sanitized.pop("contract_type")

    for section in ("components", "uses", "assumptions", "controls", "signals", "recipes"):
        raw_items = sanitized.get(section)
        if not isinstance(raw_items, list):
            continue
        normalized_items: list[object] = []
        for item in raw_items:
            if not isinstance(item, dict):
                normalized_items.append(item)
                continue
            filtered_item = dict(item)
            if "evidence" not in filtered_item and "source_refs" in filtered_item:
                filtered_item["evidence"] = _clean_id_list(filtered_item.pop("source_refs"))
            if "when" in filtered_item:
                filtered_item["when"] = _normalize_condition(filtered_item.get("when"))
            if section == "signals":
                probe = filtered_item.get("probe")
                if not isinstance(probe, dict):
                    probe = {}
                if "name" not in probe and "target" in probe:
                    probe["name"] = probe.pop("target")
                if not probe.get("name") and filtered_item.get("tool"):
                    probe["name"] = filtered_item.pop("tool")
                filtered_item["probe"] = probe
            if section == "recipes":
                if "for" not in filtered_item and "hypothesis_kind" in filtered_item:
                    filtered_item["for"] = filtered_item.pop("hypothesis_kind")
                if "accept" not in filtered_item and "accept_if" in filtered_item:
                    filtered_item["accept"] = filtered_item.pop("accept_if")
                if "rollback" not in filtered_item and "rollback_if" in filtered_item:
                    filtered_item["rollback"] = filtered_item.pop("rollback_if")
            normalized_items.append(filtered_item)
        sanitized[section] = normalized_items

    raw_assumptions = sanitized.get("assumptions")
    if isinstance(raw_assumptions, list):
        for item in raw_assumptions:
            if not isinstance(item, dict):
                continue
            predicate = item.get("predicate")
            if not isinstance(predicate, dict):
                continue
            kind = str(predicate.get("kind", "")).strip()
            if kind in {"total_at_most", "competitors_at_most"}:
                item["predicate"] = {"kind": kind, "value": predicate.get("value")}
            elif kind in {"exclusive", "not_nested", "placement_compatible"}:
                item["predicate"] = {"kind": kind}

    raw_controls = sanitized.get("controls")
    if isinstance(raw_controls, list):
        for item in raw_controls:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target", "")).strip()
            if ".attrs." in target:
                item["target"] = target.replace(".attrs.", ".")
            effect = item.get("effect")
            if not isinstance(effect, dict):
                continue
            kind = str(effect.get("kind", "")).strip()
            if kind == "set":
                item["effect"] = {"kind": "set", "value": effect.get("value", "input")}
            elif kind == "partition":
                filtered_effect: dict[str, Any] = {"kind": "partition"}
                if effect.get("resource") is not None:
                    filtered_effect["resource"] = effect.get("resource")
                item["effect"] = filtered_effect
            elif kind in {"increase", "decrease", "disable", "enable"}:
                item["effect"] = {"kind": kind}
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


def _format_validation_issues(exc: ValidationError) -> list[str]:
    issues: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        message = str(error.get("msg", "invalid contract payload"))
        issues.append(f"{location}: {message}" if location else message)
    return issues


def _invalid_sections(issues: list[str]) -> list[str]:
    sections: list[str] = []
    valid_sections = {
        "contract",
        "type",
        "version",
        "components",
        "uses",
        "assumptions",
        "controls",
        "signals",
        "recipes",
    }
    for issue in issues:
        prefix = issue.split(":", 1)[0].split(".", 1)[0].strip()
        if prefix in valid_sections and prefix not in sections:
            sections.append(prefix)
    return sections


def _normalized_library_contract(
    payload: object,
    contract_name: str,
    source_artifacts: list[SourceArtifact],
) -> tuple[LibraryContract | None, list[str], dict[str, Any]]:
    sanitized = _sanitize_contract_payload(payload)
    try:
        contract = LibraryContract.model_validate(sanitized)
    except ValidationError as exc:
        return None, _format_validation_issues(exc), sanitized

    normalized = contract.model_copy(
        update={
            "contract": contract_name,
            "contract_type": "library",
            "version": CONTRACT_VERSION,
            "source_artifacts": source_artifacts,
        }
    )
    result = check_contract(normalized)
    if not result.valid:
        return None, result.issues, sanitized
    return normalized, [], sanitized


async def _repair_library_contract_payload(
    payload: dict[str, Any],
    *,
    issues: list[str],
    contract_name: str,
    source_artifacts: list[SourceArtifact],
    llm_client: ManualLLMProtocol,
) -> dict[str, Any]:
    repair_payload: dict[str, Any] = {
        "contract": payload,
        "target_contract": contract_name,
        "source_artifacts": [artifact.model_dump(mode="json") for artifact in source_artifacts],
        "issues": issues,
        "invalid_sections": _invalid_sections(issues),
        "schema": LibraryContract.model_json_schema(),
        "repair_schema": ContractVerificationResult.model_json_schema(),
    }
    response_text = await llm_client.complete_prompt("verify_contract.txt", repair_payload)
    parsed = parse_contract_yaml(response_text)
    verification = ContractVerificationResult.model_validate(parsed)

    if verification.valid and verification.repaired_contract is None:
        if issues:
            raise ValueError(
                "Contract verification reported valid without providing required repair payload"
            )
        return payload

    if verification.repaired_contract is None:
        raise ValueError(
            "Contract verification returned invalid without a repaired_contract payload"
        )

    return verification.repaired_contract


async def synthesize_library_contract(
    contract_name: str,
    source_artifacts: list[SourceArtifact],
    evidence_items: list[EvidenceItem],
    llm_client: ManualLLMProtocol,
) -> LibraryContract:
    max_repair_attempts = 2
    payload: dict[str, Any] = {
        "contract": contract_name,
        "source_artifacts": [artifact.model_dump(mode="json") for artifact in source_artifacts],
        "evidence_items": _selected_evidence_payload(evidence_items),
        "schema": LibraryContract.model_json_schema(),
        "version": CONTRACT_VERSION,
        "resource_vocab": list(RESOURCE_TYPES),
        "component_kind_vocab": list(COMPONENT_KINDS),
        "phase_vocab": list(PHASES),
        "scope_vocab": list(SCOPES),
        "share_vocab": list(SHARE_MODES),
        "predicate_vocab": list(PREDICATE_KINDS),
        "control_kind_vocab": list(CONTROL_KINDS),
        "control_type_vocab": list(CONTROL_TYPES),
        "effect_kind_vocab": list(EFFECT_KINDS),
        "probe_kind_vocab": list(PROBE_KINDS),
        "hypothesis_kind_vocab": list(HYPOTHESIS_KINDS),
    }
    response_text = await llm_client.complete_prompt("synthesize_contract.txt", payload)
    current_payload = parse_contract_yaml(response_text)

    for attempt in range(max_repair_attempts + 1):
        verified, issues, sanitized = _normalized_library_contract(
            current_payload,
            contract_name,
            source_artifacts,
        )
        if verified is not None:
            return verified
        if attempt >= max_repair_attempts:
            raise ValueError(
                "Contract verification failed after repair attempts: " + "; ".join(issues)
            )
        current_payload = await _repair_library_contract_payload(
            sanitized,
            issues=issues,
            contract_name=contract_name,
            source_artifacts=source_artifacts,
            llm_client=llm_client,
        )

    raise RuntimeError("Unreachable contract synthesis state")
