from __future__ import annotations

from collections import defaultdict
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
    EXPECTED_DIRECTIONS,
    HYPOTHESIS_KINDS,
    LibraryContract,
    PHASES,
    PREDICATE_KINDS,
    RESOURCE_TYPES,
    SIGNAL_ROLES,
    SCOPES,
    SHARE_MODES,
    SourceArtifact,
)
from syspilot_manual.llm_client import ManualLLMProtocol
from syspilot_manual.schemas import EvidenceItem


def _require_contract_object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Contract synthesis returned a non-object payload")
    return dict(payload)


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
    sanitized = _require_contract_object(payload)
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
        "signal_role_vocab": list(SIGNAL_ROLES),
        "expected_direction_when_mitigated_vocab": list(EXPECTED_DIRECTIONS),
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
