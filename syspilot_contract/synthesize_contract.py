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
    EXECUTOR_WORKER_CONTROL_NAMES,
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


def _canonicalize_component_attr_target(target: object, component_ids: set[str]) -> object:
    if not isinstance(target, str):
        return target
    text = target.strip()
    head, sep, remainder = text.partition(".")
    if not sep or head not in component_ids:
        return target
    if remainder == "attrs":
        return head
    if remainder.startswith("attrs."):
        return f"{head}.{remainder[len('attrs.') :]}"
    return target


def _canonicalize_target_prefixes(
    target: object,
    component_ids: set[str],
    use_ids: set[str],
) -> object:
    if not isinstance(target, str):
        return target
    parts = [part for part in target.strip().split(".") if part]
    if len(parts) < 2:
        return target
    first, second = parts[0], parts[1]
    if first == "use" and second in use_ids:
        return ".".join(parts[1:])
    if first in {"component", *COMPONENT_KINDS} and second in component_ids:
        return ".".join(parts[1:])
    return target


def _find_nested_attr_path(mapping: object, key: str) -> list[str] | None:
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return [key]
    for child_key, child_value in mapping.items():
        child_path = _find_nested_attr_path(child_value, key)
        if child_path is not None:
            return [str(child_key), *child_path]
    return None


def _set_component_workers_attr(mapping: object, worker_symbol: object) -> list[str] | None:
    if not isinstance(mapping, dict):
        return None
    if "workers" not in mapping:
        mapping["workers"] = worker_symbol
    return ["workers"]


def _looks_like_worker_control_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    stripped = name.strip()
    if stripped in EXECUTOR_WORKER_CONTROL_NAMES:
        return True
    upper = stripped.upper()
    lower = stripped.lower()
    return (
        "THREAD" in upper
        or lower.endswith("nthread")
        or lower.endswith("workers")
        or lower.endswith("n_jobs")
    )


def _canonicalize_contract_targets(payload: dict[str, Any]) -> dict[str, Any]:
    component_map = {
        str(item.get("id", "")).strip(): item.get("attrs", {})
        for item in payload.get("components", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    component_ids = set(component_map)
    use_ids = {
        str(item.get("id", "")).strip()
        for item in payload.get("uses", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    if not component_map:
        return payload

    control_section = payload.get("controls")
    control_target_by_id: dict[str, str] = {}
    if isinstance(control_section, list):
        for item in control_section:
            if not isinstance(item, dict):
                continue
            target = _canonicalize_target_prefixes(item.get("target"), component_ids, use_ids)
            target = _canonicalize_component_attr_target(target, component_ids)
            if isinstance(target, str):
                head, sep, remainder = target.partition(".")
                attrs = component_map.get(head)
                workers_path = _find_nested_attr_path(attrs, "workers")
                if (
                    head in component_map
                    and _looks_like_worker_control_name(item.get("name"))
                ):
                    if workers_path is None:
                        workers_path = _set_component_workers_attr(attrs, item.get("name"))
                    if workers_path is not None:
                        worker_target = f"{head}.{'.'.join(workers_path)}"
                        if not sep or remainder != ".".join(workers_path):
                            target = worker_target
            item["target"] = target
            control_id = str(item.get("id", "")).strip()
            if control_id and isinstance(target, str):
                control_target_by_id[control_id] = target

    signal_section = payload.get("signals")
    if isinstance(signal_section, list):
        for item in signal_section:
            if not isinstance(item, dict):
                continue
            target = _canonicalize_target_prefixes(item.get("target"), component_ids, use_ids)
            target = _canonicalize_component_attr_target(target, component_ids)
            if isinstance(target, str):
                target = control_target_by_id.get(target.strip(), target)
            item["target"] = target
    return payload


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
        lowered = issue.lower()
        keyword_sections = {
            "control ": "controls",
            "signal ": "signals",
            "assumption ": "assumptions",
            "predicate ": "assumptions",
            "component ": "components",
            "recipe ": "recipes",
            "use ": "uses",
        }
        for keyword, section_name in keyword_sections.items():
            if keyword in lowered and section_name not in sections:
                sections.append(section_name)
    return sections


def _normalized_library_contract(
    payload: object,
    contract_name: str,
    source_artifacts: list[SourceArtifact],
) -> tuple[LibraryContract | None, list[str], dict[str, Any]]:
    sanitized = _require_contract_object(payload)
    sanitized = _canonicalize_contract_targets(sanitized)
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
            return payload
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
