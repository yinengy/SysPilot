from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml
from pydantic import ValidationError
from syspilot_contract.schemas import (
    AnyContract,
    ComposedContract,
    ContractCheckResult,
    LibraryContract,
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ValueError(f"duplicate key '{key}' at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _path_string(path: tuple[object, ...]) -> str:
    return ".".join(str(item) for item in path)


def _issues_from_validation_error(exc: ValidationError) -> list[str]:
    return [f"{_path_string(tuple(error['loc']))}: {error['msg']}" for error in exc.errors()]


def parse_contract_yaml(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped:
        raise ValueError("contract document is empty")
    documents = list(yaml.load_all(stripped, Loader=_UniqueKeySafeLoader))
    if len(documents) != 1:
        raise ValueError("contract documents must contain exactly one YAML document")
    payload = documents[0]
    if not isinstance(payload, dict):
        raise ValueError("contract document must be a YAML mapping")
    return payload


def _normalized_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("contract document must be a mapping")
    normalized = dict(payload)
    contract_type = normalized.get("type")
    if contract_type not in {"library", "composed"}:
        raise ValueError(f"Unknown contract type {contract_type}")
    return normalized


def validate_contract_payload(payload: object) -> AnyContract:
    normalized = _normalized_payload(payload)
    contract_type = normalized.get("type")
    if contract_type == "composed":
        return ComposedContract.model_validate(normalized)
    return LibraryContract.model_validate(normalized)


def load_contract(path: Path) -> AnyContract:
    return validate_contract_payload(parse_contract_yaml(path.read_text(encoding="utf-8")))


def dump_contract_yaml(contract: AnyContract) -> str:
    return yaml.safe_dump(
        contract.model_dump(mode="json", by_alias=True, exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )


def _duplicate_id_issues(section: str, ids: list[str]) -> list[str]:
    counts = Counter(ids)
    return [
        f"{section}:{item_id} is duplicated"
        for item_id, count in sorted(counts.items())
        if count > 1
    ]


def _ref_contract(ref: str) -> str | None:
    if "." not in ref:
        return None
    contract_name, _ = ref.split(".", 1)
    contract_name = contract_name.strip()
    return contract_name or None


def _check_library_contract(contract: LibraryContract) -> list[str]:
    issues: list[str] = []
    issues.extend(_duplicate_id_issues("components", [item.id for item in contract.components]))
    issues.extend(_duplicate_id_issues("uses", [item.id for item in contract.uses]))
    issues.extend(_duplicate_id_issues("assumptions", [item.id for item in contract.assumptions]))
    issues.extend(_duplicate_id_issues("controls", [item.id for item in contract.controls]))
    issues.extend(_duplicate_id_issues("signals", [item.id for item in contract.signals]))
    issues.extend(_duplicate_id_issues("recipes", [item.id for item in contract.recipes]))
    return issues


def _check_composed_contract(contract: ComposedContract) -> list[str]:
    issues: list[str] = []
    selected = set(contract.selected_contracts)
    instance_ids = [item.instance_id for item in contract.contract_instances]
    issues.extend(_duplicate_id_issues("contract_instances", instance_ids))
    if contract.selected_contracts != instance_ids:
        issues.append("selected_contracts must match contract_instances.instance_id order")

    for item in contract.shared_resources:
        unknown_contracts = [name for name in item.contracts if name not in selected]
        if unknown_contracts:
            issues.append(
                f"shared_resources:{item.resource}:{item.phase} references unknown contracts: {', '.join(sorted(unknown_contracts))}"
            )
        invalid_refs = [
            ref
            for ref in item.uses + item.components
            if (_ref_contract(ref) is None or _ref_contract(ref) not in selected)
        ]
        if invalid_refs:
            issues.append(
                f"shared_resources:{item.resource}:{item.phase} has invalid refs: {', '.join(sorted(invalid_refs))}"
            )

    for item in contract.hypotheses:
        unknown_contracts = [name for name in item.contracts if name not in selected]
        if unknown_contracts:
            issues.append(
                f"hypotheses:{item.id} references unknown contracts: {', '.join(sorted(unknown_contracts))}"
            )
        ref_sections = {
            "components": item.components,
            "uses": item.uses,
            "validation": item.validation.collect,
            "candidate_actions": [ref for ref in item.candidate_actions if "." in ref],
            "recipes": item.recipes,
            "overlaps": [
                ref for overlap in item.basis.overlaps for ref in overlap.uses + overlap.components
            ],
            "assumption_violations": [
                violation.assumption
                for violation in item.basis.assumption_violations
                if "." in violation.assumption
            ]
            + [
                ref
                for violation in item.basis.assumption_violations
                for ref in violation.conflicting_uses
            ],
            "control_collisions": [
                ref
                for collision in item.basis.control_collisions
                for ref in collision.controls
                if "." in ref
            ],
        }
        for section_name, refs in ref_sections.items():
            invalid_refs = [
                ref
                for ref in refs
                if (_ref_contract(ref) is None or _ref_contract(ref) not in selected)
            ]
            if invalid_refs:
                issues.append(
                    f"hypotheses:{item.id} has invalid {section_name}: {', '.join(sorted(invalid_refs))}"
                )
    return issues


def check_contract(contract: AnyContract) -> ContractCheckResult:
    issues = (
        _check_library_contract(contract)
        if isinstance(contract, LibraryContract)
        else _check_composed_contract(contract)
    )
    return ContractCheckResult(valid=not issues, issues=issues)


def validate_contract_text(
    text: str,
) -> tuple[AnyContract | None, ContractCheckResult]:
    try:
        payload = parse_contract_yaml(text)
        contract = validate_contract_payload(payload)
    except ValidationError as exc:
        return None, ContractCheckResult(valid=False, issues=_issues_from_validation_error(exc))
    except (ValueError, yaml.YAMLError) as exc:
        return None, ContractCheckResult(valid=False, issues=[str(exc)])
    result = check_contract(contract)
    return contract, result


def validate_contract_file(
    path: Path,
) -> tuple[AnyContract | None, ContractCheckResult]:
    return validate_contract_text(path.read_text(encoding="utf-8"))


def assert_valid_contract(contract: AnyContract) -> AnyContract:
    result = check_contract(contract)
    if not result.valid:
        raise ValueError("; ".join(result.issues))
    return contract
