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
    parts: list[str] = []
    for item in path:
        if isinstance(item, int):
            parts.append(str(item))
        else:
            parts.append(str(item))
    return ".".join(parts)


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


def validate_contract_payload(payload: object) -> AnyContract:
    if not isinstance(payload, dict):
        raise ValueError("contract document must be a mapping")
    contract_type = str(payload.get("contract_type", "library")).strip() or "library"
    if contract_type == "composed":
        return ComposedContract.model_validate(payload)
    return LibraryContract.model_validate(payload)


def load_contract(path: Path) -> AnyContract:
    return validate_contract_payload(parse_contract_yaml(path.read_text(encoding="utf-8")))


def dump_contract_yaml(contract: AnyContract) -> str:
    return yaml.safe_dump(
        contract.model_dump(mode="json", by_alias=True),
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


def _ref_library(ref: str) -> str | None:
    if "." not in ref:
        return None
    library, _ = ref.split(".", 1)
    library = library.strip()
    return library or None


def _check_library_contract(contract: LibraryContract) -> list[str]:
    issues: list[str] = []
    issues.extend(_duplicate_id_issues("uses", [item.id for item in contract.uses]))
    issues.extend(_duplicate_id_issues("assumptions", [item.id for item in contract.assumptions]))
    issues.extend(_duplicate_id_issues("controls", [item.id for item in contract.controls]))
    issues.extend(_duplicate_id_issues("signals", [item.id for item in contract.signals]))
    issues.extend(_duplicate_id_issues("recipes", [item.id for item in contract.recipes]))

    use_by_id = {item.id: item for item in contract.uses}
    signal_ids = {item.id for item in contract.signals}
    control_ids = {item.id for item in contract.controls}

    for control in contract.controls:
        missing_uses = [item for item in control.affects if item not in use_by_id]
        if missing_uses:
            issues.append(
                f"controls:{control.id} affects unknown uses: {', '.join(sorted(missing_uses))}"
            )
        affected_resources = {
            use_by_id[item].resource for item in control.affects if item in use_by_id
        }
        if affected_resources and control.effect.resource not in affected_resources:
            issues.append(
                f"controls:{control.id} effect.resource={control.effect.resource} does not match affected uses"
            )
        if (
            control.type == "enum"
            and control.allowed_values
            and isinstance(control.activation_value, str)
            and control.activation_value not in control.allowed_values
        ):
            issues.append(
                f"controls:{control.id} activation_value={control.activation_value} is not in allowed_values"
            )

    for recipe in contract.recipes:
        missing_signals = [item for item in recipe.collect if item not in signal_ids]
        if missing_signals:
            issues.append(
                f"recipes:{recipe.id} collects unknown signals: {', '.join(sorted(missing_signals))}"
            )
        missing_controls = [item for item in recipe.control_refs if item not in control_ids]
        if missing_controls:
            issues.append(
                f"recipes:{recipe.id} references unknown controls: {', '.join(sorted(missing_controls))}"
            )

    return issues


def _check_composed_contract(contract: ComposedContract) -> list[str]:
    issues: list[str] = []
    selected = set(contract.selected_contracts)
    instance_ids = [item.instance_id for item in contract.contract_instances]
    issues.extend(_duplicate_id_issues("contract_instances", instance_ids))
    if contract.selected_contracts != instance_ids:
        issues.append("selected_contracts must match contract_instances.instance_id order")
    for item in contract.shared_resources:
        unknown_libraries = [library for library in item.libraries if library not in selected]
        if unknown_libraries:
            issues.append(
                f"shared_resources:{item.resource}:{item.phase} references unknown libraries: {', '.join(sorted(unknown_libraries))}"
            )
    for item in contract.hypotheses:
        unknown_libraries = [library for library in item.libraries if library not in selected]
        if unknown_libraries:
            issues.append(
                f"hypotheses:{item.id} references unknown libraries: {', '.join(sorted(unknown_libraries))}"
            )
        if (
            not item.basis.overlaps
            and not item.basis.violated_assumptions
            and not item.basis.colliding_controls
        ):
            issues.append(f"hypotheses:{item.id} is missing a supporting basis")
        if not item.validation.collect:
            issues.append(f"hypotheses:{item.id} must include validation signals")
        ref_sections = {
            "overlaps": [ref for overlap in item.basis.overlaps for ref in overlap.uses],
            "violated_assumptions": item.basis.violated_assumptions,
            "colliding_controls": item.basis.colliding_controls,
            "validation": item.validation.collect,
            "candidate_control_refs": item.candidate_control_refs,
            "candidate_recipe_refs": item.candidate_recipe_refs,
        }
        for section_name, refs in ref_sections.items():
            invalid_refs = [
                ref
                for ref in refs
                if (_ref_library(ref) is None or _ref_library(ref) not in selected)
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


def validate_contract_text(text: str) -> tuple[AnyContract | None, ContractCheckResult]:
    try:
        payload = parse_contract_yaml(text)
        contract = validate_contract_payload(payload)
    except ValidationError as exc:
        return None, ContractCheckResult(valid=False, issues=_issues_from_validation_error(exc))
    except (ValueError, yaml.YAMLError) as exc:
        return None, ContractCheckResult(valid=False, issues=[str(exc)])
    result = check_contract(contract)
    return contract, result


def validate_contract_file(path: Path) -> tuple[AnyContract | None, ContractCheckResult]:
    return validate_contract_text(path.read_text(encoding="utf-8"))


def assert_valid_contract(contract: AnyContract) -> AnyContract:
    result = check_contract(contract)
    if not result.valid:
        raise ValueError("; ".join(result.issues))
    return contract
