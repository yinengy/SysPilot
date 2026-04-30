from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from syspilot_contract.checker import (
    assert_valid_contract,
    parse_contract_yaml,
    validate_contract_payload,
)
from syspilot_contract.schemas import (
    AnyContract,
    ComposedContract,
    ContractInstance,
    ConflictHypothesis,
    Control,
    HypothesisBasis,
    LibraryContract,
    SharedResourceItem,
)


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

DEMAND_KEY_ORDER = ("dimension", "min", "max", "value", "source")
PREDICATE_KEY_ORDER = ("kind",)
EFFECT_KEY_ORDER = ("kind",)
PROBE_KEY_ORDER = ("kind", "name")


def _quote_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _require_safe_identifier(value: str, label: str) -> str:
    if not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid LPCL identifier for {label}: {value!r}")
    return value


def _format_scalar(value: object) -> str:
    if value is None:
        raise ValueError("Null fields cannot be compiled to LPCL")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, str):
        return value if SAFE_IDENTIFIER_RE.fullmatch(value) else _quote_string(value)
    raise TypeError(f"Unsupported scalar value for LPCL compilation: {type(value)!r}")


def _format_list(values: Sequence[object]) -> str:
    return "[" + ", ".join(_format_scalar(value) for value in values) + "]"


def _ordered_keys(mapping: Mapping[str, Any], preferred: Sequence[str] | None = None) -> list[str]:
    keys = list(mapping.keys())
    if preferred is None:
        return keys
    ordered = [key for key in preferred if key in mapping]
    ordered.extend(key for key in keys if key not in ordered)
    return ordered


def _emit_mapping_entries(
    lines: list[str],
    mapping: Mapping[str, Any],
    *,
    indent: str,
    preferred_keys: Sequence[str] | None = None,
) -> None:
    for key in _ordered_keys(mapping, preferred_keys):
        _require_safe_identifier(key, f"mapping key {key}")
        value = mapping[key]
        if value is None:
            raise ValueError(f"Null field '{key}' cannot be compiled to LPCL")
        if isinstance(value, Mapping):
            lines.append(f"{indent}{key} {{")
            _emit_mapping_entries(lines, value, indent=indent + "  ")
            lines.append(indent + "}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            lines.append(f"{indent}{key} = {_format_list(value)};")
        else:
            lines.append(f"{indent}{key} = {_format_scalar(value)};")


def _emit_named_block(
    lines: list[str],
    name: str,
    mapping: Mapping[str, Any],
    *,
    indent: str,
    preferred_keys: Sequence[str] | None = None,
) -> None:
    lines.append(f"{indent}{name} {{")
    _emit_mapping_entries(lines, mapping, indent=indent + "  ", preferred_keys=preferred_keys)
    lines.append(indent + "}")


def _emit_scalar_assignment(lines: list[str], key: str, value: object, *, indent: str) -> None:
    lines.append(f"{indent}{key} = {_format_scalar(value)};")


def _emit_list_assignment(
    lines: list[str], key: str, values: Sequence[object], *, indent: str
) -> None:
    lines.append(f"{indent}{key} = {_format_list(values)};")


def _emit_evidence(
    lines: list[str],
    evidence: Sequence[object],
    *,
    indent: str,
    include_evidence: bool,
) -> None:
    if include_evidence:
        _emit_list_assignment(lines, "evidence", evidence, indent=indent)


def _compile_component(item: Any, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"component id {item.id}")
    _require_safe_identifier(item.kind, f"component kind {item.kind}")
    lines = [f"  component {item.id} : {item.kind} {{"]
    _emit_named_block(lines, "attrs", item.attrs, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_use(item: Any, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"use id {item.id}")
    _require_safe_identifier(item.resource, f"use resource {item.resource}")
    _require_safe_identifier(item.by, f"use component ref {item.by}")
    _require_safe_identifier(item.phase, f"use phase {item.phase}")
    _require_safe_identifier(item.scope, f"use scope {item.scope}")
    _require_safe_identifier(item.share, f"use share mode {item.share}")
    lines = [f"  use {item.id} : {item.resource} {{"]
    _emit_scalar_assignment(lines, "by", item.by, indent="    ")
    _emit_scalar_assignment(lines, "phase", item.phase, indent="    ")
    _emit_scalar_assignment(lines, "scope", item.scope, indent="    ")
    _emit_named_block(lines, "demand", item.demand, indent="    ", preferred_keys=DEMAND_KEY_ORDER)
    _emit_scalar_assignment(lines, "share", item.share, indent="    ")
    if item.when is not None:
        _emit_scalar_assignment(lines, "when", item.when, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_assumption(item: Any, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"assumption id {item.id}")
    _require_safe_identifier(item.resource, f"assumption resource {item.resource}")
    lines = [f"  assume {item.id} : {item.resource} {{"]
    _emit_named_block(
        lines,
        "predicate",
        item.predicate,
        indent="    ",
        preferred_keys=PREDICATE_KEY_ORDER,
    )
    if item.when is not None:
        _emit_scalar_assignment(lines, "when", item.when, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_control(item: Control, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"control id {item.id}")
    _require_safe_identifier(item.kind, f"control kind {item.kind}")
    _require_safe_identifier(item.type, f"control type {item.type}")
    lines = [f"  control {item.id} : {item.kind}<{item.type}> {{"]
    _emit_scalar_assignment(lines, "name", item.name, indent="    ")
    _emit_scalar_assignment(lines, "target", item.target, indent="    ")
    _emit_named_block(
        lines,
        "effect",
        item.effect.model_dump(mode="json", exclude_none=True),
        indent="    ",
        preferred_keys=EFFECT_KEY_ORDER,
    )
    if item.when is not None:
        _emit_scalar_assignment(lines, "when", item.when, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_signal(item: Any, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"signal id {item.id}")
    lines = [f"  signal {item.id} {{"]
    _emit_scalar_assignment(lines, "target", item.target, indent="    ")
    _emit_scalar_assignment(lines, "metric", item.metric, indent="    ")
    _emit_scalar_assignment(lines, "indicates", item.indicates, indent="    ")
    _emit_named_block(
        lines,
        "probe",
        item.probe.model_dump(mode="json"),
        indent="    ",
        preferred_keys=PROBE_KEY_ORDER,
    )
    if item.when is not None:
        _emit_scalar_assignment(lines, "when", item.when, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_recipe(item: Any, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"recipe id {item.id}")
    _require_safe_identifier(item.hypothesis_kind, f"recipe hypothesis kind {item.hypothesis_kind}")
    lines = [f"  recipe {item.id} {{"]
    _emit_scalar_assignment(lines, "for", item.hypothesis_kind, indent="    ")
    _emit_list_assignment(lines, "collect", item.collect, indent="    ")
    _emit_list_assignment(lines, "actions", item.actions, indent="    ")
    _emit_scalar_assignment(lines, "accept", item.accept, indent="    ")
    _emit_scalar_assignment(lines, "rollback", item.rollback, indent="    ")
    if item.when is not None:
        _emit_scalar_assignment(lines, "when", item.when, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_contract_instance(item: ContractInstance) -> list[str]:
    _require_safe_identifier(item.instance_id, f"instance id {item.instance_id}")
    lines = [f"  instance {item.instance_id} {{"]
    _emit_scalar_assignment(lines, "contract", item.contract_id, indent="    ")
    lines.append("  }")
    return lines


def _compile_shared_resource(item: SharedResourceItem, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.resource, f"shared resource {item.resource}")
    _require_safe_identifier(item.phase, f"shared resource phase {item.phase}")
    lines = [f"  shared_resource {item.resource} {{"]
    _emit_scalar_assignment(lines, "phase", item.phase, indent="    ")
    _emit_list_assignment(lines, "contracts", item.contracts, indent="    ")
    _emit_list_assignment(lines, "components", item.components, indent="    ")
    _emit_list_assignment(lines, "uses", item.uses, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _emit_basis(lines: list[str], basis: HypothesisBasis, *, indent: str) -> None:
    lines.append(f"{indent}basis {{")
    for overlap in basis.overlaps:
        lines.append(f"{indent}  overlap {{")
        _emit_list_assignment(lines, "uses", overlap.uses, indent=indent + "    ")
        _emit_list_assignment(lines, "components", overlap.components, indent=indent + "    ")
        _emit_list_assignment(lines, "contracts", overlap.contracts, indent=indent + "    ")
        lines.append(indent + "  }")
    for violation in basis.assumption_violations:
        lines.append(f"{indent}  assumption_violation {{")
        _emit_scalar_assignment(lines, "assumption", violation.assumption, indent=indent + "    ")
        _emit_list_assignment(
            lines,
            "conflicting_uses",
            violation.conflicting_uses,
            indent=indent + "    ",
        )
        lines.append(indent + "  }")
    for collision in basis.control_collisions:
        lines.append(f"{indent}  control_collision {{")
        _emit_scalar_assignment(lines, "resource", collision.resource, indent=indent + "    ")
        _emit_scalar_assignment(lines, "dimension", collision.dimension, indent=indent + "    ")
        _emit_list_assignment(lines, "controls", collision.controls, indent=indent + "    ")
        lines.append(indent + "  }")
    lines.append(indent + "}")


def _compile_hypothesis(item: ConflictHypothesis, *, include_evidence: bool) -> list[str]:
    _require_safe_identifier(item.id, f"hypothesis id {item.id}")
    _require_safe_identifier(item.kind, f"hypothesis kind {item.kind}")
    _require_safe_identifier(item.resource, f"hypothesis resource {item.resource}")
    _require_safe_identifier(item.phase, f"hypothesis phase {item.phase}")
    lines = [f"  hypothesis {item.id} : {item.kind} {{"]
    _emit_scalar_assignment(lines, "resource", item.resource, indent="    ")
    _emit_scalar_assignment(lines, "phase", item.phase, indent="    ")
    _emit_list_assignment(lines, "contracts", item.contracts, indent="    ")
    _emit_list_assignment(lines, "components", item.components, indent="    ")
    _emit_list_assignment(lines, "uses", item.uses, indent="    ")
    _emit_basis(lines, item.basis, indent="    ")
    lines.append("    validate {")
    _emit_list_assignment(lines, "collect", item.validation.collect, indent="      ")
    lines.append("    }")
    _emit_list_assignment(lines, "candidate_actions", item.candidate_actions, indent="    ")
    _emit_list_assignment(lines, "recipes", item.recipes, indent="    ")
    _emit_evidence(lines, item.evidence, indent="    ", include_evidence=include_evidence)
    lines.append("  }")
    return lines


def _compile_library_contract(contract: LibraryContract, *, include_evidence: bool) -> str:
    _require_safe_identifier(contract.contract, f"contract name {contract.contract}")
    declaration_blocks: list[list[str]] = []
    declaration_blocks.extend(
        _compile_component(item, include_evidence=include_evidence) for item in contract.components
    )
    declaration_blocks.extend(
        _compile_use(item, include_evidence=include_evidence) for item in contract.uses
    )
    declaration_blocks.extend(
        _compile_assumption(item, include_evidence=include_evidence)
        for item in contract.assumptions
    )
    declaration_blocks.extend(
        _compile_control(item, include_evidence=include_evidence) for item in contract.controls
    )
    declaration_blocks.extend(
        _compile_signal(item, include_evidence=include_evidence) for item in contract.signals
    )
    declaration_blocks.extend(
        _compile_recipe(item, include_evidence=include_evidence) for item in contract.recipes
    )

    lines = ['lpcl "lpcl/v1";', "", f"contract {contract.contract} : library {{"]
    if declaration_blocks:
        for block_index, block in enumerate(declaration_blocks):
            lines.extend(block)
            if block_index != len(declaration_blocks) - 1:
                lines.append("")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _compile_composed_contract(contract: ComposedContract, *, include_evidence: bool) -> str:
    _require_safe_identifier(contract.contract, f"contract name {contract.contract}")
    declaration_blocks: list[list[str]] = []
    declaration_blocks.append(
        [
            "  selected_contracts {",
            f"    values = {_format_list(contract.selected_contracts)};",
            "  }",
        ]
    )
    declaration_blocks.extend(
        _compile_contract_instance(item) for item in contract.contract_instances
    )
    declaration_blocks.extend(
        _compile_shared_resource(item, include_evidence=include_evidence)
        for item in contract.shared_resources
    )
    declaration_blocks.extend(
        _compile_hypothesis(item, include_evidence=include_evidence) for item in contract.hypotheses
    )

    lines = ['lpcl "lpcl/v1";', "", f"contract {contract.contract} : composed {{"]
    if declaration_blocks:
        for block_index, block in enumerate(declaration_blocks):
            lines.extend(block)
            if block_index != len(declaration_blocks) - 1:
                lines.append("")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_lpcl(contract: AnyContract, *, include_evidence: bool = False) -> str:
    validated = assert_valid_contract(contract)
    if isinstance(validated, LibraryContract):
        return _compile_library_contract(validated, include_evidence=include_evidence)
    if isinstance(validated, ComposedContract):
        return _compile_composed_contract(validated, include_evidence=include_evidence)
    raise TypeError(f"Unsupported contract type for LPCL compilation: {type(validated)!r}")


def compile_lpcl_from_yaml_text(yaml_text: str, *, include_evidence: bool = False) -> str:
    payload = parse_contract_yaml(yaml_text)
    contract = validate_contract_payload(payload, allow_migration=False)
    return compile_lpcl(contract, include_evidence=include_evidence)


def compile_lpcl_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    include_evidence: bool = False,
) -> None:
    source_path = Path(input_path)
    target_path = Path(output_path)
    compiled = compile_lpcl_from_yaml_text(
        source_path.read_text(encoding="utf-8"),
        include_evidence=include_evidence,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(compiled, encoding="utf-8")
