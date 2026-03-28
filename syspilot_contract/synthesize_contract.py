from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any

from pydantic import ValidationError

from syspilot_contract.checker import check_contract
from syspilot_contract.schemas import (
    ContractVerificationResult,
    LibraryContract,
    SourceArtifact,
)
from syspilot_manual.llm_client import ManualLLMProtocol, extract_json_object
from syspilot_manual.schemas import EvidenceItem


_HOST_SYMBOL_ALIASES = {
    "host.physical_cores": "host.physical_cores",
    "physical cores": "host.physical_cores",
    "physical core count": "host.physical_cores",
    "host.visible_cpus": "host.visible_cpus",
    "visible cpus": "host.visible_cpus",
    "visible cpu count": "host.visible_cpus",
    "visible processors": "host.visible_cpus",
    "visible processor count": "host.visible_cpus",
    "available cpus": "host.visible_cpus",
    "host.numa_nodes": "host.numa_nodes",
    "numa nodes": "host.numa_nodes",
    "available numa nodes": "host.numa_nodes",
}

_CONTROL_TYPE_ALIASES = {
    "int": "int",
    "integer": "int",
    "bool": "bool",
    "boolean": "bool",
    "enum": "enum",
    "string": "string",
    "cpu_set": "cpu_set",
    "cpuset": "cpu_set",
    "numa_node_set": "numa_node_set",
    "numa nodes": "numa_node_set",
    "thread_count": "int",
    "parallelism_cap": "int",
    "nested_parallelism_cap": "int",
    "wait_policy": "enum",
    "affinity_policy": "enum",
    "placement_set": "string",
    "runtime_specific_affinity": "enum",
    "hardware_subset": "string",
    "observability": "bool",
    "observability_format": "string",
    "numa_node_cpu_binding": "numa_node_set",
    "physical_cpu_binding": "cpu_set",
    "strict_memory_binding": "numa_node_set",
    "preferred_memory_policy": "numa_node_set",
    "local_memory_preference": "bool",
    "interleaved_memory_policy": "numa_node_set",
}

_KNOWN_CONTROL_TYPES = {
    "OMP_NUM_THREADS": "int",
    "OMP_THREAD_LIMIT": "int",
    "OMP_MAX_ACTIVE_LEVELS": "int",
    "OMP_PLACES": "string",
    "OMP_PROC_BIND": "enum",
    "GOMP_CPU_AFFINITY": "cpu_set",
    "KMP_AFFINITY": "enum",
    "KMP_HW_SUBSET": "string",
    "KMP_BLOCKTIME": "int",
    "OMP_DISPLAY_ENV": "enum",
    "OMP_DISPLAY_AFFINITY": "bool",
    "OMP_AFFINITY_FORMAT": "string",
    "KMP_SETTINGS": "bool",
    "--cpunodebind": "numa_node_set",
    "--physcpubind": "cpu_set",
    "--membind": "numa_node_set",
    "--preferred": "numa_node_set",
    "--localalloc": "bool",
    "--interleave": "numa_node_set",
}

_KNOWN_CONTROL_ALLOWED_VALUES = {
    "OMP_PROC_BIND": ["false", "true", "master", "close", "spread", "primary"],
    "OMP_DISPLAY_ENV": ["false", "true", "verbose"],
}

_RECIPE_KIND_ALIASES = {
    "shared_cpu_thread_conflict": "shared_resource_conflict",
    "shared_resource_conflict": "shared_resource_conflict",
    "baseline_validation": "baseline_validation",
    "parameter_sweep": "parameter_sweep",
    "validation": "validation",
    "starting_point": "starting_point",
    "comparison": "parameter_sweep",
}


def _clean_id_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _normalize_host_symbol(value: object) -> object:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    alias = _HOST_SYMBOL_ALIASES.get(text.lower())
    return alias or text


def _normalize_activation_value(raw_value: object, control_type: str) -> str | int | bool | None:
    if raw_value is None:
        return None
    if control_type == "int":
        text = str(raw_value).strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return text
    if control_type == "bool":
        text = str(raw_value).strip().lower()
        if text in {"1", "true", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disabled"}:
            return False
        return str(raw_value).strip()
    return str(raw_value).strip()


def _canonical_probe(tool_value: object) -> dict[str, str]:
    text = str(tool_value).strip()
    target = text.split("=", 1)[0].strip() if "=" in text else text
    if text.startswith("/proc/"):
        return {"kind": "procfs", "target": text}
    if re.fullmatch(r"[A-Z0-9_]+=?.*", text) and "/" not in text and not text.startswith("--"):
        return {"kind": "runtime_env", "target": target}
    return {"kind": "command", "target": text}


def _infer_control_refs(actions: list[str], controls: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for control in controls:
        control_id = str(control.get("id", "")).strip()
        control_name = str(control.get("name", "")).strip()
        if not control_id or not control_name:
            continue
        if any(control_name in action for action in actions):
            refs.append(control_id)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in refs:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _sanitize_contract_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Contract synthesis returned a non-object payload")
    sanitized = deepcopy(payload)

    def _normalize_source_ref_items(section: str) -> None:
        raw_items = sanitized.get(section)
        if not isinstance(raw_items, list):
            return
        normalized_items: list[object] = []
        for item in raw_items:
            if not isinstance(item, dict):
                normalized_items.append(item)
                continue
            filtered_item = dict(item)
            filtered_item["source_refs"] = _clean_id_list(item.get("source_refs"))
            normalized_items.append(filtered_item)
        sanitized[section] = normalized_items

    for section in ("uses", "assumptions", "signals"):
        _normalize_source_ref_items(section)

    raw_uses = sanitized.get("uses")
    if isinstance(raw_uses, list):
        for item in raw_uses:
            if not isinstance(item, dict):
                continue
            demand = item.get("demand")
            if not isinstance(demand, dict):
                continue
            if "lower" in demand:
                demand["lower"] = _normalize_host_symbol(demand["lower"])
            if "upper" in demand:
                demand["upper"] = _normalize_host_symbol(demand["upper"])

    raw_assumptions = sanitized.get("assumptions")
    if isinstance(raw_assumptions, list):
        for item in raw_assumptions:
            if not isinstance(item, dict):
                continue
            predicate = item.get("predicate")
            if isinstance(predicate, dict) and "value" in predicate:
                predicate["value"] = _normalize_host_symbol(predicate["value"])

    raw_controls = sanitized.get("controls")
    if isinstance(raw_controls, list):
        filtered_controls: list[object] = []
        for item in raw_controls:
            if not isinstance(item, dict):
                filtered_controls.append(item)
                continue
            affects = _clean_id_list(item.get("affects"))
            source_refs = _clean_id_list(item.get("source_refs"))
            raw_name = str(item.get("name", "")).strip()
            activation_value = item.get("activation_value")
            if "=" in raw_name and activation_value is None:
                raw_name, activation_value = raw_name.split("=", 1)
                raw_name = raw_name.strip()
                activation_value = str(activation_value).strip()
            raw_type = str(item.get("type", "")).strip().lower()
            control_type = (
                _KNOWN_CONTROL_TYPES.get(raw_name)
                or _CONTROL_TYPE_ALIASES.get(raw_type)
                or ("enum" if activation_value is not None else "string")
            )
            filtered_item = dict(item)
            filtered_item["name"] = raw_name
            filtered_item["type"] = control_type
            filtered_item["affects"] = affects
            filtered_item["allowed_values"] = _clean_id_list(item.get("allowed_values"))
            if not filtered_item["allowed_values"] and raw_name in _KNOWN_CONTROL_ALLOWED_VALUES:
                filtered_item["allowed_values"] = _KNOWN_CONTROL_ALLOWED_VALUES[raw_name]
            filtered_item["activation_value"] = _normalize_activation_value(
                activation_value,
                control_type,
            )
            filtered_item["source_refs"] = source_refs
            filtered_controls.append(filtered_item)
        sanitized["controls"] = filtered_controls

    raw_signals = sanitized.get("signals")
    if isinstance(raw_signals, list):
        filtered_signals: list[object] = []
        for item in raw_signals:
            if not isinstance(item, dict):
                filtered_signals.append(item)
                continue
            source_refs = _clean_id_list(item.get("source_refs"))
            filtered_item = dict(item)
            probe = filtered_item.get("probe")
            if not isinstance(probe, dict):
                probe = _canonical_probe(filtered_item.pop("tool", ""))
            filtered_item["probe"] = {
                "kind": str(probe.get("kind", "command")).strip(),
                "target": str(probe.get("target", "")).strip(),
            }
            filtered_item["source_refs"] = source_refs
            filtered_signals.append(filtered_item)
        sanitized["signals"] = filtered_signals

    raw_recipes = sanitized.get("recipes")
    if isinstance(raw_recipes, list):
        filtered_recipes: list[object] = []
        controls_payload = sanitized.get("controls")
        controls: list[dict[str, Any]] = (
            [item for item in controls_payload if isinstance(item, dict)]
            if isinstance(controls_payload, list)
            else []
        )
        for item in raw_recipes:
            if not isinstance(item, dict):
                filtered_recipes.append(item)
                continue
            collect = _clean_id_list(item.get("collect"))
            actions = _clean_id_list(item.get("actions"))
            source_refs = _clean_id_list(item.get("source_refs"))
            hypothesis_kind = _RECIPE_KIND_ALIASES.get(
                str(item.get("hypothesis_kind", "")).strip().lower(),
                "validation",
            )
            filtered_item = dict(item)
            filtered_item["hypothesis_kind"] = hypothesis_kind
            filtered_item["collect"] = collect
            control_refs = _clean_id_list(item.get("control_refs"))
            filtered_item["control_refs"] = control_refs or _infer_control_refs(actions, controls)
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
        "source_artifacts",
        "uses",
        "assumptions",
        "controls",
        "signals",
        "recipes",
        "contract",
        "contract_type",
        "version",
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
            "version": "lpcl/v0",
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
    parsed = extract_json_object(response_text)
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
        "host_symbol_vocab": [
            "host.physical_cores",
            "host.visible_cpus",
            "host.numa_nodes",
        ],
        "control_type_vocab": ["int", "bool", "enum", "string", "cpu_set", "numa_node_set"],
        "probe_kind_vocab": ["command", "procfs", "runtime_env"],
        "recipe_hypothesis_vocab": [
            "baseline_validation",
            "parameter_sweep",
            "validation",
            "starting_point",
            "shared_resource_conflict",
        ],
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
    current_payload = extract_json_object(response_text)

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
