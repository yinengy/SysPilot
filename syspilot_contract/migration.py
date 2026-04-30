from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from syspilot_contract.schemas import CONTRACT_VERSION, RESOURCE_TO_HYPOTHESIS_KIND


LEGACY_TO_V1_RESOURCE: dict[str, str] = {
    "cpu.thread": "cpu.compute",
    "worker.pool": "cpu.compute",
    "cpu.affinity": "cpu.placement",
    "gpu.device": "gpu.compute",
    "gpu.stream": "gpu.compute",
    "mem.capacity": "mem.capacity",
    "mem.bandwidth": "mem.bandwidth",
    "mem.allocator": "mem.capacity",
    "io.bandwidth": "io.bandwidth",
    "io.queue": "io.queue",
    "net.bandwidth": "io.bandwidth",
    "cache.capacity": "mem.capacity",
    "cpu.core": "cpu.placement",
}

LEGACY_SHARING_TO_V1: dict[str, str] = {
    "shareable": "shared",
    "partitionable": "partitioned",
    "exclusive": "exclusive",
    "global": "process_global",
}

LEGACY_CONTROL_KIND_TO_V1: dict[str, str] = {
    "env": "env",
    "api": "api",
    "config": "config",
    "cli": "cli",
    "runtime": "api",
}

LEGACY_CONTROL_TYPE_TO_V1: dict[str, str] = {
    "int": "int",
    "integer": "int",
    "float": "float",
    "bool": "bool",
    "boolean": "bool",
    "string": "string",
    "enum": "enum",
    "cpu_set": "set",
    "cpuset": "set",
    "numa_node_set": "set",
    "set": "set",
}

LEGACY_PREDICATE_TO_V1: dict[str, str] = {
    "max_competitors": "competitors_at_most",
    "budget_at_most": "total_at_most",
    "dedicated_partition": "exclusive",
    "affinity_compatible": "placement_compatible",
    "not_nested": "not_nested",
}


def _clean_strings(values: Sequence[object]) -> list[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _condition_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = _clean_strings(value)
        return "; ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _evidence(value: object) -> list[str]:
    if isinstance(value, list):
        cleaned = _clean_strings(value)
        if cleaned:
            return cleaned
    raise ValueError("Legacy LPCL item is missing source_refs/evidence")


def _resource_dimension(resource: str) -> str:
    return {
        "cpu.compute": "workers",
        "cpu.placement": "core_set",
        "gpu.compute": "devices",
        "gpu.memory": "bytes",
        "mem.capacity": "bytes",
        "mem.bandwidth": "intensity",
        "io.bandwidth": "bandwidth",
        "io.queue": "depth",
    }[resource]


def _map_legacy_resource(resource: object) -> str:
    text = str(resource).strip()
    mapped = LEGACY_TO_V1_RESOURCE.get(text)
    if mapped is None:
        raise ValueError(
            f"Unsupported legacy resource {text}. Migrate it explicitly before validation."
        )
    return mapped


def _component_kind_for_legacy_resource(resource: str) -> str:
    if resource in {"cpu.thread", "worker.pool", "cpu.affinity", "cpu.core", "gpu.device"}:
        return "executor"
    if resource in {"gpu.stream", "io.queue"}:
        return "queue"
    if resource == "mem.allocator":
        return "allocator"
    if resource == "cache.capacity":
        return "cache"
    return "backend"


def _component_id_from_use(use_id: str) -> str:
    text = use_id.strip()
    for prefix in ("use_", "u_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text or "component"


def _normalize_bound(value: object, default: object) -> object:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    return text or default


def _migrate_demand(legacy_resource: str, resource: str, legacy_demand: object) -> dict[str, Any]:
    demand = legacy_demand if isinstance(legacy_demand, dict) else {}
    kind = str(demand.get("kind", "unknown")).strip()

    if resource == "cpu.compute":
        if kind == "fixed":
            value = _normalize_bound(demand.get("value"), 1)
            return {"dimension": "workers", "min": value, "max": value}
        if kind in {"range", "scaled_by"}:
            return {
                "dimension": "workers",
                "min": _normalize_bound(demand.get("lower"), 1),
                "max": _normalize_bound(demand.get("upper"), "unknown"),
            }
        return {"dimension": "workers", "min": 1, "max": "unknown"}

    if resource == "cpu.placement":
        source = demand.get("source")
        if not source:
            source = "runtime_affinity_mask"
        return {"dimension": "core_set", "source": str(source).strip() or "runtime_affinity_mask"}

    if resource == "gpu.compute":
        if kind == "fixed":
            value = _normalize_bound(demand.get("value"), 1)
            return {"dimension": "devices", "min": value, "max": value}
        if kind in {"range", "scaled_by"}:
            return {
                "dimension": "devices",
                "min": _normalize_bound(demand.get("lower"), 1),
                "max": _normalize_bound(demand.get("upper"), "host.gpu_devices"),
            }
        if legacy_resource == "gpu.stream":
            return {"dimension": "partition", "source": "runtime_queue_partition"}
        return {"dimension": "devices", "min": 1, "max": "host.gpu_devices"}

    if resource in {"gpu.memory", "mem.capacity"}:
        if kind == "fixed":
            value = _normalize_bound(demand.get("value"), 0)
            return {"dimension": "bytes", "min": value, "max": value}
        if kind in {"range", "scaled_by"}:
            return {
                "dimension": "bytes",
                "min": _normalize_bound(demand.get("lower"), 0),
                "max": _normalize_bound(demand.get("upper"), "unknown"),
            }
        return {"dimension": "bytes", "min": 0, "max": "unknown"}

    if resource == "mem.bandwidth":
        if kind == "proportional_to":
            return {
                "dimension": "intensity",
                "value": _normalize_bound(demand.get("metric"), "unknown"),
            }
        return {"dimension": "intensity", "value": "unknown"}

    if resource == "io.bandwidth":
        if kind == "fixed":
            value = _normalize_bound(demand.get("value"), 0)
            return {"dimension": "bandwidth", "min": value, "max": value}
        if kind in {"range", "scaled_by"}:
            return {
                "dimension": "bandwidth",
                "min": _normalize_bound(demand.get("lower"), 0),
                "max": _normalize_bound(demand.get("upper"), "unknown"),
            }
        return {"dimension": "bandwidth", "min": 0, "max": "unknown"}

    if resource == "io.queue":
        if kind == "fixed":
            value = _normalize_bound(demand.get("value"), 1)
            return {"dimension": "depth", "min": value, "max": value}
        if kind in {"range", "scaled_by"}:
            return {
                "dimension": "depth",
                "min": _normalize_bound(demand.get("lower"), 1),
                "max": _normalize_bound(demand.get("upper"), "unknown"),
            }
        return {"dimension": "depth", "min": 1, "max": "unknown"}

    raise ValueError(f"Unsupported migrated resource {resource}")


def _component_attrs(kind: str, legacy_resource: str, demand: dict[str, Any]) -> dict[str, Any]:
    if kind == "executor":
        attrs: dict[str, Any] = {
            "executor_kind": (
                "thread_pool"
                if legacy_resource in {"cpu.thread", "worker.pool"}
                else "runtime_executor"
            )
        }
        if demand.get("dimension") == "workers":
            attrs["workers"] = {"min": demand.get("min"), "max": demand.get("max")}
        return attrs
    if kind == "queue":
        return {"queue_kind": "runtime_queue"}
    if kind == "allocator":
        return {"memory_space": "host"}
    if kind == "backend":
        return {"backend_name": legacy_resource.replace(".", "_"), "backend_type": "runtime"}
    if kind == "cache":
        return {"capacity": "unknown"}
    return {}


def _infer_probe_kind(name: str, legacy_kind: str) -> str:
    probe_name = name.strip()
    if legacy_kind == "procfs":
        return "os"
    if legacy_kind == "runtime_env":
        return "log"
    if probe_name.startswith("perf") or probe_name == "perf_stat":
        return "perf"
    if probe_name.startswith("omp_") or "get_" in probe_name or probe_name.startswith("mkl_"):
        return "api"
    return "log"


def _effect_payload(legacy_op: str, activation_value: object) -> dict[str, Any]:
    if activation_value is not None:
        return {"kind": "set", "value": activation_value}
    if legacy_op == "increase":
        return {"kind": "increase"}
    if legacy_op == "decrease":
        return {"kind": "decrease"}
    if legacy_op == "report":
        return {"kind": "enable"}
    return {"kind": "set", "value": "input"}


def _infer_recipe_resource(
    recipe: dict[str, Any],
    signal_resource_by_id: dict[str, str],
    control_resources_by_id: dict[str, list[str]],
) -> str:
    collect = _clean_strings(recipe.get("collect", []))
    for signal_id in collect:
        resource = signal_resource_by_id.get(signal_id)
        if resource:
            return resource
    for control_id in _clean_strings(recipe.get("control_refs", [])):
        resources = control_resources_by_id.get(control_id, [])
        if resources:
            return resources[0]
    return "cpu.compute"


def _migrate_library_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(payload)
    legacy_uses = migrated.get("uses", [])
    components: list[dict[str, Any]] = []
    uses: list[dict[str, Any]] = []
    use_id_to_component: dict[str, str] = {}
    use_id_to_resource: dict[str, str] = {}
    use_id_to_use: dict[str, dict[str, Any]] = {}
    used_component_ids: set[str] = set()

    if isinstance(legacy_uses, list):
        for item in legacy_uses:
            if not isinstance(item, dict):
                continue
            use_id = str(item.get("id", "")).strip()
            if not use_id:
                continue
            legacy_resource = str(item.get("resource", "")).strip()
            resource = _map_legacy_resource(legacy_resource)
            component_base = _component_id_from_use(use_id)
            component_id = component_base
            suffix = 2
            while component_id in used_component_ids:
                component_id = f"{component_base}_{suffix}"
                suffix += 1
            used_component_ids.add(component_id)
            demand = _migrate_demand(legacy_resource, resource, item.get("demand", {}))
            component = {
                "id": component_id,
                "kind": _component_kind_for_legacy_resource(legacy_resource),
                "attrs": _component_attrs(
                    _component_kind_for_legacy_resource(legacy_resource),
                    legacy_resource,
                    demand,
                ),
                "evidence": _evidence(item.get("source_refs") or item.get("evidence") or []),
            }
            components.append(component)
            migrated_use = {
                "id": use_id,
                "resource": resource,
                "by": component_id,
                "phase": item.get("phase", "unknown"),
                "scope": item.get("scope", "library"),
                "demand": demand,
                "share": LEGACY_SHARING_TO_V1.get(str(item.get("sharing", "shareable")), "shared"),
                "when": _condition_string(item.get("when")),
                "evidence": _evidence(item.get("source_refs") or item.get("evidence") or []),
            }
            uses.append(migrated_use)
            use_id_to_component[use_id] = component_id
            use_id_to_resource[use_id] = resource
            use_id_to_use[use_id] = migrated_use

    assumptions: list[dict[str, Any]] = []
    if isinstance(migrated.get("assumptions"), list):
        for item in migrated["assumptions"]:
            if not isinstance(item, dict):
                continue
            predicate = item.get("predicate", {}) if isinstance(item.get("predicate"), dict) else {}
            legacy_resource = predicate.get("resource") or item.get("resource")
            resource = _map_legacy_resource(legacy_resource)
            legacy_kind = str(predicate.get("kind", "")).strip()
            new_kind = LEGACY_PREDICATE_TO_V1.get(legacy_kind)
            if new_kind is None:
                raise ValueError(f"Unsupported legacy predicate kind {legacy_kind}")
            new_predicate: dict[str, Any] = {"kind": new_kind}
            if new_kind in {"total_at_most", "competitors_at_most"}:
                new_predicate["value"] = predicate.get("value")
            when_parts = _clean_strings(
                [
                    _condition_string(item.get("when")) or "",
                    (
                        f"phase == {predicate.get('phase')}"
                        if predicate.get("phase") and predicate.get("phase") != "unknown"
                        else ""
                    ),
                ]
            )
            assumptions.append(
                {
                    "id": item.get("id"),
                    "resource": resource,
                    "predicate": new_predicate,
                    "when": "; ".join(when_parts) if when_parts else None,
                    "evidence": _evidence(item.get("source_refs") or item.get("evidence") or []),
                }
            )

    controls: list[dict[str, Any]] = []
    control_resources_by_id: dict[str, list[str]] = {}
    if isinstance(migrated.get("controls"), list):
        for item in migrated["controls"]:
            if not isinstance(item, dict):
                continue
            control_id = str(item.get("id", "")).strip()
            if not control_id:
                continue
            legacy_effect = item.get("effect", {}) if isinstance(item.get("effect"), dict) else {}
            legacy_effect_resource = legacy_effect.get("resource")
            resource = (
                _map_legacy_resource(legacy_effect_resource) if legacy_effect_resource else None
            )
            affects = _clean_strings(item.get("affects", []))
            target_use_id = next(
                (affect for affect in affects if use_id_to_resource.get(affect) == resource),
                affects[0] if affects else "",
            )
            target_component = use_id_to_component.get(target_use_id)
            target_use = use_id_to_use.get(target_use_id)
            name = str(item.get("name", "")).strip()
            upper_name = name.upper()
            if target_component and (
                upper_name.endswith("NUM_THREADS")
                or "NUM_THREADS" in upper_name
                or upper_name.endswith("THREAD_LIMIT")
                or name in {"-nt", "-ntmpi", "-ntomp", "-ntomp_pme", "-npme"}
                or name.lower().endswith("nthread")
            ):
                target = f"{target_component}.workers"
            elif target_component and ("TIMEOUT" in upper_name or "BLOCKTIME" in upper_name):
                target = f"{target_component}.idle_timeout"
            elif target_component and "WAIT_POLICY" in upper_name:
                target = f"{target_component}.wait_policy"
            elif resource == "cpu.placement" and target_use is not None:
                target = target_use["id"]
            elif resource and target_use is not None:
                target = target_use["id"]
            elif target_component:
                target = target_component
            else:
                target = resource or "cpu.compute"
            effect = _effect_payload(
                str(legacy_effect.get("op", "set")).strip(), item.get("activation_value")
            )
            controls.append(
                {
                    "id": control_id,
                    "name": name,
                    "kind": LEGACY_CONTROL_KIND_TO_V1.get(
                        str(item.get("kind", "env")).strip(), "env"
                    ),
                    "type": LEGACY_CONTROL_TYPE_TO_V1.get(
                        str(item.get("type", "string")).strip().lower(), "string"
                    ),
                    "target": target,
                    "effect": effect,
                    "when": _condition_string(item.get("when")),
                    "evidence": _evidence(item.get("source_refs") or item.get("evidence") or []),
                }
            )
            if resource is not None:
                control_resources_by_id[control_id] = [resource]
            else:
                control_resources_by_id[control_id] = []

    signals: list[dict[str, Any]] = []
    signal_resource_by_id: dict[str, str] = {}
    if isinstance(migrated.get("signals"), list):
        for item in migrated["signals"]:
            if not isinstance(item, dict):
                continue
            signal_id = str(item.get("id", "")).strip()
            if not signal_id:
                continue
            legacy_resource = str(item.get("resource", "")).strip()
            resource = _map_legacy_resource(legacy_resource)
            metric = str(item.get("metric", "")).strip()
            raw_probe = item.get("probe")
            probe: dict[str, Any] = raw_probe if isinstance(raw_probe, dict) else {}
            legacy_probe_kind = str(probe.get("kind", "command")).strip() or "command"
            legacy_probe_name = str(probe.get("target", item.get("tool", ""))).strip()
            compute_component = next(
                (
                    component_id
                    for use_id, component_id in use_id_to_component.items()
                    if use_id_to_resource.get(use_id) == resource
                ),
                None,
            )
            if resource == "cpu.compute" and compute_component and "thread" in metric.lower():
                target = f"{compute_component}.workers"
            elif resource in {
                "cpu.compute",
                "cpu.placement",
                "mem.bandwidth",
                "io.bandwidth",
                "io.queue",
            }:
                target = resource
            elif compute_component is not None:
                target = compute_component
            else:
                target = resource
            signals.append(
                {
                    "id": signal_id,
                    "target": target,
                    "metric": metric,
                    "indicates": item.get("indicates"),
                    "probe": {
                        "kind": _infer_probe_kind(legacy_probe_name, legacy_probe_kind),
                        "name": legacy_probe_name,
                    },
                    "when": _condition_string(item.get("when")),
                    "evidence": _evidence(item.get("source_refs") or item.get("evidence") or []),
                }
            )
            signal_resource_by_id[signal_id] = resource

    recipes: list[dict[str, Any]] = []
    if isinstance(migrated.get("recipes"), list):
        for item in migrated["recipes"]:
            if not isinstance(item, dict):
                continue
            recipe_id = str(item.get("id", "")).strip()
            if not recipe_id:
                continue
            resource = _infer_recipe_resource(item, signal_resource_by_id, control_resources_by_id)
            recipes.append(
                {
                    "id": recipe_id,
                    "for": RESOURCE_TO_HYPOTHESIS_KIND[resource],
                    "collect": _clean_strings(item.get("collect", [])),
                    "actions": _clean_strings(item.get("actions", [])),
                    "accept": item.get("accept_if")
                    or item.get("accept")
                    or "no acceptance condition provided",
                    "rollback": item.get("rollback_if")
                    or item.get("rollback")
                    or "no rollback condition provided",
                    "when": _condition_string(item.get("when")),
                    "evidence": _evidence(item.get("source_refs") or item.get("evidence") or []),
                }
            )

    return {
        "version": CONTRACT_VERSION,
        "contract": migrated.get("contract"),
        "type": "library",
        "components": components,
        "uses": uses,
        "assumptions": assumptions,
        "controls": controls,
        "signals": signals,
        "recipes": recipes,
        "source_artifacts": migrated.get("source_artifacts", []),
    }


def _component_refs_from_use_refs(use_refs: list[str]) -> list[str]:
    component_refs: list[str] = []
    for ref in use_refs:
        if "." not in ref:
            continue
        contract_name, use_id = ref.split(".", 1)
        if use_id.startswith("use_"):
            use_id = use_id[len("use_") :]
        elif use_id.startswith("u_"):
            use_id = use_id[len("u_") :]
        component_refs.append(f"{contract_name}.{use_id}")
    return _clean_strings(component_refs)


def _migrate_composed_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selected_contracts = _clean_strings(payload.get("selected_contracts", []))
    contract_instances = payload.get("contract_instances", [])
    shared_resources: list[dict[str, Any]] = []
    if isinstance(payload.get("shared_resources"), list):
        for item in payload["shared_resources"]:
            if not isinstance(item, dict):
                continue
            use_refs = _clean_strings(item.get("use_refs", item.get("uses", [])))
            shared_resources.append(
                {
                    "resource": _map_legacy_resource(item.get("resource")),
                    "phase": item.get("phase", "unknown"),
                    "contracts": _clean_strings(item.get("libraries", item.get("contracts", []))),
                    "components": _component_refs_from_use_refs(use_refs),
                    "uses": use_refs,
                    "evidence": _clean_strings(item.get("source_refs", item.get("evidence", []))),
                }
            )

    hypotheses: list[dict[str, Any]] = []
    if isinstance(payload.get("hypotheses"), list):
        for item in payload["hypotheses"]:
            if not isinstance(item, dict):
                continue
            basis = item.get("basis", {}) if isinstance(item.get("basis"), dict) else {}
            overlaps_payload = (
                basis.get("overlaps", []) if isinstance(basis.get("overlaps"), list) else []
            )
            overlap_use_refs = _clean_strings(
                [
                    ref
                    for overlap in overlaps_payload
                    if isinstance(overlap, dict)
                    for ref in _clean_strings(overlap.get("uses", []))
                ]
            )
            resource = _map_legacy_resource(item.get("resource"))
            hypotheses.append(
                {
                    "id": item.get("id"),
                    "kind": RESOURCE_TO_HYPOTHESIS_KIND[resource],
                    "resource": resource,
                    "phase": item.get("phase", "unknown"),
                    "contracts": _clean_strings(item.get("libraries", item.get("contracts", []))),
                    "components": _component_refs_from_use_refs(overlap_use_refs),
                    "uses": overlap_use_refs,
                    "basis": {
                        "overlaps": [
                            {
                                "uses": _clean_strings(overlap.get("uses", [])),
                                "components": _component_refs_from_use_refs(
                                    _clean_strings(overlap.get("uses", []))
                                ),
                                "contracts": _clean_strings(
                                    item.get("libraries", item.get("contracts", []))
                                ),
                            }
                            for overlap in overlaps_payload
                            if isinstance(overlap, dict)
                        ],
                        "assumption_violations": [
                            {
                                "assumption": ref,
                                "conflicting_uses": overlap_use_refs,
                            }
                            for ref in _clean_strings(
                                basis.get(
                                    "violated_assumptions", basis.get("assumption_violations", [])
                                )
                            )
                        ],
                        "control_collisions": (
                            [
                                {
                                    "resource": resource,
                                    "dimension": _resource_dimension(resource),
                                    "controls": _clean_strings(
                                        basis.get(
                                            "colliding_controls",
                                            basis.get("control_collisions", []),
                                        )
                                    ),
                                }
                            ]
                            if _clean_strings(
                                basis.get("colliding_controls", basis.get("control_collisions", []))
                            )
                            else []
                        ),
                    },
                    "validate": {
                        "collect": _clean_strings(
                            (
                                (item.get("validate") or item.get("validation") or {})
                                if isinstance(
                                    (item.get("validate") or item.get("validation") or {}), dict
                                )
                                else {}
                            ).get("collect", [])
                        )
                    },
                    "candidate_actions": _clean_strings(
                        item.get("candidate_control_refs", item.get("candidate_actions", []))
                    ),
                    "recipes": _clean_strings(
                        item.get("candidate_recipe_refs", item.get("recipes", []))
                    ),
                    "evidence": _clean_strings(item.get("source_refs", item.get("evidence", []))),
                }
            )

    return {
        "version": CONTRACT_VERSION,
        "contract": payload.get("contract"),
        "type": "composed",
        "selected_contracts": selected_contracts,
        "contract_instances": contract_instances,
        "shared_resources": shared_resources,
        "hypotheses": hypotheses,
    }


def migrate_contract_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("contract document must be a mapping")
    normalized = deepcopy(payload)
    if str(normalized.get("version", "")).strip() == CONTRACT_VERSION and "type" in normalized:
        return normalized
    contract_type = str(
        normalized.get("type") or normalized.get("contract_type") or "library"
    ).strip()
    if contract_type == "composed":
        return _migrate_composed_contract_payload(normalized)
    return _migrate_library_contract_payload(normalized)
