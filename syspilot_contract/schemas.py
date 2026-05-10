from __future__ import annotations

import re
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONTRACT_VERSION = "lpcl/v1"

RESOURCE_TYPES: tuple[str, ...] = (
    "cpu.compute",
    "cpu.placement",
    "gpu.compute",
    "gpu.memory",
    "mem.capacity",
    "mem.bandwidth",
    "io.bandwidth",
    "io.queue",
)

COMPONENT_KINDS: tuple[str, ...] = (
    "executor",
    "queue",
    "allocator",
    "backend",
    "cache",
)

PHASES: tuple[str, ...] = (
    "init",
    "compute",
    "communication",
    "flush",
    "background",
    "teardown",
    "unknown",
)

SCOPES: tuple[str, ...] = (
    "thread",
    "process",
    "library",
    "device",
    "system",
)

SHARE_MODES: tuple[str, ...] = (
    "shared",
    "exclusive",
    "partitioned",
    "process_global",
)

PREDICATE_KINDS: tuple[str, ...] = (
    "total_at_most",
    "competitors_at_most",
    "exclusive",
    "not_nested",
    "placement_compatible",
)

CONTROL_KINDS: tuple[str, ...] = (
    "env",
    "api",
    "config",
    "cli",
    "vlc",
)

CONTROL_TYPES: tuple[str, ...] = (
    "int",
    "float",
    "bool",
    "string",
    "enum",
    "set",
)

EFFECT_KINDS: tuple[str, ...] = (
    "set",
    "increase",
    "decrease",
    "partition",
    "disable",
    "enable",
)

SIGNAL_ROLES: tuple[str, ...] = (
    "config_check",
    "resource_signal",
    "subcomponent_signal",
    "outcome",
    "guardrail",
)

EXPECTED_DIRECTIONS: tuple[str, ...] = (
    "increase",
    "decrease",
    "stable",
    "unknown",
)

HYPOTHESIS_KINDS: tuple[str, ...] = (
    "shared_cpu_compute_conflict",
    "cpu_placement_conflict",
    "shared_gpu_compute_conflict",
    "gpu_memory_pressure",
    "host_memory_capacity_pressure",
    "memory_bandwidth_contention",
    "io_bandwidth_contention",
    "io_queue_contention",
)

PHASE_OR_MIXED: tuple[str, ...] = PHASES + ("mixed",)

RESOURCE_TO_HYPOTHESIS_KIND: dict[str, str] = {
    "cpu.compute": "shared_cpu_compute_conflict",
    "cpu.placement": "cpu_placement_conflict",
    "gpu.compute": "shared_gpu_compute_conflict",
    "gpu.memory": "gpu_memory_pressure",
    "mem.capacity": "host_memory_capacity_pressure",
    "mem.bandwidth": "memory_bandwidth_contention",
    "io.bandwidth": "io_bandwidth_contention",
    "io.queue": "io_queue_contention",
}

LEGACY_RESOURCE_ERRORS: dict[str, str] = {
    "worker.pool": "Invalid resource worker.pool. worker.pool is a component, not a resource.",
    "cpu.thread": "Invalid resource cpu.thread. Use resource cpu.compute with demand.dimension = workers.",
    "cpu.affinity": "Invalid resource cpu.affinity. Use resource cpu.placement.",
    "gpu.device": "Invalid resource gpu.device. Use resource gpu.compute.",
    "gpu.stream": "Invalid resource gpu.stream. Model GPU streams as queue components using gpu.compute.",
    "mem.allocator": "Invalid resource mem.allocator. mem.allocator is a component, not a resource.",
    "net.bandwidth": "Invalid resource net.bandwidth. Use resource io.bandwidth.",
    "cache.capacity": "Invalid resource cache.capacity. cache is a component; use mem.capacity for host memory pressure.",
    "cpu.core": "Invalid resource cpu.core. Use cpu.compute or cpu.placement depending on whether the claim is execution or placement related.",
}

EXECUTOR_WORKER_CONTROL_NAMES: set[str] = {
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "openblas_set_num_threads",
    "goto_set_num_threads",
    "mkl_set_num_threads",
    "mkl_domain_set_num_threads",
    "mkl_set_num_threads_local",
    "-nt",
    "-ntmpi",
    "-ntomp",
    "-ntomp_pme",
    "-npme",
    "nthread",
}

TARGET_RESERVED_PREFIXES: tuple[str, ...] = (
    "components",
    "uses",
    "assumptions",
    "controls",
    "signals",
    "recipes",
)

ResourceType = Literal[
    "cpu.compute",
    "cpu.placement",
    "gpu.compute",
    "gpu.memory",
    "mem.capacity",
    "mem.bandwidth",
    "io.bandwidth",
    "io.queue",
]

ComponentKind = Literal["executor", "queue", "allocator", "backend", "cache"]
PhaseName = Literal[
    "init", "compute", "communication", "flush", "background", "teardown", "unknown"
]
ScopeName = Literal["thread", "process", "library", "device", "system"]
ShareMode = Literal["shared", "exclusive", "partitioned", "process_global"]
PredicateKind = Literal[
    "total_at_most",
    "competitors_at_most",
    "exclusive",
    "not_nested",
    "placement_compatible",
]
ControlKind = Literal["env", "api", "config", "cli", "vlc"]
ControlType = Literal["int", "float", "bool", "string", "enum", "set"]
EffectKind = Literal["set", "increase", "decrease", "partition", "disable", "enable"]
SignalRole = Literal[
    "config_check", "resource_signal", "subcomponent_signal", "outcome", "guardrail"
]
ExpectedDirection = Literal["increase", "decrease", "stable", "unknown"]
HypothesisKind = Literal[
    "shared_cpu_compute_conflict",
    "cpu_placement_conflict",
    "shared_gpu_compute_conflict",
    "gpu_memory_pressure",
    "host_memory_capacity_pressure",
    "memory_bandwidth_contention",
    "io_bandwidth_contention",
    "io_queue_contention",
]


def _clean_strings(values: list[str]) -> list[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_condition(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = _clean_strings([str(item) for item in value])
        return "; ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _require_non_empty_evidence(evidence: list[str], label: str) -> list[str]:
    cleaned = _clean_strings(evidence)
    if not cleaned:
        raise ValueError(f"{label} must include non-empty evidence")
    return cleaned


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_symbol(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_number_or_symbol(value: object) -> bool:
    return _is_number(value) or _is_symbol(value)


def _validate_resource(resource: str) -> str:
    if resource in RESOURCE_TYPES:
        return resource
    if resource in LEGACY_RESOURCE_ERRORS:
        raise ValueError(LEGACY_RESOURCE_ERRORS[resource])
    raise ValueError(f"Unknown resource {resource}. Allowed resources: {', '.join(RESOURCE_TYPES)}")


def _validate_enum(value: str, allowed: tuple[str, ...], label: str) -> str:
    text = str(value).strip()
    if text not in allowed:
        raise ValueError(f"Unknown {label} {text}. Allowed values: {', '.join(allowed)}")
    return text


def _normalize_probe_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".")


def _validate_metric_name(metric: str) -> str:
    text = str(metric).strip()
    if not text:
        raise ValueError("Signal metric must not be empty")
    normalized = _normalize_probe_key(text)
    vague = {
        "metric",
        "value",
        "usage",
        "performance",
        "runtime",
        "memory",
        "cpu",
        "gpu",
        "io",
        "signal",
        "status",
        "settings",
    }
    if normalized in vague:
        raise ValueError(f"Signal metric '{text}' is too vague")
    return text


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _clean_strings([str(item) for item in value])
    if value is None:
        return []
    return _clean_strings([str(value)])


def _resource_root(target: str) -> str | None:
    parts = target.split(".")
    if len(parts) < 2:
        return None
    root = ".".join(parts[:2])
    return root if root in RESOURCE_TYPES else None


def _duplicate_id_errors(section: str, ids: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item_id in ids:
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)
    return [f"Duplicate {section} id {item_id}" for item_id in duplicates]


def _resources_for_target(target: str, contract: "LibraryContract") -> list[str]:
    if target in RESOURCE_TYPES:
        return [target]
    root = _resource_root(target)
    if root is not None:
        return [root]

    use_by_id = {item.id: item for item in contract.uses}
    component_ids = {item.id for item in contract.components}
    if target in use_by_id:
        return [use_by_id[target].resource]

    head = target.split(".", 1)[0]
    if head in use_by_id:
        return [use_by_id[head].resource]
    if head in component_ids:
        resources = sorted({item.resource for item in contract.uses if item.by == head})
        return resources
    return []


def _nested_path_exists(value: object, path_parts: list[str]) -> bool:
    if not path_parts:
        return True
    if not isinstance(value, dict):
        return False
    head, *tail = path_parts
    if head not in value:
        return False
    return _nested_path_exists(value[head], tail)


def _target_exists(target: str, contract: "LibraryContract") -> bool:
    if target in RESOURCE_TYPES:
        return True
    if _resource_root(target) is not None:
        return True
    use_by_id = {item.id: item for item in contract.uses}
    component_by_id = {item.id: item for item in contract.components}
    if target in use_by_id or target in component_by_id:
        return True
    head, sep, remainder = target.partition(".")
    if not sep:
        return False
    if head in component_by_id:
        return _nested_path_exists(component_by_id[head].attrs, remainder.split("."))
    if head in use_by_id:
        if remainder == "demand":
            return True
        if not remainder.startswith("demand."):
            return False
        return _nested_path_exists(use_by_id[head].demand, remainder.split(".")[1:])
    return False


def _looks_like_worker_control(name: str) -> bool:
    stripped = name.strip()
    upper = stripped.upper()
    if any(token in upper for token in ("TIMEOUT", "BLOCKTIME", "WAIT_POLICY")):
        return False
    if stripped in EXECUTOR_WORKER_CONTROL_NAMES:
        return True
    lower = stripped.lower()
    return "THREAD" in upper or lower.endswith("nthread") or lower.endswith("workers")


def _validate_component_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    for key in attrs:
        if key in TARGET_RESERVED_PREFIXES:
            raise ValueError("Component attrs must not contain nested LPCL declarations")
    return attrs


def _validate_use_demand(resource: str, demand: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(demand, dict):
        raise ValueError("Use demand must be an object")

    dimension = demand.get("dimension")
    if resource == "cpu.compute":
        if (
            dimension != "workers"
            or not _is_number_or_symbol(demand.get("min"))
            or not _is_number_or_symbol(demand.get("max"))
        ):
            raise ValueError(
                "Invalid demand schema for cpu.compute. Use {dimension: workers, min: <number_or_symbol>, max: <number_or_symbol>}."
            )
    elif resource == "cpu.placement":
        if dimension != "core_set" or not _is_symbol(demand.get("source")):
            raise ValueError(
                "Invalid demand schema for cpu.placement. Use {dimension: core_set, source: <string_or_symbol>}."
            )
    elif resource == "gpu.compute":
        if dimension == "devices":
            if not _is_number_or_symbol(demand.get("min")) or not _is_number_or_symbol(
                demand.get("max")
            ):
                raise ValueError(
                    "Invalid demand schema for gpu.compute. devices demand requires min and max."
                )
        elif dimension == "partition":
            if not _is_symbol(demand.get("source")):
                raise ValueError(
                    "Invalid demand schema for gpu.compute. partition demand requires source."
                )
        else:
            raise ValueError(
                "Invalid demand schema for gpu.compute. Use {dimension: devices, min, max} or {dimension: partition, source}."
            )
    elif resource in {"gpu.memory", "mem.capacity"}:
        if (
            dimension != "bytes"
            or not _is_number_or_symbol(demand.get("min"))
            or not _is_number_or_symbol(demand.get("max"))
        ):
            raise ValueError(
                f"Invalid demand schema for {resource}. Use {{dimension: bytes, min: <number_or_symbol>, max: <number_or_symbol>}}."
            )
    elif resource == "mem.bandwidth":
        if dimension != "intensity" or not (
            _is_number_or_symbol(demand.get("value")) or demand.get("value") == "unknown"
        ):
            raise ValueError(
                "Invalid demand schema for mem.bandwidth. Use {dimension: intensity, value: <number_or_symbol_or_unknown>}."
            )
    elif resource == "io.bandwidth":
        if (
            dimension != "bandwidth"
            or not _is_number_or_symbol(demand.get("min"))
            or not _is_number_or_symbol(demand.get("max"))
        ):
            raise ValueError(
                "Invalid demand schema for io.bandwidth. Use {dimension: bandwidth, min: <number_or_symbol>, max: <number_or_symbol>}."
            )
    elif resource == "io.queue":
        if (
            dimension != "depth"
            or not _is_number_or_symbol(demand.get("min"))
            or not _is_number_or_symbol(demand.get("max"))
        ):
            raise ValueError(
                "Invalid demand schema for io.queue. Use {dimension: depth, min: <number_or_symbol>, max: <number_or_symbol>}."
            )
    return demand


def _validate_predicate(resource: str, predicate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(predicate, dict):
        raise ValueError("Assumption predicate must be an object")
    kind = _validate_enum(str(predicate.get("kind", "")).strip(), PREDICATE_KINDS, "predicate kind")
    if kind in {"total_at_most", "competitors_at_most"}:
        if not _is_number_or_symbol(predicate.get("value")):
            raise ValueError(f"Predicate {kind} requires a value")
        return {"kind": kind, "value": predicate.get("value")}
    if kind == "not_nested" and resource not in {"cpu.compute", "gpu.compute"}:
        raise ValueError(f"Predicate not_nested is not valid for resource {resource}")
    if kind == "placement_compatible" and resource != "cpu.placement":
        raise ValueError(f"Predicate placement_compatible is not valid for resource {resource}")
    return {"kind": kind}


class ContractBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceArtifact(ContractBaseModel):
    source_id: str
    path: str
    kind: Literal["doc", "code_hint", "metadata"]


class Component(ContractBaseModel):
    id: str
    kind: str
    attrs: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str]

    @field_validator("kind", mode="after")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return _validate_enum(value, COMPONENT_KINDS, "component kind")

    @field_validator("attrs", mode="after")
    @classmethod
    def _validate_attrs(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_component_attrs(value)

    @field_validator("evidence", mode="after")
    @classmethod
    def _validate_evidence(cls, values: list[str]) -> list[str]:
        return _require_non_empty_evidence(values, "Components")


class ResourceUse(ContractBaseModel):
    id: str
    resource: str
    by: str
    phase: str
    scope: str
    demand: dict[str, Any]
    share: str
    when: str | None = None
    evidence: list[str]

    @field_validator("resource", mode="after")
    @classmethod
    def _validate_resource_value(cls, value: str) -> str:
        return _validate_resource(value)

    @field_validator("phase", mode="after")
    @classmethod
    def _validate_phase(cls, value: str) -> str:
        return _validate_enum(value, PHASES, "phase")

    @field_validator("scope", mode="after")
    @classmethod
    def _validate_scope(cls, value: str) -> str:
        return _validate_enum(value, SCOPES, "scope")

    @field_validator("share", mode="after")
    @classmethod
    def _validate_share(cls, value: str) -> str:
        return _validate_enum(value, SHARE_MODES, "share mode")

    @field_validator("when", mode="before")
    @classmethod
    def _normalize_when(cls, value: Any) -> str | None:
        return _normalize_condition(value)

    @field_validator("evidence", mode="after")
    @classmethod
    def _validate_evidence(cls, values: list[str]) -> list[str]:
        return _require_non_empty_evidence(values, "Uses")

    @model_validator(mode="after")
    def _validate_demand_shape(self) -> "ResourceUse":
        self.demand = _validate_use_demand(self.resource, self.demand)
        return self


class Assumption(ContractBaseModel):
    id: str
    resource: str
    predicate: dict[str, Any]
    when: str | None = None
    evidence: list[str]

    @field_validator("resource", mode="after")
    @classmethod
    def _validate_resource_value(cls, value: str) -> str:
        return _validate_resource(value)

    @field_validator("when", mode="before")
    @classmethod
    def _normalize_when(cls, value: Any) -> str | None:
        return _normalize_condition(value)

    @field_validator("evidence", mode="after")
    @classmethod
    def _validate_evidence(cls, values: list[str]) -> list[str]:
        return _require_non_empty_evidence(values, "Assumptions")

    @model_validator(mode="after")
    def _validate_predicate_shape(self) -> "Assumption":
        self.predicate = _validate_predicate(self.resource, self.predicate)
        return self


class ControlEffect(ContractBaseModel):
    kind: str
    value: Any | None = None
    formula: str | None = None
    direction: str | None = None
    resource: str | None = None

    @field_validator("kind", mode="after")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return _validate_enum(value, EFFECT_KINDS, "effect kind")

    @field_validator("resource", mode="after")
    @classmethod
    def _validate_resource_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_resource(value)

    @field_validator("formula", mode="before")
    @classmethod
    def _normalize_formula(cls, value: Any) -> str | None:
        return _normalize_condition(value)

    @field_validator("direction", mode="after")
    @classmethod
    def _validate_direction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_enum(value, EXPECTED_DIRECTIONS, "control effect direction")

    @model_validator(mode="after")
    def _validate_shape(self) -> "ControlEffect":
        if self.kind == "set" and self.value is None:
            raise ValueError("Control effect set requires a value")
        if self.kind == "partition" and self.resource is None:
            raise ValueError("Control effect partition requires a resource")
        return self


class Control(ContractBaseModel):
    id: str
    name: str
    kind: str
    type: str
    target: str
    effect: ControlEffect
    interacts_with: list[str] = Field(default_factory=list)
    when: str | None = None
    evidence: list[str]

    @field_validator("kind", mode="after")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return _validate_enum(value, CONTROL_KINDS, "control kind")

    @field_validator("type", mode="after")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        return _validate_enum(value, CONTROL_TYPES, "control type")

    @field_validator("when", mode="before")
    @classmethod
    def _normalize_when(cls, value: Any) -> str | None:
        return _normalize_condition(value)

    @field_validator("evidence", mode="after")
    @classmethod
    def _validate_evidence(cls, values: list[str]) -> list[str]:
        return _require_non_empty_evidence(values, "Controls")

    @field_validator("interacts_with", mode="after")
    @classmethod
    def _normalize_interacts_with(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

class Signal(ContractBaseModel):
    id: str
    target: str
    role: str
    metric: str
    indicates: str
    virtual_probe: str
    expected_direction_when_mitigated: str
    when: str | None = None
    evidence: list[str]

    @field_validator("metric", mode="after")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        return _validate_metric_name(value)

    @field_validator("role", mode="after")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        return _validate_enum(value, SIGNAL_ROLES, "signal role")

    @field_validator("virtual_probe", mode="after")
    @classmethod
    def _validate_virtual_probe(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Signal must declare a non-empty virtual_probe")
        if "/" in text or text.endswith(".py"):
            raise ValueError("Signal virtual_probe must not be a concrete script path")
        return text

    @field_validator("expected_direction_when_mitigated", mode="after")
    @classmethod
    def _validate_expected_direction(cls, value: str) -> str:
        return _validate_enum(value, EXPECTED_DIRECTIONS, "expected direction")

    @field_validator("when", mode="before")
    @classmethod
    def _normalize_when(cls, value: Any) -> str | None:
        return _normalize_condition(value)

    @field_validator("evidence", mode="after")
    @classmethod
    def _validate_evidence(cls, values: list[str]) -> list[str]:
        return _require_non_empty_evidence(values, "Signals")


class RecipeExpectedSignalChange(ContractBaseModel):
    signal: str
    expected_direction_when_mitigated: str
    reason: str | None = None

    @field_validator("expected_direction_when_mitigated", mode="after")
    @classmethod
    def _validate_direction(cls, value: str) -> str:
        return _validate_enum(value, EXPECTED_DIRECTIONS, "expected direction")

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: Any) -> str | None:
        return _normalize_condition(value)


class RecipeDiagnose(ContractBaseModel):
    label: str
    signals: list[str] = Field(default_factory=list)
    question: str | None = None

    @field_validator("label", mode="after")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Recipe diagnose blocks must include a label")
        return text

    @field_validator("signals", mode="after")
    @classmethod
    def _normalize_signals(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("question", mode="before")
    @classmethod
    def _normalize_question(cls, value: Any) -> str | None:
        return _normalize_condition(value)


class RecipeAction(ContractBaseModel):
    label: str
    summary: str
    control: str | None = None
    effect: str | None = None
    value: Any | None = None
    expected_changes: list[RecipeExpectedSignalChange] = Field(default_factory=list)

    @field_validator("label", mode="after")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Recipe actions must include a label")
        return text

    @field_validator("summary", mode="after")
    @classmethod
    def _validate_summary(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Recipe actions must include a summary")
        return text

    @field_validator("control", "effect", mode="before")
    @classmethod
    def _normalize_action_strings(cls, value: Any) -> str | None:
        text = _normalize_condition(value)
        return text


class Recipe(ContractBaseModel):
    id: str
    hypothesis_kind: str = Field(alias="for", serialization_alias="for")
    collect: list[str] = Field(default_factory=list)
    diagnose: list[RecipeDiagnose] = Field(default_factory=list)
    config_checks: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    search_space_hints: list[str] = Field(default_factory=list)
    actions: list[RecipeAction] = Field(default_factory=list)
    when: str | None = None
    evidence: list[str]

    @field_validator("hypothesis_kind", mode="after")
    @classmethod
    def _validate_hypothesis_kind(cls, value: str) -> str:
        return _validate_enum(value, HYPOTHESIS_KINDS, "hypothesis kind")

    @field_validator(
        "collect", "config_checks", "outcomes", "guardrails", "search_space_hints", mode="after"
    )
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("when", mode="before")
    @classmethod
    def _normalize_when(cls, value: Any) -> str | None:
        return _normalize_condition(value)

    @field_validator("evidence", mode="after")
    @classmethod
    def _validate_evidence(cls, values: list[str]) -> list[str]:
        return _require_non_empty_evidence(values, "Recipes")

    @model_validator(mode="after")
    def _validate_content(self) -> "Recipe":
        if not self.collect:
            raise ValueError("Recipes must collect at least one signal")
        if not self.diagnose:
            raise ValueError("Recipes must include at least one diagnose block")
        if not self.actions:
            raise ValueError("Recipes must include at least one action")
        if not self.config_checks:
            raise ValueError("Recipes must include at least one config_check")
        if not self.outcomes:
            raise ValueError("Recipes must include at least one outcome")
        if not self.guardrails:
            raise ValueError("Recipes must include at least one guardrail")
        return self


class LibraryContract(ContractBaseModel):
    version: Literal["lpcl/v1"] = CONTRACT_VERSION
    contract: str
    contract_type: Literal["library"] = Field(
        default="library", alias="type", serialization_alias="type"
    )
    components: list[Component] = Field(default_factory=list)
    uses: list[ResourceUse] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    controls: list[Control] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)
    recipes: list[Recipe] = Field(default_factory=list)
    source_artifacts: list[SourceArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_graph(self) -> "LibraryContract":
        issues: list[str] = []
        issues.extend(_duplicate_id_errors("component", [item.id for item in self.components]))
        issues.extend(_duplicate_id_errors("use", [item.id for item in self.uses]))
        issues.extend(_duplicate_id_errors("assumption", [item.id for item in self.assumptions]))
        issues.extend(_duplicate_id_errors("control", [item.id for item in self.controls]))
        issues.extend(_duplicate_id_errors("signal", [item.id for item in self.signals]))
        issues.extend(_duplicate_id_errors("recipe", [item.id for item in self.recipes]))

        component_ids = {item.id for item in self.components}
        signal_ids = {item.id for item in self.signals}
        control_ids = {item.id for item in self.controls}

        for item in self.uses:
            if item.by not in component_ids:
                issues.append(f"Use {item.id} references missing component {item.by}.")

        for control in self.controls:
            if control.name.startswith("BENCHMARK_") and not any(
                token in self.contract.lower() for token in ("benchmark", "wrapper")
            ):
                issues.append(
                    f"Control {control.name} is a benchmark-wrapper variable and must not appear in library contract {self.contract}."
                )
            if not _target_exists(control.target, self):
                if control.target == "cpu.thread" and _looks_like_worker_control(control.name):
                    issues.append(
                        f"Control {control.name} should target an executor workers attribute, not cpu.thread."
                    )
                elif control.target == "worker.pool":
                    issues.append(
                        f"Control {control.name} should target a component or component attribute, not worker.pool."
                    )
                else:
                    issues.append(f"Control {control.id} targets nonexistent {control.target}.")
            elif _looks_like_worker_control(control.name) and not control.target.endswith(
                ".workers"
            ):
                issues.append(
                    f"Control {control.name} should target an executor workers attribute, not {control.target}."
                )

        for signal in self.signals:
            if not _target_exists(signal.target, self):
                issues.append(f"Signal {signal.id} targets nonexistent {signal.target}.")

        for recipe in self.recipes:
            missing_signals = [item for item in recipe.collect if item not in signal_ids]
            if missing_signals:
                issues.append(
                    f"Recipe {recipe.id} collects missing signals: {', '.join(sorted(missing_signals))}."
                )
            for diagnose in recipe.diagnose:
                missing_diagnose_signals = [item for item in diagnose.signals if item not in signal_ids]
                if missing_diagnose_signals:
                    issues.append(
                        f"Recipe {recipe.id} diagnose block '{diagnose.label}' references missing signals: {', '.join(sorted(missing_diagnose_signals))}."
                    )
            missing_config_checks = [item for item in recipe.config_checks if item not in signal_ids]
            if missing_config_checks:
                issues.append(
                    f"Recipe {recipe.id} config_checks reference missing signals: {', '.join(sorted(missing_config_checks))}."
                )
            for action in recipe.actions:
                if action.control and action.control not in control_ids:
                    issues.append(
                        f"Recipe {recipe.id} action '{action.summary}' references missing control {action.control}."
                    )
                for change in action.expected_changes:
                    if change.signal not in signal_ids:
                        issues.append(
                            f"Recipe {recipe.id} action '{action.summary}' references missing signal {change.signal}."
                        )

        if issues:
            raise ValueError("; ".join(issues))
        return self


class ContractInstance(ContractBaseModel):
    instance_id: str
    contract_id: str


class SharedResourceItem(ContractBaseModel):
    resource: str
    phase: str
    contracts: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    uses: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("resource", mode="after")
    @classmethod
    def _validate_resource_value(cls, value: str) -> str:
        return _validate_resource(value)

    @field_validator("phase", mode="after")
    @classmethod
    def _validate_phase(cls, value: str) -> str:
        return _validate_enum(value, PHASES, "phase")

    @field_validator("contracts", "components", "uses", "evidence", mode="after")
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class HypothesisOverlap(ContractBaseModel):
    uses: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    contracts: list[str] = Field(default_factory=list)

    @field_validator("uses", "components", "contracts", mode="after")
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class AssumptionViolation(ContractBaseModel):
    assumption: str
    conflicting_uses: list[str] = Field(default_factory=list)

    @field_validator("conflicting_uses", mode="after")
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ControlCollision(ContractBaseModel):
    resource: str
    dimension: str
    controls: list[str] = Field(default_factory=list)

    @field_validator("resource", mode="after")
    @classmethod
    def _validate_resource_value(cls, value: str) -> str:
        return _validate_resource(value)

    @field_validator("controls", mode="after")
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class HypothesisBasis(ContractBaseModel):
    overlaps: list[HypothesisOverlap] = Field(default_factory=list)
    assumption_violations: list[AssumptionViolation] = Field(default_factory=list)
    control_collisions: list[ControlCollision] = Field(default_factory=list)


class HypothesisValidation(ContractBaseModel):
    collect: list[str] = Field(default_factory=list)

    @field_validator("collect", mode="after")
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ConflictHypothesis(ContractBaseModel):
    id: str
    kind: str
    resource: str
    phase: str
    contracts: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    uses: list[str] = Field(default_factory=list)
    basis: HypothesisBasis
    validation: HypothesisValidation = Field(alias="validate", serialization_alias="validate")
    candidate_actions: list[str] = Field(default_factory=list)
    recipes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="after")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return _validate_enum(value, HYPOTHESIS_KINDS, "hypothesis kind")

    @field_validator("resource", mode="after")
    @classmethod
    def _validate_resource_value(cls, value: str) -> str:
        return _validate_resource(value)

    @field_validator("phase", mode="after")
    @classmethod
    def _validate_phase(cls, value: str) -> str:
        return _validate_enum(value, PHASE_OR_MIXED, "hypothesis phase")

    @field_validator(
        "contracts", "components", "uses", "candidate_actions", "recipes", "evidence", mode="after"
    )
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _validate_content(self) -> "ConflictHypothesis":
        if not self.contracts:
            raise ValueError("Conflict hypotheses must include contracts")
        if not self.components:
            raise ValueError("Conflict hypotheses must include components")
        if not self.uses:
            raise ValueError("Conflict hypotheses must include uses")
        if not self.validation.collect:
            raise ValueError("Conflict hypotheses must include validation signals")
        if (
            not self.basis.overlaps
            and not self.basis.assumption_violations
            and not self.basis.control_collisions
        ):
            raise ValueError("Conflict hypotheses must include supporting basis items")
        return self


class ComposedContract(ContractBaseModel):
    version: Literal["lpcl/v1"] = CONTRACT_VERSION
    contract: str
    contract_type: Literal["composed"] = Field(
        default="composed", alias="type", serialization_alias="type"
    )
    selected_contracts: list[str] = Field(default_factory=list)
    contract_instances: list[ContractInstance] = Field(default_factory=list)
    shared_resources: list[SharedResourceItem] = Field(default_factory=list)
    hypotheses: list[ConflictHypothesis] = Field(default_factory=list)

    @field_validator("selected_contracts", mode="after")
    @classmethod
    def _normalize_selected_contracts(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _validate_contract_instances(self) -> "ComposedContract":
        if not self.selected_contracts:
            raise ValueError("selected_contracts must not be empty")
        if not self.contract_instances:
            self.contract_instances = [
                ContractInstance(instance_id=item, contract_id=item)
                for item in self.selected_contracts
            ]
        instance_ids = [item.instance_id for item in self.contract_instances]
        if self.selected_contracts != instance_ids:
            raise ValueError("selected_contracts must match contract_instances.instance_id order")
        return self


class ContractCheckResult(ContractBaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)

    @field_validator("issues", mode="after")
    @classmethod
    def _normalize_issues(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ContractVerificationResult(ContractBaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    repaired_contract: dict[str, Any] | None = None

    @field_validator("issues", mode="after")
    @classmethod
    def _normalize_issues(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


AnyContract = LibraryContract | ComposedContract


def as_resource_type(values: list[str]) -> list[ResourceType]:
    cleaned = _clean_strings(values)
    return cast(list[ResourceType], [value for value in RESOURCE_TYPES if value in cleaned])
