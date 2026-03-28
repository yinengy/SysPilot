from __future__ import annotations

from typing import Any, Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONTRACT_VERSION = "lpcl/v0"

RESOURCE_TYPES: tuple[str, ...] = (
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

ROLES: tuple[str, ...] = (
    "consume",
    "reserve",
    "pin",
    "serialize",
    "buffer",
    "amplify",
)

SHARING_MODES: tuple[str, ...] = (
    "shareable",
    "partitionable",
    "exclusive",
    "global",
)

ResourceType = Literal[
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
]

PhaseName = Literal[
    "init",
    "compute",
    "communication",
    "flush",
    "background",
    "teardown",
    "unknown",
]

ScopeName = Literal[
    "thread",
    "process",
    "library",
    "device",
    "system",
]

RoleName = Literal[
    "consume",
    "reserve",
    "pin",
    "serialize",
    "buffer",
    "amplify",
]

SharingMode = Literal[
    "shareable",
    "partitionable",
    "exclusive",
    "global",
]

DemandKind = Literal[
    "fixed",
    "range",
    "scaled_by",
    "proportional_to",
    "burst",
    "unknown",
]

HostSymbol = Literal[
    "host.physical_cores",
    "host.visible_cpus",
    "host.numa_nodes",
]

ControlValueType = Literal[
    "int",
    "bool",
    "enum",
    "string",
    "cpu_set",
    "numa_node_set",
]

ProbeKind = Literal[
    "command",
    "procfs",
    "runtime_env",
]

RecipeHypothesisKind = Literal[
    "baseline_validation",
    "parameter_sweep",
    "validation",
    "starting_point",
    "shared_resource_conflict",
]

ConflictHypothesisKind = Literal["shared_resource_conflict"]

PredicateKind = Literal[
    "max_competitors",
    "budget_at_most",
    "dedicated_partition",
    "affinity_compatible",
    "not_nested",
]


def _clean_strings(values: list[str]) -> list[str]:
    cleaned = [value.strip() for value in values if value and value.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


class ContractBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceArtifact(ContractBaseModel):
    source_id: str
    path: str
    kind: Literal["doc", "code_hint", "metadata"]


class FixedDemand(ContractBaseModel):
    kind: Literal["fixed"] = "fixed"
    value: int = Field(ge=0)


class RangeDemand(ContractBaseModel):
    kind: Literal["range"] = "range"
    lower: int | HostSymbol
    upper: int | HostSymbol


class ScaledByDemand(ContractBaseModel):
    kind: Literal["scaled_by"] = "scaled_by"
    knob: str
    lower: int | HostSymbol
    upper: int | HostSymbol


class ProportionalToDemand(ContractBaseModel):
    kind: Literal["proportional_to"] = "proportional_to"
    metric: str


class BurstDemand(ContractBaseModel):
    kind: Literal["burst"] = "burst"
    trigger: str


class UnknownDemand(ContractBaseModel):
    kind: Literal["unknown"] = "unknown"


DemandExpression = Annotated[
    FixedDemand | RangeDemand | ScaledByDemand | ProportionalToDemand | BurstDemand | UnknownDemand,
    Field(discriminator="kind"),
]


class ResourceUse(ContractBaseModel):
    id: str
    resource: ResourceType
    phase: PhaseName
    scope: ScopeName
    role: RoleName
    sharing: SharingMode
    demand: DemandExpression
    when: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("when", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_source_refs(self) -> "ResourceUse":
        if not self.source_refs:
            raise ValueError("Resource uses require source_refs")
        return self


class MaxCompetitorsPredicate(ContractBaseModel):
    kind: Literal["max_competitors"] = "max_competitors"
    resource: ResourceType
    phase: PhaseName
    value: int = Field(ge=1)


class BudgetAtMostPredicate(ContractBaseModel):
    kind: Literal["budget_at_most"] = "budget_at_most"
    resource: ResourceType
    phase: PhaseName
    value: int | HostSymbol


class DedicatedPartitionPredicate(ContractBaseModel):
    kind: Literal["dedicated_partition"] = "dedicated_partition"
    resource: ResourceType
    phase: PhaseName


class AffinityCompatiblePredicate(ContractBaseModel):
    kind: Literal["affinity_compatible"] = "affinity_compatible"
    resource: ResourceType
    phase: PhaseName


class NotNestedPredicate(ContractBaseModel):
    kind: Literal["not_nested"] = "not_nested"
    resource: ResourceType
    phase: PhaseName


PredicateExpression = Annotated[
    MaxCompetitorsPredicate
    | BudgetAtMostPredicate
    | DedicatedPartitionPredicate
    | AffinityCompatiblePredicate
    | NotNestedPredicate,
    Field(discriminator="kind"),
]


class Assumption(ContractBaseModel):
    id: str
    predicate: PredicateExpression
    when: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("when", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_source_refs(self) -> "Assumption":
        if not self.source_refs:
            raise ValueError("Assumptions require source_refs")
        return self


class ControlEffect(ContractBaseModel):
    op: Literal["increase", "decrease", "constrain", "select", "report", "coordinate"]
    resource: ResourceType
    strength: Literal["weak", "moderate", "strong"] = "moderate"


class SignalProbe(ContractBaseModel):
    kind: ProbeKind
    target: str


class Control(ContractBaseModel):
    id: str
    name: str
    kind: Literal["env", "config", "api", "cli", "runtime"]
    type: ControlValueType
    affects: list[str] = Field(default_factory=list)
    effect: ControlEffect
    scope: ScopeName
    allowed_values: list[str] = Field(default_factory=list)
    activation_value: str | int | bool | None = None
    source_refs: list[str]

    @field_validator("affects", "allowed_values", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_references(self) -> "Control":
        if "=" in self.name:
            raise ValueError("Controls must use canonical knob names without inline values")
        if not self.affects:
            raise ValueError("Controls require at least one affected use")
        if not self.source_refs:
            raise ValueError("Controls require source_refs")
        return self


class Signal(ContractBaseModel):
    id: str
    metric: str
    resource: ResourceType
    indicates: str
    probe: SignalProbe
    source_refs: list[str]

    @field_validator("source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_source_refs(self) -> "Signal":
        if not self.source_refs:
            raise ValueError("Signals require source_refs")
        return self


class Recipe(ContractBaseModel):
    id: str
    hypothesis_kind: RecipeHypothesisKind
    collect: list[str] = Field(default_factory=list)
    control_refs: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    accept_if: str
    rollback_if: str
    source_refs: list[str]

    @field_validator("collect", "control_refs", "actions", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_recipe_content(self) -> "Recipe":
        if not self.collect:
            raise ValueError("Recipes require at least one collected signal")
        if not self.actions:
            raise ValueError("Recipes require at least one action")
        if not self.source_refs:
            raise ValueError("Recipes require source_refs")
        return self


class LibraryContract(ContractBaseModel):
    version: Literal["lpcl/v0"] = CONTRACT_VERSION
    contract: str
    contract_type: Literal["library"] = "library"
    source_artifacts: list[SourceArtifact] = Field(default_factory=list)
    uses: list[ResourceUse] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    controls: list[Control] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)
    recipes: list[Recipe] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_sections(self) -> "LibraryContract":
        if not self.source_artifacts:
            raise ValueError("source_artifacts must not be empty")
        if not self.uses:
            raise ValueError("uses must not be empty")
        if not self.controls:
            raise ValueError("controls must not be empty")
        if not self.signals:
            raise ValueError("signals must not be empty")
        return self


class ContractInstance(ContractBaseModel):
    instance_id: str
    contract_id: str


class SharedResourceItem(ContractBaseModel):
    resource: ResourceType
    phase: PhaseName
    libraries: list[str] = Field(default_factory=list)
    use_refs: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("libraries", "use_refs", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class HypothesisOverlap(ContractBaseModel):
    uses: list[str] = Field(default_factory=list)

    @field_validator("uses", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class HypothesisBasis(ContractBaseModel):
    overlaps: list[HypothesisOverlap] = Field(default_factory=list)
    violated_assumptions: list[str] = Field(default_factory=list)
    colliding_controls: list[str] = Field(default_factory=list)

    @field_validator("violated_assumptions", "colliding_controls", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class HypothesisValidation(ContractBaseModel):
    collect: list[str] = Field(default_factory=list)

    @field_validator("collect", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ConflictHypothesis(ContractBaseModel):
    id: str
    kind: ConflictHypothesisKind
    resource: ResourceType
    phase: PhaseName
    libraries: list[str] = Field(default_factory=list)
    basis: HypothesisBasis
    validation: HypothesisValidation = Field(alias="validate", serialization_alias="validate")
    candidate_control_refs: list[str] = Field(default_factory=list)
    candidate_recipe_refs: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator(
        "libraries",
        "candidate_control_refs",
        "candidate_recipe_refs",
        "source_refs",
        mode="after",
    )
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_hypothesis_content(self) -> "ConflictHypothesis":
        if not self.libraries:
            raise ValueError("Conflict hypotheses require libraries")
        if not self.source_refs:
            raise ValueError("Conflict hypotheses require source_refs")
        return self


class ComposedContract(ContractBaseModel):
    version: Literal["lpcl/v0"] = CONTRACT_VERSION
    contract: str
    contract_type: Literal["composed"] = "composed"
    selected_contracts: list[str] = Field(default_factory=list)
    contract_instances: list[ContractInstance] = Field(default_factory=list)
    shared_resources: list[SharedResourceItem] = Field(default_factory=list)
    hypotheses: list[ConflictHypothesis] = Field(default_factory=list)

    @field_validator("selected_contracts", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_selected_contracts(self) -> "ComposedContract":
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
    def _normalize_verification_issues(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


AnyContract = LibraryContract | ComposedContract


def as_resource_type(values: list[str]) -> list[ResourceType]:
    cleaned = _clean_strings(values)
    return cast(list[ResourceType], [value for value in RESOURCE_TYPES if value in cleaned])
