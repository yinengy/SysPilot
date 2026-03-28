from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RESOURCE_DOMAINS: tuple[str, ...] = (
    "cpu_threads",
    "cpu_affinity",
    "memory_capacity",
    "memory_bandwidth",
    "allocator",
    "cache",
    "io_queue",
    "disk_io",
    "network_io",
    "batching",
    "workers",
)

ResourceDomain = Literal[
    "cpu_threads",
    "cpu_affinity",
    "memory_capacity",
    "memory_bandwidth",
    "allocator",
    "cache",
    "io_queue",
    "disk_io",
    "network_io",
    "batching",
    "workers",
]

EvidenceKind = Literal[
    "feature",
    "observability",
    "intervention",
    "validation",
    "composition_hint",
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


def _normalize_domains(values: list[ResourceDomain]) -> list[ResourceDomain]:
    cleaned = _clean_strings(cast(list[str], values))
    return cast(list[ResourceDomain], [domain for domain in RESOURCE_DOMAINS if domain in cleaned])


class ManualBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class SourceArtifact(ManualBaseModel):
    source_id: str
    path: str
    kind: Literal["doc", "code_hint", "metadata"]


class IngestedSource(SourceArtifact):
    title: str
    content: str


class CodeHint(ManualBaseModel):
    hint_type: Literal["env_var", "cli_flag", "config_key", "default", "warning"]
    value: str
    snippet: str


class SourceChunk(ManualBaseModel):
    source_id: str
    chunk_id: str
    path: str
    kind: Literal["doc", "code_hint", "metadata"]
    order: int
    heading_path: list[str] = Field(default_factory=list)
    text: str
    code_hints: list[CodeHint] = Field(default_factory=list)


class EvidenceItem(ManualBaseModel):
    evidence_id: str
    library_name: str
    chunk_id: str
    kind: EvidenceKind
    title: str
    content: str
    keywords: list[str] = Field(default_factory=list)
    resource_domains: list[ResourceDomain] = Field(default_factory=list)
    source_quote: str
    source_refs: list[str]
    confidence_note: str | None = None

    @field_validator("keywords", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("resource_domains", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)

    @model_validator(mode="after")
    def _require_source_refs(self) -> "EvidenceItem":
        if not self.source_refs:
            raise ValueError("Evidence items require source_refs")
        return self


class EvidenceBatch(ManualBaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)


class FeatureGraphItem(ManualBaseModel):
    feature_id: str
    name: str
    description: str
    resource_domains: list[ResourceDomain] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("keywords", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("resource_domains", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)


class ObservabilitySignal(ManualBaseModel):
    name: str
    tool: str
    reason: str


class ObservabilityItem(ManualBaseModel):
    obs_id: str
    feature_id: str
    question: str
    priority: int = Field(ge=1)
    symptoms: list[str] = Field(default_factory=list)
    signals: list[ObservabilitySignal] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("symptoms", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_signals(self) -> "ObservabilityItem":
        if not self.signals:
            raise ValueError("Observability items require at least one signal")
        return self


class InterventionAction(ManualBaseModel):
    kind: Literal["env", "config", "cli", "api"]
    name: str
    suggestion: str


class InterventionItem(ManualBaseModel):
    intervention_id: str
    feature_id: str
    when: list[str] = Field(default_factory=list)
    actions: list[InterventionAction] = Field(default_factory=list)
    resource_domains: list[ResourceDomain] = Field(default_factory=list)
    expected_effect: str
    side_effects: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("when", "side_effects", "conflicts_with", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("resource_domains", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)

    @model_validator(mode="after")
    def _require_actions(self) -> "InterventionItem":
        if not self.actions:
            raise ValueError("Intervention items require at least one action")
        return self


class ValidationRecipe(ManualBaseModel):
    validation_id: str
    for_intervention: str
    reruns: int = Field(ge=1)
    metrics: list[str] = Field(default_factory=list)
    accept_if: str
    rollback_if: str
    source_refs: list[str]

    @field_validator("metrics", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ConflictPattern(ManualBaseModel):
    with_keywords: list[str] = Field(default_factory=list)
    reason: str
    resolution_hint: str
    source_refs: list[str]

    @field_validator("with_keywords", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class CompositionHints(ManualBaseModel):
    shared_resources: list[ResourceDomain] = Field(default_factory=list)
    conflict_patterns: list[ConflictPattern] = Field(default_factory=list)

    @field_validator("shared_resources", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)


class LibraryManual(ManualBaseModel):
    manual_id: str
    manual_type: Literal["library"] = "library"
    library_name: str
    library_summary: str
    source_artifacts: list[SourceArtifact] = Field(default_factory=list)
    feature_graph: list[FeatureGraphItem] = Field(default_factory=list)
    observability_map: list[ObservabilityItem] = Field(default_factory=list)
    intervention_graph: list[InterventionItem] = Field(default_factory=list)
    validation_recipes: list[ValidationRecipe] = Field(default_factory=list)
    composition_hints: CompositionHints

    @model_validator(mode="after")
    def _require_sections(self) -> "LibraryManual":
        if not self.feature_graph:
            raise ValueError("feature_graph must not be empty")
        if not self.observability_map:
            raise ValueError("observability_map must not be empty")
        if not self.intervention_graph:
            raise ValueError("intervention_graph must not be empty")
        if not self.validation_recipes:
            raise ValueError("validation_recipes must not be empty")
        if not self.source_artifacts:
            raise ValueError("source_artifacts must not be empty")
        return self


class SharedResourceItem(ManualBaseModel):
    resource_domain: ResourceDomain
    libraries: list[str] = Field(default_factory=list)
    note: str
    source_refs: list[str]

    @field_validator("libraries", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ConflictSummaryItem(ManualBaseModel):
    conflict_id: str
    title: str
    libraries: list[str] = Field(default_factory=list)
    resource_domains: list[ResourceDomain] = Field(default_factory=list)
    reason: str
    resolution_hint: str
    source_refs: list[str]

    @field_validator("libraries", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("resource_domains", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)


class CombinedObservabilityItem(ManualBaseModel):
    obs_id: str
    library: str
    question: str
    priority: int = Field(ge=1)
    symptoms: list[str] = Field(default_factory=list)
    signals: list[ObservabilitySignal] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("symptoms", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class CombinedInterventionItem(ManualBaseModel):
    intervention_id: str
    library: str
    when: list[str] = Field(default_factory=list)
    actions: list[InterventionAction] = Field(default_factory=list)
    resource_domains: list[ResourceDomain] = Field(default_factory=list)
    expected_effect: str
    side_effects: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    source_refs: list[str]

    @field_validator("when", "side_effects", "conflicts_with", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("resource_domains", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)


class CombinedValidationItem(ManualBaseModel):
    validation_id: str
    library: str
    for_intervention: str
    reruns: int = Field(ge=1)
    metrics: list[str] = Field(default_factory=list)
    accept_if: str
    rollback_if: str
    source_refs: list[str]

    @field_validator("metrics", "source_refs", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)


class ComposedManual(ManualBaseModel):
    manual_id: str
    manual_type: Literal["composed"] = "composed"
    app_name: str
    selected_libraries: list[str] = Field(default_factory=list)
    app_summary: str
    shared_resources: list[SharedResourceItem] = Field(default_factory=list)
    conflict_summary: list[ConflictSummaryItem] = Field(default_factory=list)
    combined_observability: list[CombinedObservabilityItem] = Field(default_factory=list)
    combined_interventions: list[CombinedInterventionItem] = Field(default_factory=list)
    combined_validation: list[CombinedValidationItem] = Field(default_factory=list)
    recommended_query_order: list[str] = Field(default_factory=list)

    @field_validator("selected_libraries", "recommended_query_order", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @model_validator(mode="after")
    def _require_sections(self) -> "ComposedManual":
        if not self.selected_libraries:
            raise ValueError("selected_libraries must not be empty")
        if not self.shared_resources:
            raise ValueError("shared_resources must not be empty")
        if not self.combined_observability:
            raise ValueError("combined_observability must not be empty")
        if not self.combined_interventions:
            raise ValueError("combined_interventions must not be empty")
        if not self.combined_validation:
            raise ValueError("combined_validation must not be empty")
        if not self.recommended_query_order:
            raise ValueError("recommended_query_order must not be empty")
        return self


class ManualIndexEntry(ManualBaseModel):
    manual_id: str
    manual_type: Literal["library", "composed"]
    section: str
    item_id: str
    title: str
    text: str
    keywords: list[str] = Field(default_factory=list)
    resource_domains: list[ResourceDomain] = Field(default_factory=list)
    source_refs: list[str]
    library: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)

    @field_validator("keywords", "source_refs", "symptoms", "goals", mode="after")
    @classmethod
    def _normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)

    @field_validator("resource_domains", mode="after")
    @classmethod
    def _normalize_resource_domains(cls, values: list[ResourceDomain]) -> list[ResourceDomain]:
        return _normalize_domains(values)


class ManualVerificationResult(ManualBaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    repaired_manual: dict[str, Any] | None = None

    @field_validator("issues", mode="after")
    @classmethod
    def _normalize_issues(cls, values: list[str]) -> list[str]:
        return _clean_strings(values)
