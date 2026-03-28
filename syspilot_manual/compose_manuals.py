from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from syspilot_manual.llm_client import ManualLLMProtocol, extract_json_object
from syspilot_manual.schemas import (
    ComposedManual,
    ConflictSummaryItem,
    LibraryManual,
    ResourceDomain,
    SharedResourceItem,
)


def _library_domains(manual: LibraryManual) -> set[ResourceDomain]:
    domains: set[ResourceDomain] = set(manual.composition_hints.shared_resources)
    for item in manual.feature_graph:
        domains.update(item.resource_domains)
    for item in manual.intervention_graph:
        domains.update(item.resource_domains)
    return domains


def derive_shared_resources(manuals: list[LibraryManual]) -> list[SharedResourceItem]:
    owners: dict[ResourceDomain, set[str]] = defaultdict(set)
    refs: dict[ResourceDomain, set[str]] = defaultdict(set)
    for manual in manuals:
        for domain in _library_domains(manual):
            owners[domain].add(manual.library_name)
        for feature in manual.feature_graph:
            for domain in feature.resource_domains:
                refs[domain].update(feature.source_refs)
        for intervention in manual.intervention_graph:
            for domain in intervention.resource_domains:
                refs[domain].update(intervention.source_refs)

    shared: list[SharedResourceItem] = []
    for domain, libraries in sorted(owners.items(), key=lambda item: item[0]):
        if len(libraries) < 2:
            continue
        shared.append(
            SharedResourceItem(
                resource_domain=domain,
                libraries=sorted(libraries),
                note=f"{domain} is shared across {', '.join(sorted(libraries))}.",
                source_refs=sorted(refs.get(domain, set())) or [f"shared_resource:{domain}"],
            )
        )
    return shared


def derive_conflicts(manuals: list[LibraryManual]) -> list[ConflictSummaryItem]:
    conflicts: list[ConflictSummaryItem] = []
    seen: set[tuple[str, str, str]] = set()
    for left, right in combinations(sorted(manuals, key=lambda manual: manual.manual_id), 2):
        for left_item in left.intervention_graph:
            for right_item in right.intervention_graph:
                shared_domains = sorted(
                    set(left_item.resource_domains) & set(right_item.resource_domains)
                )
                explicit_overlap = sorted(
                    set(left_item.conflicts_with) & {action.name for action in right_item.actions}
                )
                explicit_overlap += sorted(
                    set(right_item.conflicts_with) & {action.name for action in left_item.actions}
                )
                if not shared_domains and not explicit_overlap:
                    continue
                key = (
                    left_item.intervention_id,
                    right_item.intervention_id,
                    ",".join(shared_domains),
                )
                if key in seen:
                    continue
                seen.add(key)
                conflicts.append(
                    ConflictSummaryItem(
                        conflict_id=f"{left_item.intervention_id}__{right_item.intervention_id}",
                        title=f"{left.library_name} and {right.library_name} can compete for the same knobs",
                        libraries=[left.library_name, right.library_name],
                        resource_domains=shared_domains,
                        reason=(
                            f"{left.library_name} intervention '{left_item.intervention_id}' and "
                            f"{right.library_name} intervention '{right_item.intervention_id}' both touch "
                            f"{', '.join(shared_domains) or 'related tuning knobs'}."
                        ),
                        resolution_hint=(
                            "Change one competing knob family at a time and validate before combining settings."
                        ),
                        source_refs=sorted(
                            set(left_item.source_refs) | set(right_item.source_refs)
                        ),
                    )
                )
    return conflicts


def _composition_manual_payload(manual: LibraryManual) -> dict[str, Any]:
    feature_items = sorted(
        manual.feature_graph,
        key=lambda item: (-len(item.source_refs), -len(item.resource_domains), item.name.lower()),
    )[:3]
    observability_items = sorted(
        manual.observability_map,
        key=lambda item: (item.priority, -len(item.source_refs), item.question.lower()),
    )[:3]
    intervention_items = sorted(
        manual.intervention_graph,
        key=lambda item: (
            -len(item.source_refs),
            -len(item.resource_domains),
            item.intervention_id.lower(),
        ),
    )[:4]
    validation_items = sorted(
        manual.validation_recipes,
        key=lambda item: (-len(item.source_refs), item.validation_id.lower()),
    )[:3]
    conflict_patterns = sorted(
        manual.composition_hints.conflict_patterns,
        key=lambda item: (-len(item.source_refs), -len(item.with_keywords), item.reason.lower()),
    )[:3]

    return {
        "manual_id": manual.manual_id,
        "manual_type": manual.manual_type,
        "library_name": manual.library_name,
        "library_summary": manual.library_summary,
        "feature_graph": [
            {
                "feature_id": item.feature_id,
                "name": item.name,
                "resource_domains": item.resource_domains,
                "keywords": item.keywords,
            }
            for item in feature_items
        ],
        "observability_map": [
            {
                "obs_id": item.obs_id,
                "question": item.question,
                "priority": item.priority,
                "symptoms": item.symptoms,
                "signals": [
                    {
                        "name": signal.name,
                        "tool": signal.tool,
                    }
                    for signal in item.signals[:3]
                ],
            }
            for item in observability_items
        ],
        "intervention_graph": [
            {
                "intervention_id": item.intervention_id,
                "when": item.when,
                "actions": [
                    {
                        "kind": action.kind,
                        "name": action.name,
                        "suggestion": action.suggestion,
                    }
                    for action in item.actions[:3]
                ],
                "resource_domains": item.resource_domains,
                "expected_effect": item.expected_effect,
                "conflicts_with": item.conflicts_with,
            }
            for item in intervention_items
        ],
        "validation_recipes": [
            {
                "validation_id": item.validation_id,
                "for_intervention": item.for_intervention,
                "metrics": item.metrics,
                "accept_if": item.accept_if,
                "rollback_if": item.rollback_if,
            }
            for item in validation_items
        ],
        "composition_hints": {
            "shared_resources": manual.composition_hints.shared_resources,
            "conflict_patterns": [
                {
                    "with_keywords": item.with_keywords,
                    "reason": item.reason,
                    "resolution_hint": item.resolution_hint,
                }
                for item in conflict_patterns
            ],
        },
    }


async def compose_manuals(
    manual_id: str,
    app_name: str,
    manuals: list[LibraryManual],
    llm_client: ManualLLMProtocol,
    *,
    objective: str | None = None,
    symptom: str | None = None,
) -> ComposedManual:
    shared_resources = derive_shared_resources(manuals)
    conflict_summary = derive_conflicts(manuals)
    payload: dict[str, Any] = {
        "manual_id": manual_id,
        "app_name": app_name,
        "selected_libraries": [manual.library_name for manual in manuals],
        "objective": objective,
        "symptom": symptom,
        "shared_resources": [item.model_dump(mode="json") for item in shared_resources],
        "conflict_summary": [item.model_dump(mode="json") for item in conflict_summary],
        "manuals": [_composition_manual_payload(manual) for manual in manuals],
        "schema": ComposedManual.model_json_schema(),
    }
    response_text = await llm_client.complete_prompt("compose_manuals.txt", payload)
    parsed = extract_json_object(response_text)
    manual = ComposedManual.model_validate(parsed)
    return manual.model_copy(
        update={
            "manual_id": manual_id,
            "manual_type": "composed",
            "app_name": app_name,
            "selected_libraries": [manual.library_name for manual in manuals],
            "shared_resources": shared_resources or manual.shared_resources,
            "conflict_summary": conflict_summary or manual.conflict_summary,
            "recommended_query_order": manual.recommended_query_order
            or [
                "shared_resources",
                "combined_observability",
                "conflict_summary",
                "combined_interventions",
                "combined_validation",
            ],
        }
    )
