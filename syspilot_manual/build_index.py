from __future__ import annotations

import json

from syspilot_manual.schemas import ComposedManual, LibraryManual, ManualIndexEntry


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "feature_graph": ("feature_graph",),
    "observability_map": ("observability_map", "combined_observability"),
    "intervention_graph": ("intervention_graph", "combined_interventions"),
    "validation_recipes": ("validation_recipes", "combined_validation"),
    "composition_hints": ("composition_hints",),
    "shared_resources": ("shared_resources",),
    "conflicts": ("conflict_summary",),
    "conflict_summary": ("conflict_summary",),
}


def _join(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_manual_index(manual: LibraryManual | ComposedManual) -> list[ManualIndexEntry]:
    rows: list[ManualIndexEntry] = []
    if isinstance(manual, LibraryManual):
        for item in manual.feature_graph:
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="feature_graph",
                    item_id=item.feature_id,
                    title=item.name,
                    text=item.description,
                    keywords=item.keywords,
                    resource_domains=item.resource_domains,
                    source_refs=item.source_refs,
                    library=manual.library_name,
                )
            )
        for item in manual.observability_map:
            signal_text = [
                f"{signal.name} ({signal.tool}): {signal.reason}" for signal in item.signals
            ]
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="observability_map",
                    item_id=item.obs_id,
                    title=item.question,
                    text=_join([item.question, *signal_text, *item.symptoms]),
                    keywords=[signal.name for signal in item.signals],
                    resource_domains=sorted(
                        {
                            domain
                            for feature in manual.feature_graph
                            if feature.feature_id == item.feature_id
                            for domain in feature.resource_domains
                        }
                    ),
                    source_refs=item.source_refs,
                    library=manual.library_name,
                    symptoms=item.symptoms,
                )
            )
        for item in manual.intervention_graph:
            action_text = [f"{action.name}: {action.suggestion}" for action in item.actions]
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="intervention_graph",
                    item_id=item.intervention_id,
                    title=action_text[0] if action_text else item.intervention_id,
                    text=_join(
                        [*item.when, *action_text, item.expected_effect, *item.side_effects]
                    ),
                    keywords=[action.name for action in item.actions] + item.conflicts_with,
                    resource_domains=item.resource_domains,
                    source_refs=item.source_refs,
                    library=manual.library_name,
                    symptoms=item.when,
                    goals=[item.expected_effect],
                )
            )
        for item in manual.validation_recipes:
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="validation_recipes",
                    item_id=item.validation_id,
                    title=item.for_intervention,
                    text=_join([item.accept_if, item.rollback_if, *item.metrics]),
                    keywords=item.metrics,
                    resource_domains=[],
                    source_refs=item.source_refs,
                    library=manual.library_name,
                    goals=[item.accept_if],
                )
            )
        for index, pattern in enumerate(manual.composition_hints.conflict_patterns):
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="composition_hints",
                    item_id=f"conflict_pattern_{index:03d}",
                    title=", ".join(pattern.with_keywords) or f"conflict_pattern_{index:03d}",
                    text=_join([pattern.reason, pattern.resolution_hint]),
                    keywords=pattern.with_keywords,
                    resource_domains=manual.composition_hints.shared_resources,
                    source_refs=pattern.source_refs,
                    library=manual.library_name,
                )
            )
    else:
        for item in manual.shared_resources:
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="shared_resources",
                    item_id=item.resource_domain,
                    title=item.resource_domain,
                    text=_join([item.note, ", ".join(item.libraries)]),
                    keywords=item.libraries,
                    resource_domains=[item.resource_domain],
                    source_refs=item.source_refs,
                )
            )
        for item in manual.conflict_summary:
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="conflict_summary",
                    item_id=item.conflict_id,
                    title=item.title,
                    text=_join([item.reason, item.resolution_hint]),
                    keywords=item.libraries,
                    resource_domains=item.resource_domains,
                    source_refs=item.source_refs,
                    symptoms=[item.reason],
                )
            )
        for item in manual.combined_observability:
            signal_text = [
                f"{signal.name} ({signal.tool}): {signal.reason}" for signal in item.signals
            ]
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="combined_observability",
                    item_id=item.obs_id,
                    title=f"{item.library}: {item.question}",
                    text=_join([item.question, *signal_text, *item.symptoms]),
                    keywords=[signal.name for signal in item.signals],
                    resource_domains=[],
                    source_refs=item.source_refs,
                    library=item.library,
                    symptoms=item.symptoms,
                )
            )
        for item in manual.combined_interventions:
            action_text = [f"{action.name}: {action.suggestion}" for action in item.actions]
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="combined_interventions",
                    item_id=item.intervention_id,
                    title=f"{item.library}: {action_text[0] if action_text else item.intervention_id}",
                    text=_join(
                        [*item.when, *action_text, item.expected_effect, *item.side_effects]
                    ),
                    keywords=[action.name for action in item.actions] + item.conflicts_with,
                    resource_domains=item.resource_domains,
                    source_refs=item.source_refs,
                    library=item.library,
                    symptoms=item.when,
                    goals=[item.expected_effect],
                )
            )
        for item in manual.combined_validation:
            rows.append(
                ManualIndexEntry(
                    manual_id=manual.manual_id,
                    manual_type=manual.manual_type,
                    section="combined_validation",
                    item_id=item.validation_id,
                    title=f"{item.library}: {item.for_intervention}",
                    text=_join([item.accept_if, item.rollback_if, *item.metrics]),
                    keywords=item.metrics,
                    resource_domains=[],
                    source_refs=item.source_refs,
                    library=item.library,
                    goals=[item.accept_if],
                )
            )
    return sorted(rows, key=lambda row: (row.section, row.item_id, row.title))


def render_index_jsonl(rows: list[ManualIndexEntry]) -> str:
    return "".join(
        json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
        for row in sorted(rows, key=lambda row: (row.section, row.item_id, row.title))
    )
