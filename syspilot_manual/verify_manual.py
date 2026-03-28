from __future__ import annotations

from typing import Any

from syspilot_manual.llm_client import ManualLLMProtocol, extract_json_object
from syspilot_manual.schemas import ComposedManual, LibraryManual, ManualVerificationResult


def _library_issues(manual: LibraryManual) -> list[str]:
    issues: list[str] = []
    validation_targets = {recipe.for_intervention for recipe in manual.validation_recipes}
    for item in manual.feature_graph:
        if not item.source_refs:
            issues.append(f"feature_graph:{item.feature_id} missing source_refs")
    for item in manual.observability_map:
        if not item.source_refs:
            issues.append(f"observability_map:{item.obs_id} missing source_refs")
        if not item.question.endswith("?"):
            issues.append(
                f"observability_map:{item.obs_id} should ask a concrete profiling question"
            )
    for item in manual.intervention_graph:
        if not item.source_refs:
            issues.append(f"intervention_graph:{item.intervention_id} missing source_refs")
        if item.intervention_id not in validation_targets:
            issues.append(
                f"intervention_graph:{item.intervention_id} is missing a validation recipe"
            )
    if manual.composition_hints is None:
        issues.append("composition_hints missing")
    return issues


def _composed_issues(manual: ComposedManual) -> list[str]:
    issues: list[str] = []
    for item in manual.shared_resources:
        if not item.source_refs:
            issues.append(f"shared_resources:{item.resource_domain} missing source_refs")
    for item in manual.conflict_summary:
        if not item.source_refs:
            issues.append(f"conflict_summary:{item.conflict_id} missing source_refs")
    for item in manual.combined_observability:
        if not item.source_refs:
            issues.append(f"combined_observability:{item.obs_id} missing source_refs")
    for item in manual.combined_interventions:
        if not item.source_refs:
            issues.append(f"combined_interventions:{item.intervention_id} missing source_refs")
    for item in manual.combined_validation:
        if not item.source_refs:
            issues.append(f"combined_validation:{item.validation_id} missing source_refs")
    return issues


async def verify_or_repair_manual(
    manual: LibraryManual | ComposedManual,
    llm_client: ManualLLMProtocol | None = None,
) -> LibraryManual | ComposedManual:
    issues = (
        _library_issues(manual) if isinstance(manual, LibraryManual) else _composed_issues(manual)
    )
    if llm_client is None:
        if issues:
            raise ValueError("; ".join(issues))
        return manual

    payload: dict[str, Any] = {
        "manual_type": manual.manual_type,
        "manual": manual.model_dump(mode="json"),
        "issues": issues,
        "schema": (
            LibraryManual.model_json_schema()
            if isinstance(manual, LibraryManual)
            else ComposedManual.model_json_schema()
        ),
        "verification_schema": ManualVerificationResult.model_json_schema(),
    }
    response_text = await llm_client.complete_prompt("verify_manual.txt", payload)
    parsed = extract_json_object(response_text)
    verification = ManualVerificationResult.model_validate(parsed)

    if verification.valid and not verification.repaired_manual:
        if issues:
            raise ValueError("; ".join(issues))
        return manual

    if verification.repaired_manual is None:
        raise ValueError("Manual verification returned invalid without a repair payload")

    if isinstance(manual, LibraryManual):
        return LibraryManual.model_validate(verification.repaired_manual)
    return ComposedManual.model_validate(verification.repaired_manual)
