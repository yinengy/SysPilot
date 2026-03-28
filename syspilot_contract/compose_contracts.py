from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import re

from syspilot_contract.checker import assert_valid_contract
from syspilot_contract.schemas import (
    ComposedContract,
    ConflictHypothesis,
    Control,
    HypothesisBasis,
    HypothesisOverlap,
    HypothesisValidation,
    LibraryContract,
    PhaseName,
    ResourceType,
    SharedResourceItem,
)


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "contract"


def _use_ref(contract: LibraryContract, use_id: str) -> str:
    return f"{contract.contract}.{use_id}"


def _assumption_ref(contract: LibraryContract, assumption_id: str) -> str:
    return f"{contract.contract}.{assumption_id}"


def _control_ref(contract: LibraryContract, control_id: str) -> str:
    return f"{contract.contract}.{control_id}"


def _signal_ref(contract: LibraryContract, signal_id: str) -> str:
    return f"{contract.contract}.{signal_id}"


def _recipe_ref(contract: LibraryContract, recipe_id: str) -> str:
    return f"{contract.contract}.{recipe_id}"


def _phase_overlaps(left: PhaseName, right: PhaseName) -> bool:
    return left == right or left == "unknown" or right == "unknown"


def _collect_controls_by_resource(contract: LibraryContract) -> dict[ResourceType, list[Control]]:
    grouped: dict[ResourceType, list[Control]] = defaultdict(list)
    for control in contract.controls:
        grouped[control.effect.resource].append(control)
    return grouped


def derive_shared_resources(contracts: list[LibraryContract]) -> list[SharedResourceItem]:
    grouped: dict[tuple[ResourceType, PhaseName], list[tuple[LibraryContract, str, list[str]]]] = (
        defaultdict(list)
    )
    for contract in contracts:
        for item in contract.uses:
            grouped[(item.resource, item.phase)].append((contract, item.id, item.source_refs))

    shared: list[SharedResourceItem] = []
    for (resource, phase), entries in sorted(grouped.items(), key=lambda item: item[0]):
        libraries = sorted({contract.contract for contract, _, _ in entries})
        if len(libraries) < 2:
            continue
        shared.append(
            SharedResourceItem(
                resource=resource,
                phase=phase,
                libraries=libraries,
                use_refs=sorted(_use_ref(contract, use_id) for contract, use_id, _ in entries),
                source_refs=sorted(
                    {source_ref for _, _, source_refs in entries for source_ref in source_refs}
                ),
            )
        )
    return shared


def _violated_assumptions(
    left: LibraryContract,
    right: LibraryContract,
    resource: ResourceType,
    phase: PhaseName,
) -> list[str]:
    violated: list[str] = []
    for contract, other in ((left, right), (right, left)):
        other_has_overlap = any(
            use.resource == resource and _phase_overlaps(use.phase, phase) for use in other.uses
        )
        if not other_has_overlap:
            continue
        for assumption in contract.assumptions:
            predicate = assumption.predicate
            if predicate.resource != resource or not _phase_overlaps(predicate.phase, phase):
                continue
            if predicate.kind == "max_competitors" and getattr(predicate, "value", 0) <= 1:
                violated.append(_assumption_ref(contract, assumption.id))
            elif predicate.kind == "dedicated_partition":
                violated.append(_assumption_ref(contract, assumption.id))
            elif predicate.kind == "budget_at_most":
                violated.append(_assumption_ref(contract, assumption.id))
            elif predicate.kind == "affinity_compatible" and resource == "cpu.affinity":
                violated.append(_assumption_ref(contract, assumption.id))
    return sorted(set(violated))


def _candidate_recipe_refs(
    left: LibraryContract,
    right: LibraryContract,
    resource: ResourceType,
    controls: list[Control],
) -> list[str]:
    recipe_refs: list[str] = []
    for contract in (left, right):
        relevant_signal_ids = {
            signal.id for signal in contract.signals if signal.resource == resource
        }
        relevant_control_ids = {
            control.id for control in contract.controls if control.effect.resource == resource
        }
        for recipe in contract.recipes:
            if (
                relevant_signal_ids.intersection(recipe.collect)
                or relevant_control_ids.intersection(recipe.control_refs)
                or recipe.hypothesis_kind == "shared_resource_conflict"
            ):
                recipe_refs.append(_recipe_ref(contract, recipe.id))

    if not recipe_refs and controls:
        for contract in (left, right):
            for recipe in contract.recipes:
                if recipe.hypothesis_kind in {"validation", "starting_point", "parameter_sweep"}:
                    recipe_refs.append(_recipe_ref(contract, recipe.id))
    return sorted(dict.fromkeys(recipe_refs))


def _validation_signal_refs(
    left: LibraryContract,
    right: LibraryContract,
    resource: ResourceType,
) -> list[str]:
    signal_refs = [
        _signal_ref(contract, signal.id)
        for contract in (left, right)
        for signal in contract.signals
        if signal.resource == resource
    ]
    return sorted(dict.fromkeys(signal_refs))


def derive_conflict_hypotheses(contracts: list[LibraryContract]) -> list[ConflictHypothesis]:
    hypotheses: list[ConflictHypothesis] = []
    seen: set[tuple[str, str, str]] = set()

    for left, right in combinations(sorted(contracts, key=lambda item: item.contract.lower()), 2):
        left_controls = _collect_controls_by_resource(left)
        right_controls = _collect_controls_by_resource(right)
        for left_use in left.uses:
            for right_use in right.uses:
                if left_use.resource != right_use.resource:
                    continue
                if not _phase_overlaps(left_use.phase, right_use.phase):
                    continue
                phase = left_use.phase if left_use.phase != "unknown" else right_use.phase
                key = (left.contract, right.contract, f"{left_use.resource}:{phase}")
                if key in seen:
                    continue

                controls = sorted(
                    left_controls.get(left_use.resource, [])
                    + right_controls.get(right_use.resource, []),
                    key=lambda item: (item.name.lower(), item.id.lower()),
                )
                colliding_controls = [
                    _control_ref(contract, control.id)
                    for contract in (left, right)
                    for control in contract.controls
                    if control.effect.resource == left_use.resource
                    and control.scope in {"process", "library", "system", "device"}
                ]
                violated_assumptions = _violated_assumptions(left, right, left_use.resource, phase)
                sharing_trigger = any(
                    item.sharing in {"partitionable", "exclusive", "global"}
                    for item in (left_use, right_use)
                )
                role_trigger = {left_use.role, right_use.role} & {"consume", "reserve", "pin"}

                if (
                    not colliding_controls
                    and not violated_assumptions
                    and not sharing_trigger
                    and len(role_trigger) < 2
                ):
                    continue

                seen.add(key)
                source_refs = sorted(
                    set(left_use.source_refs)
                    | set(right_use.source_refs)
                    | {
                        source_ref
                        for item in left.assumptions + right.assumptions
                        for source_ref in item.source_refs
                        if _phase_overlaps(item.predicate.phase, phase)
                        and item.predicate.resource == left_use.resource
                    }
                    | {source_ref for control in controls for source_ref in control.source_refs}
                    | {
                        source_ref
                        for contract in (left, right)
                        for signal in contract.signals
                        for source_ref in signal.source_refs
                        if signal.resource == left_use.resource
                    }
                )
                hypotheses.append(
                    ConflictHypothesis(
                        id=f"h_{_slugify(left_use.resource)}_{_slugify(left.contract)}_{_slugify(right.contract)}_{_slugify(phase)}",
                        kind="shared_resource_conflict",
                        resource=left_use.resource,
                        phase=phase,
                        libraries=sorted([left.contract, right.contract]),
                        basis=HypothesisBasis(
                            overlaps=[
                                HypothesisOverlap(
                                    uses=sorted(
                                        [_use_ref(left, left_use.id), _use_ref(right, right_use.id)]
                                    )
                                )
                            ],
                            violated_assumptions=violated_assumptions,
                            colliding_controls=sorted(set(colliding_controls)),
                        ),
                        validate=HypothesisValidation(
                            collect=_validation_signal_refs(left, right, left_use.resource)
                        ),
                        candidate_control_refs=sorted(set(colliding_controls)),
                        candidate_recipe_refs=_candidate_recipe_refs(
                            left,
                            right,
                            left_use.resource,
                            controls,
                        ),
                        source_refs=source_refs,
                    )
                )
    return sorted(hypotheses, key=lambda item: (item.resource, item.phase, item.id))


def compose_contracts(
    contract_name: str,
    contracts: list[LibraryContract],
) -> ComposedContract:
    selected_contracts = sorted(contract.contract for contract in contracts)
    composed = ComposedContract(
        contract=contract_name,
        contract_type="composed",
        version="lpcl/v0",
        selected_contracts=selected_contracts,
        shared_resources=derive_shared_resources(contracts),
        hypotheses=derive_conflict_hypotheses(contracts),
    )
    verified = assert_valid_contract(composed)
    if not isinstance(verified, ComposedContract):
        raise TypeError("Expected a composed contract after validation")
    return verified
