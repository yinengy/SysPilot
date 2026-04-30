from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import re

from syspilot_contract.checker import assert_valid_contract
from syspilot_contract.schemas import (
    ComposedContract,
    ConflictHypothesis,
    ContractInstance,
    Control,
    ControlCollision,
    HypothesisBasis,
    HypothesisOverlap,
    HypothesisValidation,
    LibraryContract,
    RESOURCE_TO_HYPOTHESIS_KIND,
    ResourceUse,
    SharedResourceItem,
    AssumptionViolation,
)


ALLOWED_SHARED_MODES = {"shared", "process_global", "partitioned"}


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "contract"


def _use_ref(contract: LibraryContract, use_id: str) -> str:
    return f"{contract.contract}.{use_id}"


def _component_ref(contract: LibraryContract, component_id: str) -> str:
    return f"{contract.contract}.{component_id}"


def _assumption_ref(contract: LibraryContract, assumption_id: str) -> str:
    return f"{contract.contract}.{assumption_id}"


def _control_ref(contract: LibraryContract, control_id: str) -> str:
    return f"{contract.contract}.{control_id}"


def _signal_ref(contract: LibraryContract, signal_id: str) -> str:
    return f"{contract.contract}.{signal_id}"


def _recipe_ref(contract: LibraryContract, recipe_id: str) -> str:
    return f"{contract.contract}.{recipe_id}"


def _instantiate_contracts(
    contracts: list[LibraryContract],
) -> tuple[list[LibraryContract], list[ContractInstance]]:
    base_totals: dict[str, int] = defaultdict(int)
    base_labels: list[str] = []
    for contract in contracts:
        base_label = _slugify(contract.contract)
        base_labels.append(base_label)
        base_totals[base_label] += 1

    base_seen: dict[str, int] = defaultdict(int)
    instantiated: list[LibraryContract] = []
    instances: list[ContractInstance] = []
    for contract, base_label in zip(contracts, base_labels, strict=False):
        base_seen[base_label] += 1
        ordinal = base_seen[base_label]
        instance_id = f"{base_label}_{ordinal}" if base_totals[base_label] > 1 else base_label
        instantiated.append(contract.model_copy(update={"contract": instance_id}))
        instances.append(ContractInstance(instance_id=instance_id, contract_id=contract.contract))
    return instantiated, instances


def _phase_overlaps(left: str, right: str) -> bool:
    return left == right or left == "unknown" or right == "unknown"


def _merged_phase(left: str, right: str) -> str:
    if left == right:
        return left
    if left == "unknown":
        return right
    if right == "unknown":
        return left
    return "mixed"


def _scope_overlaps(left: str, right: str) -> bool:
    return left == right or left in {"library", "system"} or right in {"library", "system"}


def _resource_root(target: str) -> str | None:
    parts = target.split(".")
    if len(parts) < 2:
        return None
    root = ".".join(parts[:2])
    return root if root in RESOURCE_TO_HYPOTHESIS_KIND else None


def _resources_for_target(contract: LibraryContract, target: str) -> list[str]:
    if target in RESOURCE_TO_HYPOTHESIS_KIND:
        return [target]
    root = _resource_root(target)
    if root is not None:
        return [root]
    use_by_id = {item.id: item for item in contract.uses}
    if target in use_by_id:
        return [use_by_id[target].resource]
    head = target.split(".", 1)[0]
    if head in use_by_id:
        return [use_by_id[head].resource]
    resources = sorted({item.resource for item in contract.uses if item.by == head})
    return resources


def _target_dimension(target: str, resource: str) -> str:
    if ".workers" in target or resource == "cpu.compute":
        return "workers"
    if resource == "cpu.placement":
        return "core_set"
    if resource == "gpu.compute":
        return "devices"
    if resource in {"gpu.memory", "mem.capacity"}:
        return "bytes"
    if resource == "mem.bandwidth":
        return "intensity"
    if resource == "io.bandwidth":
        return "bandwidth"
    if resource == "io.queue":
        return "depth"
    if "." in target:
        return target.rsplit(".", 1)[-1]
    return resource


def _use_share_allows_overlap(use: ResourceUse) -> bool:
    return use.share in ALLOWED_SHARED_MODES


def _component_ref_for_use(contract: LibraryContract, use: ResourceUse) -> str:
    return _component_ref(contract, use.by)


def _is_unknown_bound(value: object) -> bool:
    return not isinstance(value, (int, float)) or isinstance(value, bool)


def _worker_bound_summary(use: ResourceUse) -> tuple[object | None, object | None]:
    if use.resource != "cpu.compute":
        return None, None
    return use.demand.get("min"), use.demand.get("max")


def _cpu_compute_oversubscribed(pairs: list[tuple[ResourceUse, ResourceUse]]) -> bool:
    for left_use, right_use in pairs:
        if left_use.resource != "cpu.compute" or right_use.resource != "cpu.compute":
            continue
        if not _scope_overlaps(left_use.scope, right_use.scope):
            continue
        left_min, left_max = _worker_bound_summary(left_use)
        right_min, right_max = _worker_bound_summary(right_use)
        if _is_unknown_bound(left_max) or _is_unknown_bound(right_max):
            return True
        if isinstance(left_max, (int, float)) and isinstance(right_max, (int, float)):
            if left_max + right_max > 1:
                return True
        if left_min == right_min == 1 and left_max == right_max == 1:
            continue
    return False


def _assumption_violations(
    left: LibraryContract,
    right: LibraryContract,
    resource: str,
    overlap_pairs: list[tuple[ResourceUse, ResourceUse]],
) -> list[AssumptionViolation]:
    violations: list[AssumptionViolation] = []
    overlap_use_refs = sorted(
        {_use_ref(left, left_use.id) for left_use, _ in overlap_pairs}
        | {_use_ref(right, right_use.id) for _, right_use in overlap_pairs}
    )
    for contract, other in ((left, right), (right, left)):
        other_has_overlap = any(use.resource == resource for use in other.uses)
        if not other_has_overlap:
            continue
        for assumption in contract.assumptions:
            if assumption.resource != resource:
                continue
            predicate_kind = str(assumption.predicate.get("kind", "")).strip()
            should_record = False
            if predicate_kind == "not_nested" and resource in {"cpu.compute", "gpu.compute"}:
                should_record = True
            elif predicate_kind == "exclusive":
                should_record = True
            elif predicate_kind == "competitors_at_most" and assumption.predicate.get("value") == 1:
                should_record = True
            elif predicate_kind in {"total_at_most", "placement_compatible", "competitors_at_most"}:
                should_record = True
            if should_record:
                violations.append(
                    AssumptionViolation(
                        assumption=_assumption_ref(contract, assumption.id),
                        conflicting_uses=overlap_use_refs,
                    )
                )
    return sorted(violations, key=lambda item: item.assumption)


def _controls_for_resource(contract: LibraryContract, resource: str) -> list[Control]:
    return [
        control
        for control in contract.controls
        if resource in _resources_for_target(contract, control.target)
    ]


def _control_collisions(
    left: LibraryContract, right: LibraryContract, resource: str
) -> list[ControlCollision]:
    left_controls = _controls_for_resource(left, resource)
    right_controls = _controls_for_resource(right, resource)
    collisions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for left_control in left_controls:
        left_dimension = _target_dimension(left_control.target, resource)
        for right_control in right_controls:
            right_dimension = _target_dimension(right_control.target, resource)
            if left_dimension != right_dimension:
                continue
            key = (resource, left_dimension)
            collisions[key].add(_control_ref(left, left_control.id))
            collisions[key].add(_control_ref(right, right_control.id))
    return [
        ControlCollision(resource=resource, dimension=dimension, controls=sorted(refs))
        for (resource, dimension), refs in sorted(collisions.items())
    ]


def _signal_relevant(
    contract: LibraryContract,
    resource: str,
    component_ids: set[str],
    use_ids: set[str],
    signal_target: str,
) -> bool:
    if signal_target == resource or resource in _resources_for_target(contract, signal_target):
        return True
    head = signal_target.split(".", 1)[0]
    return head in component_ids or head in use_ids


def _signal_refs(
    left: LibraryContract,
    right: LibraryContract,
    resource: str,
    component_refs: list[str],
    use_refs: list[str],
) -> list[str]:
    relevant: list[str] = []
    component_ids_by_contract = {
        left.contract: {
            ref.split(".", 1)[1] for ref in component_refs if ref.startswith(f"{left.contract}.")
        },
        right.contract: {
            ref.split(".", 1)[1] for ref in component_refs if ref.startswith(f"{right.contract}.")
        },
    }
    use_ids_by_contract = {
        left.contract: {
            ref.split(".", 1)[1] for ref in use_refs if ref.startswith(f"{left.contract}.")
        },
        right.contract: {
            ref.split(".", 1)[1] for ref in use_refs if ref.startswith(f"{right.contract}.")
        },
    }
    for contract in (left, right):
        for signal in contract.signals:
            if _signal_relevant(
                contract,
                resource,
                component_ids_by_contract[contract.contract],
                use_ids_by_contract[contract.contract],
                signal.target,
            ):
                relevant.append(_signal_ref(contract, signal.id))
    return sorted(dict.fromkeys(relevant))


def _recipe_refs(left: LibraryContract, right: LibraryContract, hypothesis_kind: str) -> list[str]:
    refs = [
        _recipe_ref(contract, recipe.id)
        for contract in (left, right)
        for recipe in contract.recipes
        if recipe.hypothesis_kind == hypothesis_kind
    ]
    return sorted(dict.fromkeys(refs))


def _evidence_refs(
    left_use: ResourceUse,
    right_use: ResourceUse,
    assumptions: list[AssumptionViolation],
    controls: list[str],
    signal_refs: list[str],
    recipe_refs: list[str],
    left: LibraryContract,
    right: LibraryContract,
) -> list[str]:
    evidence = set(left_use.evidence) | set(right_use.evidence)
    assumption_lookup = {
        _assumption_ref(contract, item.id): item
        for contract in (left, right)
        for item in contract.assumptions
    }
    control_lookup = {
        _control_ref(contract, item.id): item
        for contract in (left, right)
        for item in contract.controls
    }
    signal_lookup = {
        _signal_ref(contract, item.id): item
        for contract in (left, right)
        for item in contract.signals
    }
    recipe_lookup = {
        _recipe_ref(contract, item.id): item
        for contract in (left, right)
        for item in contract.recipes
    }
    for violation in assumptions:
        evidence.update(assumption_lookup[violation.assumption].evidence)
    for ref in controls:
        evidence.update(control_lookup[ref].evidence)
    for ref in signal_refs:
        evidence.update(signal_lookup[ref].evidence)
    for ref in recipe_refs:
        evidence.update(recipe_lookup[ref].evidence)
    return sorted(evidence)


def derive_shared_resources(contracts: list[LibraryContract]) -> list[SharedResourceItem]:
    grouped: dict[tuple[str, str], list[tuple[LibraryContract, ResourceUse]]] = defaultdict(list)
    for contract in contracts:
        for use in contract.uses:
            grouped[(use.resource, use.phase)].append((contract, use))

    shared_resources: list[SharedResourceItem] = []
    for (resource, phase), entries in sorted(grouped.items(), key=lambda item: item[0]):
        contracts_in_group = sorted({contract.contract for contract, _ in entries})
        if len(contracts_in_group) < 2:
            continue
        shared_resources.append(
            SharedResourceItem(
                resource=resource,
                phase=phase,
                contracts=contracts_in_group,
                components=sorted({_component_ref(contract, use.by) for contract, use in entries}),
                uses=sorted({_use_ref(contract, use.id) for contract, use in entries}),
                evidence=sorted({evidence for _, use in entries for evidence in use.evidence}),
            )
        )
    return shared_resources


def derive_conflict_hypotheses(contracts: list[LibraryContract]) -> list[ConflictHypothesis]:
    hypotheses: list[ConflictHypothesis] = []

    for left, right in combinations(sorted(contracts, key=lambda item: item.contract.lower()), 2):
        grouped_pairs: dict[tuple[str, str], list[tuple[ResourceUse, ResourceUse]]] = defaultdict(
            list
        )
        for left_use in left.uses:
            for right_use in right.uses:
                if left_use.resource != right_use.resource:
                    continue
                if not _phase_overlaps(left_use.phase, right_use.phase):
                    continue
                grouped_pairs[
                    (left_use.resource, _merged_phase(left_use.phase, right_use.phase))
                ].append((left_use, right_use))

        for (resource, phase), overlap_pairs in sorted(
            grouped_pairs.items(), key=lambda item: item[0]
        ):
            hypothesis_kind = RESOURCE_TO_HYPOTHESIS_KIND[resource]
            overlaps = [
                HypothesisOverlap(
                    uses=sorted([_use_ref(left, left_use.id), _use_ref(right, right_use.id)]),
                    components=sorted(
                        [
                            _component_ref_for_use(left, left_use),
                            _component_ref_for_use(right, right_use),
                        ]
                    ),
                    contracts=sorted([left.contract, right.contract]),
                )
                for left_use, right_use in overlap_pairs
            ]
            use_refs = sorted({ref for overlap in overlaps for ref in overlap.uses})
            component_refs = sorted({ref for overlap in overlaps for ref in overlap.components})
            controls = sorted(
                {
                    _control_ref(contract, control.id)
                    for contract in (left, right)
                    for control in _controls_for_resource(contract, resource)
                }
            )
            control_collisions = _control_collisions(left, right, resource)
            assumption_violations = _assumption_violations(left, right, resource, overlap_pairs)
            signals = _signal_refs(left, right, resource, component_refs, use_refs)
            recipes = _recipe_refs(left, right, hypothesis_kind)

            shared_overlap = any(
                _use_share_allows_overlap(left_use) and _use_share_allows_overlap(right_use)
                for left_use, right_use in overlap_pairs
            )
            oversubscribed = resource == "cpu.compute" and _cpu_compute_oversubscribed(
                overlap_pairs
            )
            placement_conflict = (
                resource == "cpu.placement"
                and any(
                    left_use.share in {"exclusive", "partitioned"}
                    or right_use.share in {"exclusive", "partitioned"}
                    for left_use, right_use in overlap_pairs
                )
            ) or (
                resource == "cpu.placement"
                and any(
                    item.resource == "cpu.placement"
                    and str(item.predicate.get("kind", "")).strip() == "placement_compatible"
                    for item in left.assumptions + right.assumptions
                )
            )

            if not any(
                [
                    shared_overlap,
                    oversubscribed,
                    assumption_violations,
                    control_collisions,
                    placement_conflict,
                ]
            ):
                continue

            if not signals:
                recipe_signal_refs = {
                    _signal_ref(contract, signal_id)
                    for contract in (left, right)
                    for recipe in contract.recipes
                    if _recipe_ref(contract, recipe.id) in recipes
                    for signal_id in recipe.collect
                    if signal_id in {item.id for item in contract.signals}
                }
                signals = sorted(recipe_signal_refs)
            if not signals:
                continue

            evidence = _evidence_refs(
                overlap_pairs[0][0],
                overlap_pairs[0][1],
                assumption_violations,
                controls,
                signals,
                recipes,
                left,
                right,
            )

            hypotheses.append(
                ConflictHypothesis(
                    id=f"h_{_slugify(hypothesis_kind)}_{_slugify(left.contract)}_{_slugify(right.contract)}_{_slugify(phase)}",
                    kind=hypothesis_kind,
                    resource=resource,
                    phase=phase,
                    contracts=sorted([left.contract, right.contract]),
                    components=component_refs,
                    uses=use_refs,
                    basis=HypothesisBasis(
                        overlaps=overlaps,
                        assumption_violations=assumption_violations,
                        control_collisions=control_collisions,
                    ),
                    validate=HypothesisValidation(collect=signals),
                    candidate_actions=controls,
                    recipes=recipes,
                    evidence=evidence,
                )
            )
    return sorted(hypotheses, key=lambda item: (item.resource, item.phase, item.id))


def compose_contracts(
    contract_name: str,
    contracts: list[LibraryContract],
) -> ComposedContract:
    ordered_contracts = sorted(contracts, key=lambda item: item.contract.lower())
    instantiated_contracts, instances = _instantiate_contracts(ordered_contracts)
    selected_contracts = [instance.instance_id for instance in instances]
    composed = ComposedContract(
        contract=contract_name,
        type="composed",
        version="lpcl/v1",
        selected_contracts=selected_contracts,
        contract_instances=instances,
        shared_resources=derive_shared_resources(instantiated_contracts),
        hypotheses=derive_conflict_hypotheses(instantiated_contracts),
    )
    verified = assert_valid_contract(composed)
    if not isinstance(verified, ComposedContract):
        raise TypeError("Expected a composed contract after validation")
    return verified
