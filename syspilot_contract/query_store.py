from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping, cast

from syspilot_contract.checker import load_contract
from syspilot_contract.compose_contracts import compose_contracts as compose_contract_models
from syspilot_contract.schemas import (
    Assumption,
    Component,
    ComposedContract,
    Control,
    LibraryContract,
    Recipe,
    ResourceUse,
    Signal,
    RESOURCE_TO_HYPOTHESIS_KIND,
)


logger = logging.getLogger(__name__)


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_\.]+", value.lower()) if token]


def _score(text: str, terms: list[str]) -> int:
    haystack = set(_tokenize(text))
    return sum(1 for term in terms if term in haystack)


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


def _control_activation_value(control: Control) -> Any | None:
    if control.effect.kind == "enable":
        return True
    if control.effect.kind == "disable":
        return False
    if control.effect.kind == "set" and control.effect.value != "input":
        return control.effect.value
    return None


class ContractStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._contracts: dict[str, LibraryContract | ComposedContract] = {}
        self._contract_aliases: dict[str, str] = {}
        self._discover()

    @staticmethod
    def _normalize_contract_key(value: str) -> str:
        return value.strip().lower()

    def _register_alias(self, alias: str, bundle_id: str) -> None:
        normalized = self._normalize_contract_key(alias)
        if normalized:
            self._contract_aliases.setdefault(normalized, bundle_id)

    def _discover(self) -> None:
        if not self.root.exists():
            return
        for bundle_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            contract_path = bundle_dir / "contract.yaml"
            if not contract_path.exists():
                continue
            try:
                loaded = load_contract(contract_path)
                self._contracts[bundle_dir.name] = loaded
                self._register_alias(bundle_dir.name, bundle_dir.name)
                self._register_alias(loaded.contract, bundle_dir.name)
            except Exception as exc:
                logger.warning("Skipping invalid contract bundle %s: %s", contract_path, exc)

    def get_contract(self, contract_id: str) -> LibraryContract | ComposedContract:
        if contract_id in self._contracts:
            return self._contracts[contract_id]
        alias = self._contract_aliases[self._normalize_contract_key(contract_id)]
        return self._contracts[alias]

    def list_contracts(self) -> list[str]:
        return sorted(self._contracts)

    def _instance_contracts_for_composed(self, composed: ComposedContract) -> list[LibraryContract]:
        libraries: list[LibraryContract] = []
        for instance in composed.contract_instances:
            try:
                contract = self.get_contract(instance.contract_id)
            except KeyError as exc:
                raise KeyError(
                    f"Composed contract '{composed.contract}' references missing contract '{instance.contract_id}' for instance '{instance.instance_id}'"
                ) from exc
            if not isinstance(contract, LibraryContract):
                raise TypeError(
                    f"Composed contract '{composed.contract}' instance '{instance.instance_id}' target '{instance.contract_id}' is not a library contract"
                )
            libraries.append(contract.model_copy(update={"contract": instance.instance_id}))
        return libraries

    def _libraries_for_contract_id(self, contract_id: str) -> list[LibraryContract]:
        contract = self.get_contract(contract_id)
        if isinstance(contract, LibraryContract):
            return [contract]
        return self._instance_contracts_for_composed(contract)

    def compose_contracts(
        self, contract_ids: list[str], app_name: str | None = None
    ) -> ComposedContract:
        libraries, composed = self._resolve_context_inputs(contract_ids)
        if composed is not None and app_name is None:
            return composed
        return compose_contract_models(app_name or "context_composed", libraries)

    def list_components(self, contract_id: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for contract in self._libraries_for_contract_id(contract_id):
            for item in contract.components:
                rows.append({"contract": contract.contract, **item.model_dump(mode="json")})
        return rows

    def list_uses(
        self,
        contract_id: str,
        resource: str | None = None,
        component: str | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for contract in self._libraries_for_contract_id(contract_id):
            for item in contract.uses:
                if resource is not None and item.resource != resource:
                    continue
                if component is not None and item.by != component:
                    continue
                rows.append({"contract": contract.contract, **item.model_dump(mode="json")})
        return rows

    def list_controls(
        self,
        contract_id: str,
        target: str | None = None,
        resource: str | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for contract in self._libraries_for_contract_id(contract_id):
            for item in contract.controls:
                if target is not None and item.target != target:
                    continue
                if resource is not None and resource not in _resources_for_target(
                    contract, item.target
                ):
                    continue
                payload = item.model_dump(mode="json")
                payload["activation_value"] = _control_activation_value(item)
                rows.append({"contract": contract.contract, **payload})
        return rows

    def list_signals(
        self,
        contract_id: str,
        target: str | None = None,
        resource: str | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for contract in self._libraries_for_contract_id(contract_id):
            for item in contract.signals:
                if target is not None and item.target != target:
                    continue
                if resource is not None and resource not in _resources_for_target(
                    contract, item.target
                ):
                    continue
                rows.append({"contract": contract.contract, **item.model_dump(mode="json")})
        return rows

    def list_assumptions(
        self, contract_id: str, resource: str | None = None
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for contract in self._libraries_for_contract_id(contract_id):
            for item in contract.assumptions:
                if resource is not None and item.resource != resource:
                    continue
                rows.append({"contract": contract.contract, **item.model_dump(mode="json")})
        return rows

    def list_recipes(
        self,
        contract_id: str,
        hypothesis_kind: str | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for contract in self._libraries_for_contract_id(contract_id):
            for item in contract.recipes:
                if hypothesis_kind is not None and item.hypothesis_kind != hypothesis_kind:
                    continue
                rows.append(
                    {"contract": contract.contract, **item.model_dump(mode="json", by_alias=True)}
                )
        return rows

    def list_hypotheses(
        self,
        composed_contract_id: str,
        resource: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, object]]:
        composed = self.compose_contracts([composed_contract_id])
        rows: list[dict[str, object]] = []
        for item in composed.hypotheses:
            if resource is not None and item.resource != resource:
                continue
            if kind is not None and item.kind != kind:
                continue
            rows.append(item.model_dump(mode="json", by_alias=True))
        return rows

    def _resolve_context_inputs(
        self,
        contract_ids: list[str],
    ) -> tuple[list[LibraryContract], ComposedContract | None]:
        library_contracts: list[LibraryContract] = []
        composed_contracts: list[ComposedContract] = []
        for contract_id in contract_ids:
            contract = self.get_contract(contract_id)
            if isinstance(contract, LibraryContract):
                library_contracts.append(contract)
            else:
                composed_contracts.append(contract)

        if len(composed_contracts) > 1:
            raise ValueError("Only one composed contract can be queried at a time")

        composed = composed_contracts[0] if composed_contracts else None
        if composed is not None:
            libraries = self._instance_contracts_for_composed(composed)
            explicit_library_map = {contract.contract: contract for contract in library_contracts}
            for contract in explicit_library_map.values():
                if contract.contract not in {item.contract for item in libraries}:
                    libraries.append(contract)
            recomposed = compose_contract_models(composed.contract, libraries)
            return libraries, recomposed

        libraries = library_contracts
        if len(libraries) >= 2:
            composed = compose_contract_models("context_composed", libraries)
            libraries = self._instance_contracts_for_composed(composed)
        return libraries, composed

    @staticmethod
    def _require_component_ref(
        ref: str,
        ref_map: Mapping[str, tuple[LibraryContract, Component]],
    ) -> tuple[LibraryContract, Component]:
        if ref not in ref_map:
            raise KeyError(f"Unresolved component ref: {ref}")
        return ref_map[ref]

    @staticmethod
    def _require_use_ref(
        ref: str,
        ref_map: Mapping[str, tuple[LibraryContract, ResourceUse]],
    ) -> tuple[LibraryContract, ResourceUse]:
        if ref not in ref_map:
            raise KeyError(f"Unresolved use ref: {ref}")
        return ref_map[ref]

    @staticmethod
    def _require_assumption_ref(
        ref: str,
        ref_map: Mapping[str, tuple[LibraryContract, Assumption]],
    ) -> tuple[LibraryContract, Assumption]:
        if ref not in ref_map:
            raise KeyError(f"Unresolved assumption ref: {ref}")
        return ref_map[ref]

    @staticmethod
    def _require_control_ref(
        ref: str,
        ref_map: Mapping[str, tuple[LibraryContract, Control]],
    ) -> tuple[LibraryContract, Control]:
        if ref not in ref_map:
            raise KeyError(f"Unresolved control ref: {ref}")
        return ref_map[ref]

    @staticmethod
    def _require_signal_ref(
        ref: str,
        ref_map: Mapping[str, tuple[LibraryContract, Signal]],
    ) -> tuple[LibraryContract, Signal]:
        if ref not in ref_map:
            raise KeyError(f"Unresolved signal ref: {ref}")
        return ref_map[ref]

    @staticmethod
    def _require_recipe_ref(
        ref: str,
        ref_map: Mapping[str, tuple[LibraryContract, Recipe]],
    ) -> tuple[LibraryContract, Recipe]:
        if ref not in ref_map:
            raise KeyError(f"Unresolved recipe ref: {ref}")
        return ref_map[ref]

    def build_agent_context(
        self,
        contract_ids: list[str],
        goal: str | None = None,
        symptom: str | None = None,
        max_items_per_section: int = 5,
    ) -> dict[str, object]:
        libraries, composed = self._resolve_context_inputs(contract_ids)
        terms = _tokenize(" ".join(value for value in (goal, symptom) if value))

        component_ref_map = {
            f"{contract.contract}.{item.id}": (contract, item)
            for contract in libraries
            for item in contract.components
        }
        use_ref_map = {
            f"{contract.contract}.{item.id}": (contract, item)
            for contract in libraries
            for item in contract.uses
        }
        assumption_ref_map = {
            f"{contract.contract}.{item.id}": (contract, item)
            for contract in libraries
            for item in contract.assumptions
        }
        control_ref_map = {
            f"{contract.contract}.{item.id}": (contract, item)
            for contract in libraries
            for item in contract.controls
        }
        signal_ref_map = {
            f"{contract.contract}.{item.id}": (contract, item)
            for contract in libraries
            for item in contract.signals
        }
        recipe_ref_map = {
            f"{contract.contract}.{item.id}": (contract, item)
            for contract in libraries
            for item in contract.recipes
        }

        relevant_components = [
            {
                "contract": contract.contract,
                **item.model_dump(mode="json"),
            }
            for contract in libraries
            for item in contract.components
        ]

        shared_resources: list[dict[str, object]] = []
        likely_hypotheses: list[dict[str, object]] = []
        hypothesis_profiles: list[dict[str, object]] = []
        conflicts: list[dict[str, object]] = []

        if composed is not None:
            for item in composed.shared_resources:
                resolved_components = [
                    {
                        "ref": ref,
                        "contract": self._require_component_ref(ref, component_ref_map)[0].contract,
                        "component": self._require_component_ref(ref, component_ref_map)[
                            1
                        ].model_dump(mode="json"),
                    }
                    for ref in item.components
                ]
                resolved_uses = [
                    {
                        "ref": ref,
                        "contract": self._require_use_ref(ref, use_ref_map)[0].contract,
                        "use": self._require_use_ref(ref, use_ref_map)[1].model_dump(mode="json"),
                    }
                    for ref in item.uses
                ]
                shared_resources.append(
                    {
                        "resource": item.resource,
                        "phase": item.phase,
                        "contracts": item.contracts,
                        "components": resolved_components,
                        "uses": resolved_uses,
                        "evidence": item.evidence,
                    }
                )

            for item in composed.hypotheses:
                resolved_components = [
                    {
                        "ref": ref,
                        "contract": self._require_component_ref(ref, component_ref_map)[0].contract,
                        "component": self._require_component_ref(ref, component_ref_map)[
                            1
                        ].model_dump(mode="json"),
                    }
                    for ref in item.components
                ]
                resolved_uses = [
                    {
                        "ref": ref,
                        "contract": self._require_use_ref(ref, use_ref_map)[0].contract,
                        "use": self._require_use_ref(ref, use_ref_map)[1].model_dump(mode="json"),
                    }
                    for ref in item.uses
                ]
                resolved_overlaps = []
                for overlap in item.basis.overlaps:
                    overlap_uses = [
                        {
                            "ref": ref,
                            "contract": self._require_use_ref(ref, use_ref_map)[0].contract,
                            "use": self._require_use_ref(ref, use_ref_map)[1].model_dump(
                                mode="json"
                            ),
                        }
                        for ref in overlap.uses
                    ]
                    resolved_overlaps.append(
                        {
                            "contracts": overlap.contracts,
                            "components": overlap.components,
                            "uses": overlap_uses,
                        }
                    )
                resolved_assumption_violations = []
                for violation in item.basis.assumption_violations:
                    contract, assumption = self._require_assumption_ref(
                        violation.assumption, assumption_ref_map
                    )
                    resolved_assumption_violations.append(
                        {
                            "ref": violation.assumption,
                            "contract": contract.contract,
                            "assumption": assumption.model_dump(mode="json"),
                            "conflicting_uses": violation.conflicting_uses,
                        }
                    )
                resolved_control_collisions = []
                for collision in item.basis.control_collisions:
                    controls = []
                    for ref in collision.controls:
                        contract, control = self._require_control_ref(ref, control_ref_map)
                        control_payload = control.model_dump(mode="json")
                        control_payload["activation_value"] = _control_activation_value(control)
                        controls.append(
                            {
                                "ref": ref,
                                "contract": contract.contract,
                                "control": control_payload,
                            }
                        )
                    resolved_control_collisions.append(
                        {
                            "resource": collision.resource,
                            "dimension": collision.dimension,
                            "controls": controls,
                        }
                    )
                resolved_validation_signals = []
                for ref in item.validation.collect:
                    contract, signal = self._require_signal_ref(ref, signal_ref_map)
                    resolved_validation_signals.append(
                        {
                            "ref": ref,
                            "contract": contract.contract,
                            "signal": signal.model_dump(mode="json"),
                        }
                    )
                resolved_candidate_controls = []
                for ref in item.candidate_actions:
                    if ref not in control_ref_map:
                        continue
                    contract, control = self._require_control_ref(ref, control_ref_map)
                    control_payload = control.model_dump(mode="json")
                    control_payload["activation_value"] = _control_activation_value(control)
                    resolved_candidate_controls.append(
                        {
                            "ref": ref,
                            "contract": contract.contract,
                            "control": control_payload,
                        }
                    )
                resolved_candidate_recipes = []
                for ref in item.recipes:
                    contract, recipe = self._require_recipe_ref(ref, recipe_ref_map)
                    resolved_candidate_recipes.append(
                        {
                            "ref": ref,
                            "contract": contract.contract,
                            "recipe": recipe.model_dump(mode="json", by_alias=True),
                        }
                    )

                validation_probe_names = {
                    str(entry[1].probe.name)
                    for entry in [
                        self._require_signal_ref(ref, signal_ref_map)
                        for ref in item.validation.collect
                    ]
                }
                report_controls = [
                    control_entry
                    for control_entry in resolved_candidate_controls
                    if isinstance(control_entry.get("control"), dict)
                    and control_entry["control"].get("kind") == "env"
                    and control_entry["control"].get("name") in validation_probe_names
                    and control_entry["control"].get("activation_value") is not None
                ]

                signal_titles = [
                    entry["signal"]["metric"]
                    for entry in resolved_validation_signals
                    if isinstance(entry.get("signal"), dict)
                ]
                control_titles = [
                    entry["control"]["name"]
                    for entry in resolved_candidate_controls
                    if isinstance(entry.get("control"), dict)
                ]
                recipe_titles = [
                    entry["recipe"]["id"]
                    for entry in resolved_candidate_recipes
                    if isinstance(entry.get("recipe"), dict)
                ]
                text = (
                    f"{item.kind} {item.resource} {item.phase} components {' '.join(item.components)} "
                    f"signals {' '.join(signal_titles)} controls {' '.join(control_titles)} recipes {' '.join(recipe_titles)}"
                )
                payload: dict[str, object] = {
                    "hypothesis_id": item.id,
                    "kind": item.kind,
                    "resource": item.resource,
                    "phase": item.phase,
                    "contracts": item.contracts,
                    "components": resolved_components,
                    "uses": resolved_uses,
                    "basis": {
                        "overlaps": resolved_overlaps,
                        "assumption_violations": resolved_assumption_violations,
                        "control_collisions": resolved_control_collisions,
                    },
                    "validation_signals": resolved_validation_signals,
                    "report_controls": report_controls,
                    "candidate_controls": resolved_candidate_controls,
                    "candidate_recipes": resolved_candidate_recipes,
                    "evidence": item.evidence,
                }
                hypothesis_profiles.append(payload)
                likely_hypotheses.append(payload)
                conflicts.append(
                    {
                        "section": "hypotheses",
                        "item_id": item.id,
                        "title": f"{item.kind} / {item.resource}",
                        "text": text,
                        "keywords": control_titles + signal_titles + recipe_titles,
                        "resource_domains": [item.resource],
                        "contracts": item.contracts,
                        "components": resolved_components,
                        "uses": resolved_uses,
                        "validation_signals": resolved_validation_signals,
                        "candidate_controls": resolved_candidate_controls,
                        "candidate_recipes": resolved_candidate_recipes,
                        "evidence": item.evidence,
                        "score": _score(text, terms),
                    }
                )

        profile_first: list[dict[str, object]] = []
        signals_to_collect: list[dict[str, object]] = []
        for contract in libraries:
            for signal in contract.signals:
                resources = _resources_for_target(contract, signal.target)
                text = f"{signal.metric} {signal.indicates} {signal.target} {signal.probe.kind} {signal.probe.name} {' '.join(resources)}"
                payload: dict[str, object] = {
                    "section": "signals",
                    "item_id": signal.id,
                    "title": signal.metric,
                    "text": text,
                    "keywords": [signal.metric, signal.indicates, signal.target, signal.probe.name],
                    "resource_domains": resources,
                    "evidence": signal.evidence,
                    "library": contract.contract,
                    "signal_ref": f"{contract.contract}.{signal.id}",
                    "signal": signal.model_dump(mode="json"),
                    "score": _score(text, terms),
                }
                profile_first.append(payload)
                signals_to_collect.append(payload)

        candidate_actions: list[dict[str, object]] = []
        candidate_controls: list[dict[str, object]] = []
        for contract in libraries:
            for control in contract.controls:
                resources = _resources_for_target(contract, control.target)
                control_payload = control.model_dump(mode="json")
                control_payload["activation_value"] = _control_activation_value(control)
                text = f"{control.name} {control.kind} {control.type} target={control.target} effect={control.effect.kind} {' '.join(resources)}"
                payload: dict[str, object] = {
                    "section": "controls",
                    "item_id": control.id,
                    "title": control.name,
                    "text": text,
                    "keywords": [
                        control.name,
                        control.kind,
                        control.type,
                        control.target,
                        control.effect.kind,
                    ],
                    "resource_domains": resources,
                    "evidence": control.evidence,
                    "library": contract.contract,
                    "control_ref": f"{contract.contract}.{control.id}",
                    "control": control_payload,
                    "score": _score(text, terms),
                }
                candidate_actions.append(payload)
                candidate_controls.append(payload)

        recipes: list[dict[str, object]] = []
        validation: list[dict[str, object]] = []
        for contract in libraries:
            for recipe in contract.recipes:
                text = f"{recipe.hypothesis_kind} {' '.join(recipe.collect)} {' '.join(recipe.actions)}"
                payload: dict[str, object] = {
                    "section": "recipes",
                    "item_id": recipe.id,
                    "title": recipe.hypothesis_kind,
                    "text": text,
                    "keywords": recipe.collect + recipe.actions,
                    "evidence": recipe.evidence,
                    "library": contract.contract,
                    "recipe_ref": f"{contract.contract}.{recipe.id}",
                    "recipe": recipe.model_dump(mode="json", by_alias=True),
                    "score": _score(text, terms),
                }
                recipes.append(payload)
                validation.append(payload)
                candidate_actions.append(
                    {
                        **payload,
                        "section": "recipe_actions",
                        "title": recipe.id,
                    }
                )

        if composed is not None:
            for item in composed.hypotheses:
                validation.append(
                    {
                        "section": "hypothesis_validation",
                        "item_id": item.id,
                        "title": item.id,
                        "text": f"collect {' '.join(item.validation.collect)}",
                        "keywords": item.validation.collect,
                        "evidence": item.evidence,
                        "library": ", ".join(item.contracts),
                        "signals": [
                            {
                                "ref": ref,
                                "contract": self._require_signal_ref(ref, signal_ref_map)[
                                    0
                                ].contract,
                                "signal": self._require_signal_ref(ref, signal_ref_map)[
                                    1
                                ].model_dump(mode="json"),
                            }
                            for ref in item.validation.collect
                        ],
                        "score": _score(" ".join(item.validation.collect), terms),
                    }
                )

        def _rank(items: list[dict[str, object]]) -> list[dict[str, object]]:
            def _score_value(item: dict[str, object]) -> int:
                value = item.get("score", 0)
                return value if isinstance(value, int) else 0

            ranked = sorted(
                items,
                key=lambda item: (
                    -_score_value(item),
                    str(item.get("library", "")),
                    str(item.get("section", "")),
                    str(item.get("item_id", "")),
                ),
            )
            return [
                {key: value for key, value in item.items() if key != "score"}
                for item in ranked[:max_items_per_section]
            ]

        labels = [contract.contract for contract in libraries]
        summary = (
            f"Structured LPCL v1 context for {', '.join(sorted(labels))}. "
            f"Goal={goal or 'unspecified'}. Symptom={symptom or 'unspecified'}."
        )
        relevant_resources = sorted(
            {item.resource for contract in libraries for item in contract.uses}
        )
        return {
            "summary": summary,
            "relevant_components": relevant_components,
            "relevant_resources": relevant_resources,
            "likely_hypotheses": likely_hypotheses,
            "signals_to_collect": signals_to_collect,
            "candidate_controls": candidate_controls,
            "recipes": recipes,
            "shared_resources": shared_resources,
            "hypothesis_profiles": hypothesis_profiles,
            "profile_first": _rank(profile_first),
            "candidate_actions": _rank(candidate_actions),
            "conflicts": _rank(conflicts),
            "validation": _rank(validation),
        }
