from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Mapping

from syspilot_contract.checker import load_contract
from syspilot_contract.compose_contracts import compose_contracts
from syspilot_contract.schemas import (
    Assumption,
    ComposedContract,
    Control,
    ContractInstance,
    LibraryContract,
    Recipe,
    ResourceUse,
    Signal,
)


logger = logging.getLogger(__name__)


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_\.]+", value.lower()) if token]


def _score(text: str, terms: list[str]) -> int:
    haystack = set(_tokenize(text))
    return sum(1 for term in terms if term in haystack)


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

    def _instance_contracts_for_composed(
        self,
        composed: ComposedContract | None,
    ) -> list[LibraryContract]:
        if composed is None:
            return []
        libraries: list[LibraryContract] = []
        for instance in composed.contract_instances:
            try:
                contract = self.get_contract(instance.contract_id)
            except KeyError as exc:
                raise KeyError(
                    f"Composed contract '{composed.contract}' references missing contract '{instance.contract_id}' for instance '{instance.instance_id}'"
                ) from exc
            if isinstance(contract, LibraryContract):
                libraries.append(contract.model_copy(update={"contract": instance.instance_id}))
            else:
                raise TypeError(
                    f"Composed contract '{composed.contract}' instance '{instance.instance_id}' target '{instance.contract_id}' is not a library contract"
                )
        return libraries

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
        else:
            libraries = library_contracts
        if composed is None and len(libraries) >= 2:
            composed = compose_contracts("context_composed", libraries)
            libraries = self._instance_contracts_for_composed(composed)
        return libraries, composed

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

        use_ref_map = {
            f"{contract.contract}.{use.id}": (contract, use)
            for contract in libraries
            for use in contract.uses
        }
        assumption_ref_map = {
            f"{contract.contract}.{assumption.id}": (contract, assumption)
            for contract in libraries
            for assumption in contract.assumptions
        }

        control_ref_map = {
            f"{contract.contract}.{control.id}": (contract, control)
            for contract in libraries
            for control in contract.controls
        }
        signal_ref_map = {
            f"{contract.contract}.{signal.id}": (contract, signal)
            for contract in libraries
            for signal in contract.signals
        }
        recipe_ref_map = {
            f"{contract.contract}.{recipe.id}": (contract, recipe)
            for contract in libraries
            for recipe in contract.recipes
        }

        shared_resources: list[dict[str, object]] = []
        if composed is not None:
            for item in composed.shared_resources:
                resolved_uses = []
                for ref in item.use_refs:
                    contract, use = self._require_use_ref(ref, use_ref_map)
                    resolved_uses.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "use": use.model_dump(mode="json"),
                        }
                    )
                shared_resources.append(
                    {
                        "resource": item.resource,
                        "phase": item.phase,
                        "libraries": item.libraries,
                        "use_refs": item.use_refs,
                        "uses": resolved_uses,
                        "source_refs": item.source_refs,
                    }
                )

        observability: list[dict[str, object]] = []
        for contract in libraries:
            for signal in contract.signals:
                text = (
                    f"{signal.metric} {signal.indicates} {signal.resource} "
                    f"{signal.probe.kind} {signal.probe.target}"
                )
                observability.append(
                    {
                        "section": "signals",
                        "item_id": signal.id,
                        "title": signal.metric,
                        "text": text,
                        "keywords": [
                            signal.metric,
                            signal.indicates,
                            signal.resource,
                            signal.probe.kind,
                            signal.probe.target,
                        ],
                        "resource_domains": [signal.resource],
                        "source_refs": signal.source_refs,
                        "library": contract.contract,
                        "signal_ref": f"{contract.contract}.{signal.id}",
                        "signal": signal.model_dump(mode="json"),
                        "score": _score(text, terms),
                    }
                )

        candidate_actions: list[dict[str, object]] = []
        for contract in libraries:
            for control in contract.controls:
                related_recipe_actions = [
                    action
                    for recipe in contract.recipes
                    if control.id in recipe.control_refs
                    for action in recipe.actions
                ]
                activation = (
                    f" activation={control.activation_value}"
                    if control.activation_value is not None
                    else ""
                )
                text = (
                    f"{control.name} type={control.type}{activation} {control.effect.op} {control.effect.resource} "
                    f"scope {control.scope} {' '.join(related_recipe_actions)}"
                )
                candidate_actions.append(
                    {
                        "section": "controls",
                        "item_id": control.id,
                        "title": control.name,
                        "text": text,
                        "keywords": [
                            control.name,
                            control.type,
                            control.effect.resource,
                            control.effect.op,
                        ]
                        + control.allowed_values,
                        "resource_domains": [control.effect.resource],
                        "source_refs": control.source_refs,
                        "library": contract.contract,
                        "control_ref": f"{contract.contract}.{control.id}",
                        "control": control.model_dump(mode="json"),
                        "score": _score(text, terms),
                    }
                )

        conflicts: list[dict[str, object]] = []
        if composed is not None:
            for item in composed.hypotheses:
                resolved_overlaps = []
                for overlap in item.basis.overlaps:
                    resolved_overlap_uses = []
                    for ref in overlap.uses:
                        contract, use = self._require_use_ref(ref, use_ref_map)
                        resolved_overlap_uses.append(
                            {
                                "ref": ref,
                                "library": contract.contract,
                                "use": use.model_dump(mode="json"),
                            }
                        )
                    resolved_overlaps.append(
                        {
                            "use_refs": overlap.uses,
                            "uses": resolved_overlap_uses,
                        }
                    )
                resolved_assumptions = []
                for ref in item.basis.violated_assumptions:
                    contract, assumption = self._require_assumption_ref(ref, assumption_ref_map)
                    resolved_assumptions.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "assumption": assumption.model_dump(mode="json"),
                        }
                    )
                resolved_colliding_controls = []
                for ref in item.basis.colliding_controls:
                    contract, control = self._require_control_ref(ref, control_ref_map)
                    resolved_colliding_controls.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "control": control.model_dump(mode="json"),
                        }
                    )
                control_names: list[str] = []
                for ref in item.basis.colliding_controls:
                    _, control = self._require_control_ref(ref, control_ref_map)
                    control_names.append(control.name)
                signal_titles: list[str] = []
                for ref in item.validation.collect:
                    _, signal = self._require_signal_ref(ref, signal_ref_map)
                    signal_titles.append(signal.metric)
                recipe_titles: list[str] = []
                for ref in item.candidate_recipe_refs:
                    _, recipe = self._require_recipe_ref(ref, recipe_ref_map)
                    recipe_titles.append(recipe.id)
                resolved_validation_signals = []
                for ref in item.validation.collect:
                    contract, signal = self._require_signal_ref(ref, signal_ref_map)
                    resolved_validation_signals.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "signal": signal.model_dump(mode="json"),
                        }
                    )
                resolved_candidate_controls = []
                for ref in item.candidate_control_refs:
                    contract, control = self._require_control_ref(ref, control_ref_map)
                    resolved_candidate_controls.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "control": control.model_dump(mode="json"),
                        }
                    )
                resolved_candidate_recipes = []
                for ref in item.candidate_recipe_refs:
                    contract, recipe = self._require_recipe_ref(ref, recipe_ref_map)
                    resolved_candidate_recipes.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "recipe": recipe.model_dump(mode="json"),
                        }
                    )
                text = (
                    f"{item.kind} {item.resource} {item.phase} controls {' '.join(control_names)} "
                    f"signals {' '.join(signal_titles)} recipes {' '.join(recipe_titles)}"
                )
                conflicts.append(
                    {
                        "section": "hypotheses",
                        "item_id": item.id,
                        "title": f"{item.resource} / {item.phase}",
                        "text": text,
                        "keywords": control_names + signal_titles + recipe_titles,
                        "resource_domains": [item.resource],
                        "source_refs": item.source_refs,
                        "libraries": item.libraries,
                        "resolved_overlaps": resolved_overlaps,
                        "violated_assumptions": resolved_assumptions,
                        "colliding_controls": resolved_colliding_controls,
                        "validation_signals": resolved_validation_signals,
                        "candidate_controls": resolved_candidate_controls,
                        "candidate_recipes": resolved_candidate_recipes,
                        "score": _score(text, terms),
                    }
                )
                for ref in item.candidate_control_refs:
                    contract, control = self._require_control_ref(ref, control_ref_map)
                    candidate_actions.append(
                        {
                            "section": "hypothesis_controls",
                            "item_id": item.id,
                            "title": control.name,
                            "text": (
                                f"{control.name} type={control.type} {control.effect.op} "
                                f"{control.effect.resource} scope {control.scope}"
                            ),
                            "keywords": [control.name, control.type, control.effect.resource],
                            "resource_domains": [item.resource],
                            "source_refs": control.source_refs,
                            "library": contract.contract,
                            "control_ref": ref,
                            "control": control.model_dump(mode="json"),
                            "score": _score(control.name, terms),
                        }
                    )
                for ref in item.candidate_recipe_refs:
                    contract, recipe = self._require_recipe_ref(ref, recipe_ref_map)
                    text = (
                        f"{recipe.hypothesis_kind} {' '.join(recipe.actions)} "
                        f"{' '.join(recipe.collect)} {' '.join(recipe.control_refs)}"
                    )
                    candidate_actions.append(
                        {
                            "section": "hypothesis_recipes",
                            "item_id": item.id,
                            "title": recipe.id,
                            "text": text,
                            "keywords": recipe.collect + recipe.control_refs,
                            "resource_domains": [item.resource],
                            "source_refs": recipe.source_refs,
                            "library": contract.contract,
                            "recipe_ref": ref,
                            "recipe": recipe.model_dump(mode="json"),
                            "score": _score(text, terms),
                        }
                    )

        validation: list[dict[str, object]] = []
        for contract in libraries:
            for recipe in contract.recipes:
                text = f"{recipe.accept_if} {recipe.rollback_if} {' '.join(recipe.collect)} {' '.join(recipe.actions)}"
                validation.append(
                    {
                        "section": "recipes",
                        "item_id": recipe.id,
                        "title": recipe.hypothesis_kind,
                        "text": text,
                        "keywords": recipe.collect,
                        "source_refs": recipe.source_refs,
                        "library": contract.contract,
                        "recipe_ref": f"{contract.contract}.{recipe.id}",
                        "recipe": recipe.model_dump(mode="json"),
                        "score": _score(text, terms),
                    }
                )
        if composed is not None:
            for item in composed.hypotheses:
                signal_titles: list[str] = []
                for ref in item.validation.collect:
                    _, signal = self._require_signal_ref(ref, signal_ref_map)
                    signal_titles.append(signal.metric)
                text = f"collect {' '.join(signal_titles)} for {item.id}"
                resolved_signals = []
                for ref in item.validation.collect:
                    contract, signal = self._require_signal_ref(ref, signal_ref_map)
                    resolved_signals.append(
                        {
                            "ref": ref,
                            "library": contract.contract,
                            "signal": signal.model_dump(mode="json"),
                        }
                    )
                validation.append(
                    {
                        "section": "hypothesis_validation",
                        "item_id": item.id,
                        "title": item.id,
                        "text": text,
                        "keywords": signal_titles,
                        "source_refs": item.source_refs,
                        "library": ", ".join(item.libraries),
                        "signals": resolved_signals,
                        "score": _score(text, terms),
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
        if composed is not None and not labels:
            labels = [instance.instance_id for instance in composed.contract_instances]
        summary = (
            f"Structured contract context for {', '.join(sorted(labels))}. "
            f"Goal={goal or 'unspecified'}. Symptom={symptom or 'unspecified'}."
        )
        return {
            "summary": summary,
            "shared_resources": shared_resources,
            "profile_first": _rank(observability),
            "candidate_actions": _rank(candidate_actions),
            "conflicts": _rank(conflicts),
            "validation": _rank(validation),
        }
