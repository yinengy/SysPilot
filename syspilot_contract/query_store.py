from __future__ import annotations

import logging
import re
from pathlib import Path

from syspilot_contract.checker import load_contract
from syspilot_contract.compose_contracts import compose_contracts
from syspilot_contract.schemas import ComposedContract, LibraryContract


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
        self._discover()

    def _discover(self) -> None:
        if not self.root.exists():
            return
        for bundle_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            contract_path = bundle_dir / "contract.yaml"
            if not contract_path.exists():
                continue
            try:
                self._contracts[bundle_dir.name] = load_contract(contract_path)
            except Exception as exc:
                logger.warning("Skipping invalid contract bundle %s: %s", contract_path, exc)

    def get_contract(self, contract_id: str) -> LibraryContract | ComposedContract:
        return self._contracts[contract_id]

    def list_contracts(self) -> list[str]:
        return sorted(self._contracts)

    def _compose_if_needed(
        self,
        contract_ids: list[str],
    ) -> tuple[list[LibraryContract], ComposedContract | None]:
        libraries: list[LibraryContract] = []
        composed: ComposedContract | None = None
        for contract_id in contract_ids:
            contract = self.get_contract(contract_id)
            if isinstance(contract, LibraryContract):
                libraries.append(contract)
            elif composed is None:
                composed = contract
        if composed is None and len(libraries) >= 2:
            composed = compose_contracts("context_composed", libraries)
        return libraries, composed

    def build_agent_context(
        self,
        contract_ids: list[str],
        goal: str | None = None,
        symptom: str | None = None,
        max_items_per_section: int = 5,
    ) -> dict[str, object]:
        libraries, composed = self._compose_if_needed(sorted(set(contract_ids)))
        terms = _tokenize(" ".join(value for value in (goal, symptom) if value))

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
                        "score": _score(text, terms),
                    }
                )

        conflicts: list[dict[str, object]] = []
        if composed is not None:
            for item in composed.hypotheses:
                control_names = [
                    control.name
                    for ref in item.basis.colliding_controls
                    if ref in control_ref_map
                    for _, control in [control_ref_map[ref]]
                ]
                signal_titles = [
                    signal.metric
                    for ref in item.validation.collect
                    if ref in signal_ref_map
                    for _, signal in [signal_ref_map[ref]]
                ]
                recipe_titles = [
                    recipe.id
                    for ref in item.candidate_recipe_refs
                    if ref in recipe_ref_map
                    for _, recipe in [recipe_ref_map[ref]]
                ]
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
                        "score": _score(text, terms),
                    }
                )
                for ref in item.candidate_control_refs:
                    if ref not in control_ref_map:
                        continue
                    contract, control = control_ref_map[ref]
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
                            "score": _score(control.name, terms),
                        }
                    )
                for ref in item.candidate_recipe_refs:
                    if ref not in recipe_ref_map:
                        continue
                    contract, recipe = recipe_ref_map[ref]
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
                        "score": _score(text, terms),
                    }
                )
        if composed is not None:
            for item in composed.hypotheses:
                signal_titles = [
                    signal.metric
                    for ref in item.validation.collect
                    if ref in signal_ref_map
                    for _, signal in [signal_ref_map[ref]]
                ]
                text = f"collect {' '.join(signal_titles)} for {item.id}"
                validation.append(
                    {
                        "section": "hypothesis_validation",
                        "item_id": item.id,
                        "title": item.id,
                        "text": text,
                        "keywords": signal_titles,
                        "source_refs": item.source_refs,
                        "library": ", ".join(item.libraries),
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
            labels = composed.selected_contracts
        summary = (
            f"Structured contract context for {', '.join(sorted(labels))}. "
            f"Goal={goal or 'unspecified'}. Symptom={symptom or 'unspecified'}."
        )
        return {
            "summary": summary,
            "profile_first": _rank(observability),
            "candidate_actions": _rank(candidate_actions),
            "conflicts": _rank(conflicts),
            "validation": _rank(validation),
        }
