from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("search_space.json")


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    matrix_size: int
    task_count: int
    repetitions: int

    def to_dict(self) -> dict[str, int]:
        return {
            "matrix_size": self.matrix_size,
            "task_count": self.task_count,
            "repetitions": self.repetitions,
        }


@dataclass(frozen=True, slots=True)
class ControlSpec:
    name: str
    kind: str
    aliases: tuple[str, ...]
    minimum: int | None = None
    maximum: int | None = None
    options: tuple[str, ...] = ()

    @property
    def cardinality(self) -> int:
        if self.kind == "integer":
            assert self.minimum is not None and self.maximum is not None
            return self.maximum - self.minimum + 1
        return len(self.options)


@dataclass(frozen=True, slots=True)
class SearchSpaceManifest:
    schema_version: int
    study_id: str
    workload: WorkloadSpec
    controls: tuple[ControlSpec, ...]
    baseline_values: tuple[Any, ...]
    reference_vary: str
    reference_minimum: int
    reference_maximum: int
    reference_fixed: tuple[tuple[str, Any], ...]
    proposal_cap: int
    unique_execution_target: int
    declared_domain_cardinality: int

    @property
    def control_order(self) -> tuple[str, ...]:
        return tuple(control.name for control in self.controls)

    @property
    def baseline(self) -> dict[str, Any]:
        return dict(zip(self.control_order, self.baseline_values, strict=True))

    @property
    def domain_cardinality(self) -> int:
        cardinality = 1
        for control in self.controls:
            cardinality *= control.cardinality
        return cardinality

    def reference_configurations(self) -> tuple[dict[str, Any], ...]:
        fixed = dict(self.reference_fixed)
        return tuple(
            {
                name: value if name != self.reference_vary else varying_value
                for name, value in (
                    (control_name, fixed.get(control_name))
                    for control_name in self.control_order
                )
            }
            for varying_value in range(self.reference_minimum, self.reference_maximum + 1)
        )


def _control_from_payload(payload: dict[str, Any]) -> ControlSpec:
    kind = str(payload["kind"])
    if kind not in {"integer", "categorical"}:
        raise ValueError(f"Unsupported control kind: {kind!r}")
    return ControlSpec(
        name=str(payload["name"]),
        kind=kind,
        aliases=tuple(str(alias) for alias in payload["aliases"]),
        minimum=int(payload["minimum"]) if kind == "integer" else None,
        maximum=int(payload["maximum"]) if kind == "integer" else None,
        options=tuple(str(option) for option in payload.get("options", ())),
    )


@lru_cache(maxsize=1)
def load_manifest() -> SearchSpaceManifest:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    controls = tuple(_control_from_payload(item) for item in payload["controls"])
    control_order = tuple(control.name for control in controls)
    baseline = payload["baseline"]
    reference = payload["reference_subset"]
    reference_values = reference["values"]
    budgets = payload["budgets"]

    manifest = SearchSpaceManifest(
        schema_version=int(payload["schema_version"]),
        study_id=str(payload["study_id"]),
        workload=WorkloadSpec(**{key: int(value) for key, value in payload["workload"].items()}),
        controls=controls,
        baseline_values=tuple(baseline[name] for name in control_order),
        reference_vary=str(reference["vary"]),
        reference_minimum=int(reference_values["minimum"]),
        reference_maximum=int(reference_values["maximum"]),
        reference_fixed=tuple(
            (name, reference["fixed"][name])
            for name in control_order
            if name != reference["vary"]
        ),
        proposal_cap=int(budgets["proposal_cap"]),
        unique_execution_target=int(budgets["unique_execution_target"]),
        declared_domain_cardinality=int(payload["domain_cardinality"]),
    )
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: SearchSpaceManifest) -> None:
    if manifest.schema_version != 1:
        raise ValueError(f"Unsupported search-space schema version: {manifest.schema_version}")
    if len(set(manifest.control_order)) != len(manifest.controls):
        raise ValueError("Control names must be unique")
    aliases = [alias for control in manifest.controls for alias in control.aliases]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Control aliases must be globally unique")
    if manifest.domain_cardinality != manifest.declared_domain_cardinality:
        raise ValueError("Declared domain cardinality does not match the control domains")
    if manifest.reference_vary not in manifest.control_order:
        raise ValueError("Reference subset varies an unknown control")
    if manifest.proposal_cap < manifest.unique_execution_target:
        raise ValueError("Proposal cap cannot be lower than the unique-execution target")
