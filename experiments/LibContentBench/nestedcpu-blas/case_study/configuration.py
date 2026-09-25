from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint

from .manifest import ControlSpec, SearchSpaceManifest, load_manifest


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    outer_jobs: int
    omp_threads: int
    openblas_threads: int
    mkl_threads: int
    blis_threads: int
    numexpr_threads: int
    backend_prefer: str

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        manifest: SearchSpaceManifest | None = None,
    ) -> CanonicalConfig:
        canonical = canonicalize_config(values, manifest)
        return cls(**canonical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_jobs": self.outer_jobs,
            "omp_threads": self.omp_threads,
            "openblas_threads": self.openblas_threads,
            "mkl_threads": self.mkl_threads,
            "blis_threads": self.blis_threads,
            "numexpr_threads": self.numexpr_threads,
            "backend_prefer": self.backend_prefer,
        }


def _coerce_value(control: ControlSpec, value: Any) -> int | str:
    if control.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{control.name} must be an integer")
        integer = int(value)
        assert control.minimum is not None and control.maximum is not None
        if not control.minimum <= integer <= control.maximum:
            raise ValueError(
                f"{control.name} must be in {control.minimum}..{control.maximum}, got {integer}"
            )
        return integer
    if not isinstance(value, str) or value not in control.options:
        raise ValueError(f"{control.name} must be one of {control.options}, got {value!r}")
    return value


def canonicalize_config(
    values: Mapping[str, Any],
    manifest: SearchSpaceManifest | None = None,
) -> dict[str, Any]:
    active_manifest = manifest or load_manifest()
    alias_to_control = {
        alias: control
        for control in active_manifest.controls
        for alias in control.aliases
    }
    unknown = sorted(str(key) for key in values if str(key) not in alias_to_control)
    if unknown:
        raise ValueError(f"Unknown controls: {', '.join(unknown)}")

    canonical: dict[str, Any] = {}
    for raw_name, raw_value in values.items():
        control = alias_to_control[str(raw_name)]
        value = _coerce_value(control, raw_value)
        previous = canonical.get(control.name)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting aliases supplied for {control.name}")
        canonical[control.name] = value

    missing = [name for name in active_manifest.control_order if name not in canonical]
    if missing:
        raise ValueError(f"Missing controls: {', '.join(missing)}")
    return {name: canonical[name] for name in active_manifest.control_order}


def canonical_config_fingerprint(
    config: CanonicalConfig | Mapping[str, Any],
    manifest: SearchSpaceManifest | None = None,
) -> str:
    canonical = (
        config.to_dict()
        if isinstance(config, CanonicalConfig)
        else canonicalize_config(config, manifest)
    )
    return config_fingerprint(canonical)


def config_delta(
    before: CanonicalConfig | Mapping[str, Any],
    after: CanonicalConfig | Mapping[str, Any],
    manifest: SearchSpaceManifest | None = None,
) -> dict[str, Any]:
    active_manifest = manifest or load_manifest()
    before_values = (
        before.to_dict()
        if isinstance(before, CanonicalConfig)
        else canonicalize_config(before, active_manifest)
    )
    after_values = (
        after.to_dict()
        if isinstance(after, CanonicalConfig)
        else canonicalize_config(after, active_manifest)
    )
    return {
        name: after_values[name]
        for name in active_manifest.control_order
        if before_values[name] != after_values[name]
    }
