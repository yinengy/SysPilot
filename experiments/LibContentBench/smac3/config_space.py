"""Map a LibContentBench :class:`WorkflowSpec` onto a SMAC3 ConfigSpace.

Integer knobs become :class:`ConfigSpace.Integer` (with the baseline clamped
into bounds as the default), float knobs become :class:`ConfigSpace.Float`, and
enum/bool knobs become :class:`ConfigSpace.Categorical` over their option strings.

This module imports ConfigSpace, so it is only importable once the SMAC3 driver
environment is installed (see ``setup.sh`` / ``requirements.txt``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from experiments.LibContentBench.smac3.workflows import WorkflowSpec, baseline_config_values

__all__ = ["build_config_space", "baseline_config_values", "baseline_configuration"]


def build_config_space(spec: WorkflowSpec, *, seed: int = 0) -> Any:
    config_space_module = import_module("ConfigSpace")
    categorical = getattr(config_space_module, "Categorical")
    configuration_space = getattr(config_space_module, "ConfigurationSpace")
    float_parameter = getattr(config_space_module, "Float")
    integer = getattr(config_space_module, "Integer")

    cs = configuration_space(seed=seed)
    hyperparameters = []
    for tunable in spec.tunables:
        default = tunable.encode_default(spec.baseline_runtime[tunable.field_name])
        if tunable.kind == "integer":
            assert tunable.min_value is not None
            assert tunable.max_value is not None
            hyperparameters.append(
                integer(
                    tunable.field_name,
                    bounds=(int(tunable.min_value), int(tunable.max_value)),
                    default=int(default),
                )
            )
        elif tunable.kind == "float":
            assert tunable.min_value is not None
            assert tunable.max_value is not None
            hyperparameters.append(
                float_parameter(
                    tunable.field_name,
                    bounds=(float(tunable.min_value), float(tunable.max_value)),
                    default=float(default),
                )
            )
        else:
            hyperparameters.append(
                categorical(
                    tunable.field_name,
                    items=list(tunable.options),
                    default=default,
                )
            )
    cs.add(hyperparameters)
    return cs


def baseline_configuration(cs: Any, spec: WorkflowSpec) -> Any:
    config_space_module = import_module("ConfigSpace")
    configuration = getattr(config_space_module, "Configuration")
    return configuration(cs, values=baseline_config_values(spec))
