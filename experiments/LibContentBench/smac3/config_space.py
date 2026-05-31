"""Map a LibContentBench :class:`WorkflowSpec` onto a SMAC3 ConfigSpace.

Integer knobs become :class:`ConfigSpace.Integer` (with the baseline clamped
into bounds as the default); enum/bool knobs become :class:`ConfigSpace.Categorical`
over their option strings. The categorical default is the encoded baseline token.

This module imports ConfigSpace, so it is only importable once the SMAC3 driver
environment is installed (see ``setup.sh`` / ``requirements.txt``).
"""

from __future__ import annotations

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Integer

from experiments.LibContentBench.smac3.workflows import WorkflowSpec, baseline_config_values

__all__ = ["build_config_space", "baseline_config_values", "baseline_configuration"]


def build_config_space(spec: WorkflowSpec, *, seed: int = 0) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    hyperparameters = []
    for tunable in spec.tunables:
        default = tunable.encode_default(spec.baseline_runtime[tunable.field_name])
        if tunable.kind == "integer":
            hyperparameters.append(
                Integer(
                    tunable.field_name,
                    bounds=(int(tunable.min_value), int(tunable.max_value)),
                    default=int(default),
                )
            )
        else:
            hyperparameters.append(
                Categorical(
                    tunable.field_name,
                    items=list(tunable.options),
                    default=default,
                )
            )
    cs.add(hyperparameters)
    return cs


def baseline_configuration(cs: ConfigurationSpace, spec: WorkflowSpec) -> Configuration:
    return Configuration(cs, values=baseline_config_values(spec))
