from __future__ import annotations

import importlib
import sys
import types


def test_smac3_dali_hw_decoder_load_is_continuous_float() -> None:
    from experiments.LibContentBench.smac3.workflows import (
        baseline_config_values,
        build_trial_payload,
        get_workflow,
    )

    spec = get_workflow("dali-nvjpeg-pytorch-cudnn")
    tunable = next(item for item in spec.tunables if item.field_name == "dali_hw_decoder_load")

    assert tunable.whitelist_name == "BENCHMARK_DALI_HW_DECODER_LOAD"
    assert tunable.kind == "float"
    assert tunable.min_value == 0.0
    assert tunable.max_value == 1.0
    assert tunable.options == ()
    assert baseline_config_values(spec)["dali_hw_decoder_load"] == 0.65

    payload = build_trial_payload(
        spec,
        "small",
        {**baseline_config_values(spec), "dali_hw_decoder_load": "0.123"},
        {"case_id": "trial", "label": "trial", "mode": "smac3", "output_json": "/tmp/out.json"},
    )

    assert payload["dali_hw_decoder_load"] == 0.123


def test_smac3_config_space_materializes_dali_float(monkeypatch) -> None:
    class FakeHyperparameter:
        def __init__(self, name, *, bounds=None, default=None, items=None):
            self.name = name
            self.bounds = bounds
            self.default = default
            self.items = items

    class FakeInteger(FakeHyperparameter):
        pass

    class FakeFloat(FakeHyperparameter):
        pass

    class FakeCategorical(FakeHyperparameter):
        pass

    class FakeConfigurationSpace:
        def __init__(self, *, seed=0):
            self.seed = seed
            self.hyperparameters = []

        def add(self, hyperparameters):
            self.hyperparameters.extend(hyperparameters)

    class FakeConfiguration:
        def __init__(self, cs, values):
            self.configuration_space = cs
            self.values = values

    fake_config_space = types.ModuleType("ConfigSpace")
    setattr(fake_config_space, "Categorical", FakeCategorical)
    setattr(fake_config_space, "Configuration", FakeConfiguration)
    setattr(fake_config_space, "ConfigurationSpace", FakeConfigurationSpace)
    setattr(fake_config_space, "Float", FakeFloat)
    setattr(fake_config_space, "Integer", FakeInteger)
    monkeypatch.setitem(sys.modules, "ConfigSpace", fake_config_space)
    sys.modules.pop("experiments.LibContentBench.smac3.config_space", None)

    config_space = importlib.import_module("experiments.LibContentBench.smac3.config_space")
    spec = importlib.import_module("experiments.LibContentBench.smac3.workflows").get_workflow(
        "dali-nvjpeg-pytorch-cudnn"
    )

    cs = config_space.build_config_space(spec, seed=7)
    parameter = next(item for item in cs.hyperparameters if item.name == "dali_hw_decoder_load")

    assert isinstance(parameter, FakeFloat)
    assert parameter.bounds == (0.0, 1.0)
    assert parameter.default == 0.65


REQUESTED_DALI_BASELINE = {
    "dali_batch_size": 16,
    "dali_prefetch_queue_depth": 1,
    "dali_hw_decoder_load": 0.65,
    "dali_num_threads": 1,
    "dali_mps_share": 50,
    "torch_batch_size": 16,
    "torch_mps_share": 50,
}


def test_smac3_dali_baseline_command_uses_requested_env_for_all_sizes() -> None:
    from experiments.LibContentBench.libcontentbench.direct_commands import build_direct_command
    from experiments.LibContentBench.smac3.workflows import (
        baseline_config_values,
        build_trial_payload,
        get_workflow,
    )

    spec = get_workflow("dali-nvjpeg-pytorch-cudnn")
    assert baseline_config_values(spec) == REQUESTED_DALI_BASELINE

    expected_size_flags = {
        "small": {"--dataset-images": "1500", "--timeout-seconds": "300"},
        "medium": {"--dataset-images": "5000", "--timeout-seconds": "600"},
        "large": {"--dataset-images": "10000", "--timeout-seconds": "900"},
    }
    for size, flags in expected_size_flags.items():
        payload = build_trial_payload(
            spec,
            size,
            baseline_config_values(spec),
            {"output_json": f"/tmp/{size}.json", "runtime_root": f"/tmp/{size}/runtime"},
        )
        command = build_direct_command(spec.workflow_id, payload)
        tokens = command.split()

        assert "BENCHMARK_DALI_BATCH_SIZE=16" in tokens
        assert "BENCHMARK_DALI_PREFETCH_QUEUE_DEPTH=1" in tokens
        assert "BENCHMARK_DALI_HW_DECODER_LOAD=0.65" in tokens
        assert "BENCHMARK_DALI_NUM_THREADS=1" in tokens
        assert "BENCHMARK_DALI_MPS_SHARE=50" in tokens
        assert "BENCHMARK_TORCH_BATCH_SIZE=16" in tokens
        assert "BENCHMARK_TORCH_MPS_SHARE=50" in tokens
        assert "--case-id" not in tokens
        assert "--label" not in tokens
        assert "--mode" not in tokens
        for option, value in flags.items():
            assert tokens[tokens.index(option) + 1] == value


def test_smac3_budget_one_runs_only_baseline(monkeypatch, tmp_path) -> None:
    fake_smac = types.ModuleType("smac")

    class FakeScenario:
        def __init__(self, configspace, **kwargs):
            self.configspace = configspace
            self.n_trials = kwargs["n_trials"]

    class FakeInitialDesign:
        def __init__(self, configs):
            self.configs = configs

    class FakeFacade:
        @staticmethod
        def get_initial_design(scenario, n_configs=0, additional_configs=None):
            configs = list(additional_configs or [])
            for index in range(n_configs):
                configs.append({"candidate": index})
            return FakeInitialDesign(configs)

        def __init__(self, scenario, target_function, *, initial_design, overwrite):
            self.scenario = scenario
            self.target_function = target_function
            self.initial_design = initial_design

        def optimize(self):
            for config in self.initial_design.configs[: self.scenario.n_trials]:
                self.target_function(config)

    fake_smac.HyperparameterOptimizationFacade = FakeFacade
    fake_smac.Scenario = FakeScenario
    monkeypatch.setitem(sys.modules, "smac", fake_smac)
    sys.modules.pop("experiments.LibContentBench.smac3.optimize", None)

    optimize = importlib.import_module("experiments.LibContentBench.smac3.optimize")
    workflows = importlib.import_module("experiments.LibContentBench.smac3.workflows")
    spec = workflows.get_workflow("dali-nvjpeg-pytorch-cudnn")
    baseline = workflows.baseline_config_values(spec)

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.records = []
            self._counter = 0

        def __call__(self, config, seed=0):
            self._counter += 1
            self.records.append(
                {
                    "trial_index": self._counter,
                    "state": "OK",
                    "metric_value": 1.0,
                    "cost": -1.0,
                    "config": dict(config),
                    "elapsed_sec": 0.1,
                    "cumulative_wall_sec": 0.1,
                }
            )
            return -1.0

    monkeypatch.setattr(optimize, "build_config_space", lambda spec, seed=0: object())
    monkeypatch.setattr(optimize, "baseline_configuration", lambda cs, spec: dict(baseline))
    monkeypatch.setattr(optimize, "TrialEvaluator", FakeEvaluator)

    summary = optimize.run_smac(
        spec=spec,
        size="medium",
        budget=1,
        metric_name="score",
        metric_mode="maximize",
        output_dir=tmp_path,
        timeout_seconds=600,
        seed=0,
        n_workers=1,
        repo_root=tmp_path,
        crash_cost=1e9,
    )

    assert summary["budget"] == 1
    assert summary["total_executions"] == 1
    assert summary["n_search_trials"] == 0
    assert summary["trials"] == [
        {
            "trial_index": 1,
            "is_baseline": True,
            "state": "OK",
            "metric_value": 1.0,
            "cost": -1.0,
            "speedup_vs_baseline": 1.0,
            "elapsed_sec": 0.1,
            "cumulative_wall_sec": 0.1,
            **{f"cfg.{key}": value for key, value in baseline.items()},
        }
    ]
