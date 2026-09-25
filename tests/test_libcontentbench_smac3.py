from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


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


@pytest.mark.parametrize(
    ("budget", "expected_initial_count"),
    [(1, 1), (30, 8)],
)
def test_smac3_initial_design_budget_and_baseline_position(
    monkeypatch, tmp_path, budget, expected_initial_count
) -> None:
    fake_smac = types.ModuleType("smac")

    class FakeCallback:
        pass

    class FakeConfig(dict[str, object]):
        def __init__(self, values, origin=None):
            super().__init__(values)
            self.origin = origin

    class FakeScenario:
        def __init__(self, configspace, **kwargs):
            self.configspace = configspace
            self.n_trials = kwargs["n_trials"]

    class FakeInitialDesign:
        def __init__(self, configs):
            self.configs = configs

    class FakeFacade:
        requested_n_configs = None
        requested_max_ratio = None

        @staticmethod
        def get_initial_design(scenario, n_configs=0, max_ratio=0.25, additional_configs=None):
            FakeFacade.requested_n_configs = n_configs
            FakeFacade.requested_max_ratio = max_ratio
            additional = list(additional_configs or [])
            capped_count = n_configs
            if n_configs > 1:
                capped_count = int(max(1, min(n_configs, max_ratio * scenario.n_trials)))
            configs = []
            for index in range(capped_count):
                values = dict(additional[0])
                first_key = next(iter(values))
                values[first_key] = 1000 + index
                configs.append(FakeConfig(values, "Initial Design: Sobol"))
            configs.extend(additional)
            return FakeInitialDesign(configs)

        def __init__(self, scenario, target_function, *, initial_design, callbacks, overwrite):
            self.scenario = scenario
            self.target_function = target_function
            self.initial_design = initial_design
            self.callbacks = callbacks
            self.incumbent = None
            self.incumbent_cost = None

        def optimize(self):
            facade = self

            class FakeIntensifier:
                def get_incumbent(self):
                    return facade.incumbent

            class FakeRunHistory:
                def get_cost(self, config):
                    return facade.incumbent_cost

            smbo = types.SimpleNamespace(
                intensifier=FakeIntensifier(),
                runhistory=FakeRunHistory(),
            )
            for config in self.initial_design.configs[: self.scenario.n_trials]:
                info = types.SimpleNamespace(config=config)
                for callback in self.callbacks:
                    callback.on_ask_end(smbo, info)
                cost = self.target_function(config)
                if self.incumbent_cost is None or cost < self.incumbent_cost:
                    self.incumbent = config
                    self.incumbent_cost = cost
                value = types.SimpleNamespace(
                    cost=cost,
                    status=types.SimpleNamespace(name="SUCCESS"),
                )
                for callback in self.callbacks:
                    callback.on_tell_end(smbo, info, value)

    setattr(fake_smac, "Callback", FakeCallback)
    setattr(fake_smac, "HyperparameterOptimizationFacade", FakeFacade)
    setattr(fake_smac, "Scenario", FakeScenario)
    monkeypatch.setitem(sys.modules, "smac", fake_smac)
    sys.modules.pop("experiments.LibContentBench.smac3.optimize", None)
    sys.modules.pop("experiments.LibContentBench.smac3.trace_callback", None)

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
    monkeypatch.setattr(optimize.trace_callback, "require_smac_2_4_0", lambda: "2.4.0")
    monkeypatch.setattr(
        optimize,
        "baseline_configuration",
        lambda cs, spec: FakeConfig(baseline, "Initial Design: Baseline"),
    )
    monkeypatch.setattr(optimize, "TrialEvaluator", FakeEvaluator)

    summary = optimize.run_smac(
        spec=spec,
        size="medium",
        budget=budget,
        metric_name="score",
        metric_mode="maximize",
        output_dir=tmp_path,
        timeout_seconds=600,
        seed=0,
        n_workers=1,
        repo_root=tmp_path,
        crash_cost=1e9,
    )

    assert summary["budget"] == budget
    assert summary["total_executions"] == expected_initial_count
    assert FakeFacade.requested_n_configs == max(0, budget - 1)
    assert FakeFacade.requested_max_ratio == 0.25

    trace_summary = json.loads((tmp_path / "trace_summary.json").read_text(encoding="utf-8"))
    trace_manifest = json.loads((tmp_path / "trace_manifest.json").read_text(encoding="utf-8"))
    trace_rows = [
        json.loads(line)
        for line in (tmp_path / "optimizer_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert trace_manifest["optimizer"] == "smac3"
    assert trace_manifest["optimizer_version"] == "2.4.0"
    assert all(row["schema_version"] == 1 for row in trace_rows)
    assert all(row["optimizer"] == "smac3" for row in trace_rows)
    assert all(row["optimizer_version"] == "2.4.0" for row in trace_rows)
    if budget == 1:
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
        assert trace_summary["healthy_budget_30_baseline_expected_position"] is None
    else:
        assert trace_summary["origin_counts"]["initial_design.sobol"] == 7
        assert trace_summary["baseline_ask_order"] == 8
        assert trace_summary["healthy_budget_30_baseline_expected_position"] == 8
        assert trace_summary["healthy_budget_30_baseline_position_matches"] is True


def _load_trace_callback(monkeypatch):
    fake_smac = types.ModuleType("smac")

    class FakeCallback:
        pass

    setattr(fake_smac, "Callback", FakeCallback)
    monkeypatch.setitem(sys.modules, "smac", fake_smac)
    sys.modules.pop("experiments.LibContentBench.smac3.trace_callback", None)
    return importlib.import_module("experiments.LibContentBench.smac3.trace_callback")


def test_smac3_version_preflight_uses_distribution_metadata(monkeypatch) -> None:
    trace_callback = _load_trace_callback(monkeypatch)

    requested_distributions = []

    def fake_version(distribution_name):
        requested_distributions.append(distribution_name)
        return "2.4.0"

    monkeypatch.setattr(trace_callback.importlib.metadata, "version", fake_version)

    assert trace_callback.require_smac_2_4_0() == "2.4.0"
    assert requested_distributions == ["smac"]


def test_smac3_version_preflight_rejects_mismatch(monkeypatch) -> None:
    trace_callback = _load_trace_callback(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match=r"requires smac==2\.4\.0; installed smac version is 2\.3\.1",
    ):
        trace_callback.require_smac_2_4_0("2.3.1")


def test_tuner_trace_fingerprint_is_canonical_and_rejects_nonfinite_json(tmp_path) -> None:
    from experiments.LibContentBench.libcontentbench.tuner_trace import (
        atomic_write_json,
        compact_json,
        config_fingerprint,
    )

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    first = {"threads": Scalar(4), "policy": Scalar("close")}
    second = {"policy": "close", "threads": 4}

    assert config_fingerprint(first) == config_fingerprint(second)
    assert compact_json(first) == '{"policy":"close","threads":4}'
    atomic_write_json(tmp_path / "finite.json", first)
    assert json.loads((tmp_path / "finite.json").read_text(encoding="utf-8")) == second
    with pytest.raises(ValueError):
        compact_json({"cost": float("nan")})


@pytest.mark.parametrize(
    ("origin_raw", "expected"),
    [
        ("Initial Design: Sobol", "initial_design.sobol"),
        ("Acquisition Function Maximizer: Local Search", "model_based.local_search"),
        (
            "Acquisition Function Maximizer: Random Search (sorted)",
            "model_based.random_search",
        ),
        ("Random Search", "random_design"),
        ("Random Search (max retries, no candidates)", "random_fallback"),
        ("Custom", "other"),
    ],
)
def test_smac3_origin_mapping(monkeypatch, origin_raw, expected) -> None:
    trace_callback = _load_trace_callback(monkeypatch)

    assert trace_callback.classify_origin(origin_raw) == expected
    assert trace_callback.classify_origin(origin_raw, is_baseline=True) == (
        "initial_design.baseline"
    )


def test_smac3_trace_joins_repeated_configs_and_tracks_incumbents(monkeypatch, tmp_path) -> None:
    from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint

    trace_callback = _load_trace_callback(monkeypatch)

    class FakeConfig(dict[str, object]):
        def __init__(self, values, origin):
            super().__init__(values)
            self.origin = origin

    first = FakeConfig({"threads": 4}, "Initial Design: Baseline")
    repeated = FakeConfig({"threads": 4}, "Random Search")
    second = FakeConfig(
        {"threads": 2},
        "Acquisition Function Maximizer: Local Search",
    )
    evaluator_records = [
        {
            "trial_index": 1,
            "state": "OK",
            "metric_value": 10.0,
            "cost": -10.0,
            "config": dict(first),
        },
        {
            "trial_index": 2,
            "state": "ERROR",
            "metric_value": None,
            "cost": 1e9,
            "config": dict(repeated),
        },
        {
            "trial_index": 3,
            "state": "OK",
            "metric_value": 12.0,
            "cost": -12.0,
            "config": dict(second),
        },
    ]
    callback = trace_callback.SmacDecisionTrace(
        output_dir=tmp_path,
        evaluator_records=evaluator_records,
        baseline_fingerprint=config_fingerprint(first),
        budget=30,
        metric_name="throughput",
        metric_mode="maximize",
        installed_smac_version="2.4.0",
        initial_design_n_configs_requested=29,
        initial_design_max_ratio=0.25,
    )

    incumbent: dict[str, object] = {"config": None, "cost": None}

    class FakeIntensifier:
        def get_incumbent(self):
            return incumbent["config"]

    class FakeRunHistory:
        def get_cost(self, config):
            return incumbent["cost"]

    smbo = types.SimpleNamespace(
        intensifier=FakeIntensifier(),
        runhistory=FakeRunHistory(),
    )
    infos = [types.SimpleNamespace(config=config) for config in (first, repeated, second)]
    for info in infos:
        callback.on_ask_end(smbo, info)

    success = types.SimpleNamespace(name="SUCCESS")
    crashed = types.SimpleNamespace(name="CRASHED")
    incumbent.update(config=first, cost=-10.0)
    callback.on_tell_end(
        smbo,
        infos[0],
        types.SimpleNamespace(cost=-10.0, status=success),
    )
    incumbent.update(config=second, cost=-12.0)
    callback.on_tell_end(
        smbo,
        infos[2],
        types.SimpleNamespace(cost=-12.0, status=success),
    )
    callback.on_tell_end(
        smbo,
        infos[1],
        types.SimpleNamespace(cost=1e9, status=crashed),
    )
    callback.finalize()

    rows = [
        json.loads(line)
        for line in (tmp_path / "optimizer_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["ask_order"] for row in rows] == [1, 2, 3]
    assert [row["execution_order"] for row in rows] == [1, 2, 3]
    assert [row["tell_order"] for row in rows] == [1, 3, 2]
    assert rows[0]["origin"] == "initial_design.baseline"
    assert rows[0]["origin_raw"] == "Initial Design: Baseline"
    assert rows[0]["schema_version"] == 1
    assert rows[0]["optimizer"] == "smac3"
    assert rows[0]["optimizer_version"] == "2.4.0"
    assert rows[1]["trial_status"] == "ERROR"
    assert rows[2]["origin"] == "model_based.local_search"
    assert rows[2]["incumbent_changed"] is True
    assert all(row["acquisition_score"] is None for row in rows)
    assert all(
        row["acquisition_score_unavailable_reason"]
        == "not_exposed_by_smac_2_4_public_callbacks"
        for row in rows
    )

    summary = json.loads((tmp_path / "trace_summary.json").read_text(encoding="utf-8"))
    assert summary["incumbent_transition_count"] == 2
    assert summary["final_incumbent_fingerprint"] == config_fingerprint(second)


def test_smac3_trace_finalizes_partial_asks_and_version_mismatch(monkeypatch, tmp_path) -> None:
    from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint

    trace_callback = _load_trace_callback(monkeypatch)

    class FakeConfig(dict[str, object]):
        origin = "Initial Design: Sobol"

    config = FakeConfig({"threads": 4})
    callback = trace_callback.SmacDecisionTrace(
        output_dir=tmp_path,
        evaluator_records=[],
        baseline_fingerprint=config_fingerprint({"threads": 1}),
        budget=10,
        metric_name="throughput",
        metric_mode="maximize",
        installed_smac_version="2.3.1",
        initial_design_n_configs_requested=9,
        initial_design_max_ratio=0.25,
    )
    callback.on_ask_end(object(), types.SimpleNamespace(config=config))
    callback.finalize()

    rows = [
        json.loads(line)
        for line in (tmp_path / "optimizer_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((tmp_path / "trace_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "trace_summary.json").read_text(encoding="utf-8"))

    assert rows[0]["trial_status"] == "INCOMPLETE"
    assert rows[0]["optimizer"] == "smac3"
    assert rows[0]["optimizer_version"] == "2.3.1"
    assert summary["incomplete_trials"] == 1
    assert summary["optimizer"] == "smac3"
    assert manifest["installed_smac_version"] == "2.3.1"
    assert manifest["optimizer"] == "smac3"
    assert manifest["smac_version_matches_expected"] is False
    assert manifest["smac_version_status"] == "version_mismatch"


def _load_study_optimize(monkeypatch, proposal_values):
    fake_smac = types.ModuleType("smac")
    fake_dataclasses = types.ModuleType("smac.runhistory.dataclasses")
    fake_enumerations = types.ModuleType("smac.runhistory.enumerations")

    class FakeCallback:
        pass

    class FakeStatus:
        def __init__(self, name):
            self.name = name

    class FakeStatusType:
        SUCCESS = FakeStatus("SUCCESS")
        CRASHED = FakeStatus("CRASHED")
        TIMEOUT = FakeStatus("TIMEOUT")

    class FakeTrialValue:
        def __init__(self, cost, time=0.0, status=None, additional_info=None, **kwargs):
            self.cost = cost
            self.time = time
            self.status = status or FakeStatusType.SUCCESS
            self.additional_info = additional_info or {}

    class FakeConfig(dict[str, object]):
        def __init__(self, values, origin=None):
            super().__init__(values)
            self.origin = origin

    class FakeScenario:
        def __init__(self, configspace, **kwargs):
            self.configspace = configspace
            self.n_trials = kwargs["n_trials"]

    class FakeInitialDesign:
        def __init__(self, configs):
            self.configs = configs

    class FakeFacade:
        @staticmethod
        def get_initial_design(scenario, n_configs=0, max_ratio=0.25, additional_configs=None):
            del scenario, max_ratio
            generated = [
                FakeConfig(values, "Initial Design: Sobol")
                for values in proposal_values[:n_configs]
            ]
            generated.extend(list(additional_configs or []))
            return FakeInitialDesign(generated)

        def __init__(self, scenario, target_function, *, initial_design, callbacks, overwrite):
            del target_function, overwrite
            self.scenario = scenario
            self.configs = initial_design.configs
            self.callbacks = callbacks
            self.ask_index = 0
            self.incumbent = None
            self.incumbent_cost = None
            facade = self

            class FakeIntensifier:
                def get_incumbent(self):
                    return facade.incumbent

            class FakeRunHistory:
                def get_cost(self, config):
                    del config
                    return facade.incumbent_cost

            self.intensifier = FakeIntensifier()
            self.runhistory = FakeRunHistory()

        def ask(self):
            if self.ask_index >= len(self.configs):
                raise StopIteration
            config = self.configs[self.ask_index]
            self.ask_index += 1
            info = types.SimpleNamespace(config=config, seed=0)
            for callback in self.callbacks:
                callback.on_ask_end(self, info)
            return info

        def tell(self, info, value):
            if value.status.name == "SUCCESS" and (
                self.incumbent_cost is None or value.cost < self.incumbent_cost
            ):
                self.incumbent = info.config
                self.incumbent_cost = value.cost
            for callback in self.callbacks:
                callback.on_tell_end(self, info, value)

    setattr(fake_smac, "Callback", FakeCallback)
    setattr(fake_smac, "HyperparameterOptimizationFacade", FakeFacade)
    setattr(fake_smac, "Scenario", FakeScenario)
    setattr(fake_dataclasses, "TrialValue", FakeTrialValue)
    setattr(fake_enumerations, "StatusType", FakeStatusType)
    monkeypatch.setitem(sys.modules, "smac", fake_smac)
    monkeypatch.setitem(sys.modules, "smac.runhistory.dataclasses", fake_dataclasses)
    monkeypatch.setitem(sys.modules, "smac.runhistory.enumerations", fake_enumerations)
    sys.modules.pop("experiments.LibContentBench.smac3.optimize", None)
    sys.modules.pop("experiments.LibContentBench.smac3.trace_callback", None)

    optimize = importlib.import_module("experiments.LibContentBench.smac3.optimize")
    workflows = importlib.import_module("experiments.LibContentBench.smac3.workflows")
    spec = workflows.get_workflow("nestedcpu-blas")
    baseline = workflows.baseline_config_values(spec)

    class FakeEvaluator:
        def __init__(self, **kwargs):
            del kwargs
            self.records = []

        def __call__(self, config, seed=0):
            del seed
            trial_index = len(self.records) + 1
            state = "ERROR" if dict(config).get("outer_jobs") == 3 else "OK"
            metric = None if state == "ERROR" else float(100 - int(config["outer_jobs"]))
            cost = 1e9 if metric is None else -metric
            self.records.append(
                {
                    "trial_index": trial_index,
                    "state": state,
                    "metric_value": metric,
                    "cost": cost,
                    "config": dict(config),
                    "config_fingerprint": importlib.import_module(
                        "experiments.LibContentBench.libcontentbench.tuner_trace"
                    ).config_fingerprint(config),
                    "elapsed_sec": 0.1,
                    "cumulative_wall_sec": 0.1 * trial_index,
                    "returncode": 2 if state == "ERROR" else 0,
                    "timed_out": False,
                }
            )
            return cost

    monkeypatch.setattr(optimize, "build_config_space", lambda spec, seed=0: object())
    monkeypatch.setattr(optimize.trace_callback, "require_smac_2_4_0", lambda: "2.4.0")
    monkeypatch.setattr(
        optimize,
        "baseline_configuration",
        lambda cs, spec: FakeConfig(baseline, "Initial Design: Baseline"),
    )
    monkeypatch.setattr(optimize, "TrialEvaluator", FakeEvaluator)
    return optimize, spec


def _nested_config(outer_jobs):
    return {
        "outer_jobs": outer_jobs,
        "omp_threads": 1,
        "openblas_threads": 1,
        "mkl_threads": 1,
        "blis_threads": 1,
        "numexpr_threads": 1,
        "backend_prefer": "threads",
    }


def test_smac3_study_preserves_asks_reuses_duplicates_and_excludes_baseline(
    monkeypatch, tmp_path
) -> None:
    optimize, spec = _load_study_optimize(
        monkeypatch,
        [_nested_config(2), _nested_config(2), _nested_config(3), _nested_config(4)],
    )
    ledger_path = tmp_path / "study.jsonl"

    summary = optimize.run_smac(
        spec=spec,
        size="small",
        budget=30,
        metric_name="throughput",
        metric_mode="maximize",
        output_dir=tmp_path,
        timeout_seconds=10,
        seed=0,
        n_workers=1,
        repo_root=tmp_path,
        crash_cost=1e9,
        study_ledger=ledger_path,
        study_proposal_cap=5,
        study_unique_execution_target=2,
    )

    case_study = importlib.import_module(
        "experiments.LibContentBench.nestedcpu-blas.case_study"
    )
    events = case_study.read_events(ledger_path)
    proposals = [event for event in events if event["event_type"] == "proposal.recorded"]
    validations = [event for event in events if event["event_type"] == "proposal.validated"]
    starts = [event for event in events if event["event_type"] == "execution.started"]
    finishes = [event for event in events if event["event_type"] == "execution.finished"]

    assert [event["metadata"]["ask_order"] for event in proposals] == [1, 2, 3, 4, 5]
    assert [event["disposition"] for event in validations] == [
        "candidate",
        "duplicate",
        "candidate",
        "rejected",
        "duplicate",
    ]
    assert validations[1]["duplicate_of_proposal_sequence"] == proposals[0]["sequence"]
    assert len(starts) == 2
    assert [event["status"] for event in finishes] == ["ok", "error"]
    assert summary["total_executions"] == 3
    assert summary["n_search_trials"] == 2
    assert summary["study"]["unique_execution_count"] == 2
    assert summary["study"]["baseline_recorded"] is True
    assert summary["study"]["terminal_reason"] == "unique_execution_target_reached"

    native_rows = [
        json.loads(line)
        for line in (tmp_path / "optimizer_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["ask_order"] for row in native_rows] == [1, 2, 3, 4, 5]
    assert native_rows[1]["result_reused"] is True
    assert native_rows[1]["duplicate_of_ask_order"] == 1
    assert native_rows[1]["reused_execution_order"] == 1
    assert native_rows[3]["proposal_disposition"] == "rejected"
    assert native_rows[4]["is_baseline"] is True
    assert native_rows[4]["execution_order"] == 3


def test_smac3_study_proposal_cap_emits_explicit_shortfall(monkeypatch, tmp_path) -> None:
    optimize, spec = _load_study_optimize(
        monkeypatch,
        [_nested_config(2), _nested_config(2)],
    )
    ledger_path = tmp_path / "shortfall.jsonl"

    summary = optimize.run_smac(
        spec=spec,
        size="small",
        budget=30,
        metric_name="throughput",
        metric_mode="maximize",
        output_dir=tmp_path,
        timeout_seconds=10,
        seed=0,
        n_workers=1,
        repo_root=tmp_path,
        crash_cost=1e9,
        study_ledger=ledger_path,
        study_proposal_cap=3,
        study_unique_execution_target=3,
    )

    case_study = importlib.import_module(
        "experiments.LibContentBench.nestedcpu-blas.case_study"
    )
    terminal = case_study.read_events(ledger_path)[-1]
    assert terminal["reason"] == "proposal_cap_reached_with_shortfall"
    assert terminal["metadata"]["shortfall"] == 2
    assert summary["study"]["proposal_count"] == 3
    assert summary["study"]["unique_execution_count"] == 1


def test_smac3_study_default_keeps_seven_sobol_asks_before_baseline(
    monkeypatch, tmp_path
) -> None:
    proposals = [_nested_config(outer_jobs) for outer_jobs in range(2, 9)]
    optimize, spec = _load_study_optimize(monkeypatch, proposals)

    summary = optimize.run_smac(
        spec=spec,
        size="small",
        budget=30,
        metric_name="throughput",
        metric_mode="maximize",
        output_dir=tmp_path,
        timeout_seconds=10,
        seed=0,
        n_workers=1,
        repo_root=tmp_path,
        crash_cost=1e9,
        study_ledger=tmp_path / "default-study.jsonl",
    )

    native_summary = json.loads(
        (tmp_path / "trace_summary.json").read_text(encoding="utf-8")
    )
    assert native_summary["origin_counts"]["initial_design.sobol"] == 7
    assert native_summary["baseline_ask_order"] == 8
    assert native_summary["healthy_budget_30_baseline_expected_position"] == 8
    assert native_summary["healthy_budget_30_baseline_position_matches"] is True
    assert summary["study"]["terminal_reason"] == "terminated_with_shortfall"
    assert summary["study"]["unique_execution_count"] == 7


def test_smac3_study_cli_options_are_disabled_by_default_and_require_opt_in() -> None:
    from experiments.LibContentBench.smac3.cli import build_parser

    defaults = build_parser().parse_args(
        ["--workflow", "nestedcpu-blas", "--size", "small", "--dry-run"]
    )
    assert defaults.study_ledger is None
    assert defaults.study_proposal_cap is None
    assert defaults.study_unique_execution_target is None
