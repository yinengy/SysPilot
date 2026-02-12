import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from syspilot.core.config import TuningConfig
from syspilot.tuning.tuner import ApplicationTuner, TuningAction
from syspilot.tuning.engine import TuningEngine, IterationResult
from syspilot.profiling.collectors.base import CollectorResult


def test_tuning_action_defaults():
    action = TuningAction(
        config_type="sysctl",
        parameter="vm.swappiness",
        current_value=60,
        new_value=10,
        reason="reduce swap usage",
    )
    assert action.applied is False
    assert action.backup_path is None


def test_apply_tuning_dry_run():
    config = TuningConfig(dry_run=True)
    tuner = ApplicationTuner(config)

    actions = [
        {
            "config_type": "sysctl",
            "parameter": "vm.swappiness",
            "current_value": 60,
            "new_value": 10,
            "reason": "reduce swap",
        }
    ]

    results = tuner.apply_tuning(pid=1234, actions=actions)
    assert len(results) == 1
    assert results[0].parameter == "vm.swappiness"
    assert results[0].applied is False


def test_apply_tuning_unknown_config_type():
    config = TuningConfig(dry_run=False, backup_configs=False)
    tuner = ApplicationTuner(config)

    actions = [
        {
            "config_type": "unknown_type",
            "parameter": "foo",
            "current_value": None,
            "new_value": "bar",
            "reason": "test",
        }
    ]

    results = tuner.apply_tuning(pid=1234, actions=actions)
    assert len(results) == 1
    assert results[0].applied is False


def test_tuning_engine_compute_improvement():
    before = {"cpu.cpu_percent": 80.0, "memory.percent_used": 50.0}
    after = {"cpu.cpu_percent": 60.0, "memory.percent_used": 45.0}

    improvement = TuningEngine.compute_improvement(before, after)
    assert improvement < 0


def test_tuning_engine_compute_improvement_no_overlap():
    before = {"a": 100}
    after = {"b": 200}

    improvement = TuningEngine.compute_improvement(before, after)
    assert improvement == 0.0


def test_tuning_engine_convergence():
    engine = TuningEngine(
        profiling_manager=MagicMock(),
        tuner=MagicMock(),
        convergence_threshold=1.0,
    )

    engine.history = [
        IterationResult(iteration=1, before_metrics={}, after_metrics={},
                        actions_applied=[], improvement_pct=0.5),
        IterationResult(iteration=2, before_metrics={}, after_metrics={},
                        actions_applied=[], improvement_pct=0.3),
    ]

    assert engine.is_converged(last_n=2) is True


def test_tuning_engine_not_converged():
    engine = TuningEngine(
        profiling_manager=MagicMock(),
        tuner=MagicMock(),
        convergence_threshold=1.0,
    )

    engine.history = [
        IterationResult(iteration=1, before_metrics={}, after_metrics={},
                        actions_applied=[], improvement_pct=5.0),
        IterationResult(iteration=2, before_metrics={}, after_metrics={},
                        actions_applied=[], improvement_pct=0.3),
    ]

    assert engine.is_converged(last_n=2) is False


def test_tuning_engine_summary():
    engine = TuningEngine(
        profiling_manager=MagicMock(),
        tuner=MagicMock(),
    )

    engine.history = [
        IterationResult(iteration=1, before_metrics={}, after_metrics={},
                        actions_applied=[], improvement_pct=3.5),
    ]

    summary = engine.get_summary()
    assert summary["total_iterations"] == 1
    assert summary["iterations"][0]["improvement_pct"] == 3.5


def test_executor_sandbox_blocked():
    from syspilot.skills.executor import ScriptExecutor, ExecutionConfig

    executor = ScriptExecutor(ExecutionConfig(sandbox_enabled=True))

    with pytest.raises(ValueError, match="blocked by sandbox"):
        executor._validate_command(["rm", "-rf", "/"])


def test_executor_sandbox_sudo_restriction():
    from syspilot.skills.executor import ScriptExecutor, ExecutionConfig

    executor = ScriptExecutor(ExecutionConfig(sandbox_enabled=True))

    with pytest.raises(ValueError, match="sudo is not permitted"):
        executor._validate_command(["sudo", "python3", "malicious.py"])


def test_executor_sandbox_allowed():
    from syspilot.skills.executor import ScriptExecutor, ExecutionConfig

    executor = ScriptExecutor(ExecutionConfig(
        sandbox_enabled=True,
        allowed_commands=["python3", "bash"],
    ))

    executor._validate_command(["python3", "script.py"])


def test_executor_sandbox_not_in_allowlist():
    from syspilot.skills.executor import ScriptExecutor, ExecutionConfig

    executor = ScriptExecutor(ExecutionConfig(
        sandbox_enabled=True,
        allowed_commands=["python3"],
    ))

    with pytest.raises(ValueError, match="not in the allowed commands"):
        executor._validate_command(["bash", "script.sh"])
