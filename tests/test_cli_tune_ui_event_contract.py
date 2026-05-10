from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import cli

from syspilot.ui.emitter import TuneEventEmitter


@dataclass
class _DummyResponse:
    content: str


class _DummySpawnAgent:
    config: object
    context: SimpleNamespace

    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})

    async def step(self, prompt: str) -> _DummyResponse:
        _ = prompt
        return _DummyResponse('{"summary":"none","manual_citation":"","evidence":[],"actions":[]}')

    def enable_llm_prompt_logging(self, path: Path) -> Path:
        return path


class _ActionSpawnAgent(_DummySpawnAgent):
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self._calls: int = 0

    async def step(self, prompt: str) -> _DummyResponse:
        _ = prompt
        self._calls += 1
        if self._calls == 1:
            return _DummyResponse(
                '{"summary":"try one candidate","manual_citation":"rule","evidence":["signal"],'
                '"actions":[{"config_type":"env_var","parameter":"OMP_PROC_BIND","new_value":"close",'
                '"reason":"improve locality","manual_citation":"rule"}]}'
            )
        return _DummyResponse('{"summary":"none","manual_citation":"","evidence":[],"actions":[]}')


class _RuntimeRetrySpawnAgent(_DummySpawnAgent):
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self._calls: int = 0

    async def step(self, prompt: str) -> _DummyResponse:
        _ = prompt
        self._calls += 1
        if self._calls == 1:
            return _DummyResponse(
                '{"summary":"retry unsupported flag","manual_citation":"rule","evidence":["signal"],'
                '"actions":[{"config_type":"command","new_value":"python3 -u -V --badchild 1",'
                '"reason":"test runtime retry","manual_citation":"rule"}]}'
            )
        return _DummyResponse('{"summary":"none","manual_citation":"","evidence":[],"actions":[]}')


class _CaptureEmitter:
    instances: ClassVar[list["_CaptureEmitter"]] = []

    run_id: str
    events: list[str]
    records: list[dict[str, object]]

    def __init__(
        self,
        *,
        run_id: str | None = None,
        capacity: int = 256,
        critical_reserve: int = 16,
    ) -> None:
        _ = capacity
        _ = critical_reserve
        self.run_id = run_id or "capture-run"
        self.events = []
        self.records = []
        _CaptureEmitter.instances.append(self)

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        self.events.append(event_name)
        self.records.append(
            {
                "event": event_name,
                "payload": payload or {},
                "event_type": event_type or "event",
            }
        )
        return True


def _noop(*args: object, **kwargs: object) -> None:
    _ = args
    _ = kwargs


def _fake_run_command(*args: object, **kwargs: object) -> tuple[float, str, str, int]:
    _ = args
    _ = kwargs
    return 1.0, "ok", "", 0


async def _fake_live_profile(*args: object, **kwargs: object) -> dict[str, object]:
    _ = args
    _ = kwargs
    return {
        "pid": 9999,
        "wall_time": 0.01,
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "profiling_text": "",
        "profiling_json": {},
        "tool_calls_used": 0,
    }


def _unexpected_run_command(*args: object, **kwargs: object) -> tuple[float, str, str, int]:
    _ = args
    _ = kwargs
    raise AssertionError("spawn tune should not call _run_command for profiled spawn runs")


def test_monotonic_seq_and_required_fields() -> None:
    emitter = TuneEventEmitter(run_id="run-contract", capacity=12, critical_reserve=4)
    _ = emitter.publish("run.started", {"mode": "spawn"})
    _ = emitter.publish("progress.update", {"step": 1})
    _ = emitter.publish("run.finished", {"status": "ok"})

    events = emitter.drain()
    assert len(events) == 3

    required_keys = {
        "schema_version",
        "run_id",
        "seq",
        "ts",
        "event_type",
        "event",
        "payload",
    }
    seq_values = [cast(int, event["seq"]) for event in events]

    assert seq_values == sorted(seq_values)
    assert seq_values[0] >= 1

    for event in events:
        assert required_keys.issubset(event.keys())
        assert event["run_id"] == "run-contract"
        assert isinstance(event["schema_version"], str)
        assert isinstance(event["ts"], str)
        assert isinstance(event["event"], str)
        assert isinstance(event["payload"], dict)


def test_overflow_preserves_critical_events() -> None:
    emitter = TuneEventEmitter(run_id="run-overflow", capacity=6, critical_reserve=3)
    _ = emitter.publish("run.started", {"mode": "spawn"})

    for idx in range(32):
        _ = emitter.publish(
            "candidate.evaluated",
            {
                "candidate_id": idx,
                "verdict": "not_improved",
            },
        )

    _ = emitter.publish("error", {"message": "synthetic failure"})
    _ = emitter.publish("run.finished", {"status": "failed"})

    events = emitter.drain()
    names = [event["event"] for event in events]
    stats = emitter.stats()

    assert len(events) <= 6
    assert "run.started" in names
    assert "error" in names
    assert "run.finished" in names
    assert stats.dropped_count > 0
    assert stats.critical_sent_count >= 3


def test_pressure_coalesces_progress_updates() -> None:
    emitter = TuneEventEmitter(run_id="run-coalesce", capacity=5, critical_reserve=2)

    for step in range(15):
        _ = emitter.publish("progress.update", {"step": step})

    events = emitter.drain()
    progress_events = [event for event in events if event["event"] == "progress.update"]
    stats = emitter.stats()

    assert len(progress_events) == 1
    progress_payload = cast(dict[str, object], progress_events[0]["payload"])
    assert cast(int, progress_payload["step"]) == 14
    assert stats.coalesced_count > 0


def test_spawn_flow_emits_lifecycle_markers() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _DummySpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    events = _CaptureEmitter.instances[-1].events
    assert "run.started" in events
    assert "phase.started" in events
    assert "phase.finished" in events
    assert "run.finished" in events


def test_spawn_flow_profiles_baseline_and_candidate_runs() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()
    profile_calls: list[dict[str, object]] = []

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        profile_calls.append(kwargs)
        call_index = len(profile_calls)
        if call_index == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "baseline",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {"tools_used": [], "target_pids": [9999]},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {"tools_used": [], "target_pids": [10001]},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _ActionSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._run_command", _unexpected_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert len(profile_calls) == 2
    events = _CaptureEmitter.instances[-1].events
    assert events.count("profile.started") >= 2
    assert events.count("profile.finished") >= 2
    assert events.count("profile.snapshot") >= 2


def test_spawn_flow_threads_memory_high_into_events_and_profiled_runs() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()
    profile_calls: list[dict[str, object]] = []

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        profile_calls.append(kwargs)
        call_index = len(profile_calls)
        if call_index == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "baseline",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {"tools_used": [], "target_pids": [9999]},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {"tools_used": [], "target_pids": [10001]},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _ActionSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._run_command", _unexpected_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--memory-high",
                "1536M",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert len(profile_calls) == 2
    assert all(call["memory_high"] == "1536M" for call in profile_calls)
    run_started = next(
        record for record in _CaptureEmitter.instances[-1].records if record["event"] == "run.started"
    )
    payload = cast(dict[str, object], run_started["payload"])
    assert payload["memory_high"] == "1536M"


def test_spawn_flow_reports_contracts_and_manual_hints_separately() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _DummySpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--contract",
                "kafka,rocksdb",
                "--manual",
                "parent process starts workers after init",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert "Contracts: 2 loaded" in result.output
    assert "Manual hints: 1 provided" in result.output
    run_started = next(
        record for record in _CaptureEmitter.instances[-1].records if record["event"] == "run.started"
    )
    payload = cast(dict[str, object], run_started["payload"])
    assert payload["loaded_contracts"] == ["kafka", "rocksdb"]
    assert payload["manual_hints"] == ["parent process starts workers after init"]


def test_spawn_flow_final_output_uses_speedup_for_baseline_comparison() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()
    call_count = {"value": 0}

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "baseline",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {"tools_used": [], "target_pids": [9999]},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {"tools_used": [], "target_pids": [10001]},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _ActionSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._run_command", _unexpected_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert "Wall time: 0.80s (+20.0% vs current)" in result.output
    assert "New current wall time: 0.80s (+20.0% vs previous)" in result.output
    assert "1.25x speed up" in result.output
    assert "✓ Success! 1.25x speed up (20.0% improvement)" in result.output
    result_comparison = next(
        record
        for record in _CaptureEmitter.instances[-1].records
        if record["event"] == "result.comparison"
    )
    comparison_payload = cast(dict[str, object], result_comparison["payload"])
    assert comparison_payload["speedup"] == 1.25
    assert abs(cast(float, comparison_payload["improvement_pct"]) - 20.0) < 1e-9


def test_spawn_flow_primary_objective_updates_comparison_payload() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()
    call_count = {"value": 0}

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "ELAPSED_SEC=100\n",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {"tools_used": [], "target_pids": [9999]},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 1.2,
            "stdout": "ELAPSED_SEC=90\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {"tools_used": [], "target_pids": [10001]},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        output = cast(str, args[1])
        if "ELAPSED_SEC=100" in output:
            return {"elapsed_sec": 100.0}
        return {"elapsed_sec": 90.0}

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _ActionSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
        patch("syspilot.cli._run_command", _unexpected_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--primary-objective",
                "wall time",
            ],
        )

    assert result.exit_code == 0
    result_comparison = next(
        record
        for record in _CaptureEmitter.instances[-1].records
        if record["event"] == "result.comparison"
    )
    comparison_payload = cast(dict[str, object], result_comparison["payload"])
    assert comparison_payload["primary_objective_query"] == "wall time"
    assert comparison_payload["primary_objective_metric"] == "elapsed_sec"
    assert comparison_payload["primary_objective_mode"] == "minimize"
    assert comparison_payload["baseline_primary_objective"] == 100.0
    assert comparison_payload["tuned_primary_objective"] == 90.0
    assert abs(cast(float, comparison_payload["improvement_pct"]) - 10.0) < 1e-9


def test_spawn_flow_emits_next_action_in_recommendation_payload_and_output() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()
    call_count = {"value": 0}

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "baseline",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {"tools_used": [], "target_pids": [9999]},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {"tools_used": [], "target_pids": [10001]},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _ActionSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._run_command", _unexpected_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert "Next action: OMP_PROC_BIND=close" in result.output
    recommendation_event = next(
        record
        for record in _CaptureEmitter.instances[-1].records
        if record["event"] == "recommendation.ready"
    )
    payload = cast(dict[str, object], recommendation_event["payload"])
    assert payload["next_action"] == "OMP_PROC_BIND=close"


def test_spawn_flow_retries_candidate_after_runtime_unsupported_option() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()
    profile_calls: list[list[str]] = []

    async def _unsupported_then_success(*args: object, **kwargs: object) -> dict[str, object]:
        command = cast(list[str], args[1])
        profile_calls.append(command)
        if "--badchild" in command:
            return {
                "pid": 10001,
                "wall_time": 0.4,
                "stdout": "",
                "stderr": "unknown option '--badchild'",
                "returncode": -1,
                "timed_out": False,
                "profiling_text": "failed profile",
                "profiling_json": {},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10002,
            "wall_time": 0.2,
            "stdout": "ok",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "successful profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _RuntimeRetrySpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _unsupported_then_success),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "python3 -V",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert len(profile_calls) == 3
    assert "--badchild" in profile_calls[1]
    assert profile_calls[2] == ["python3", "-u", "-V"]
    assert "Retrying without unsupported option(s): --badchild" in result.output

