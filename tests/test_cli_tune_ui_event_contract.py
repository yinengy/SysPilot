from __future__ import annotations

from dataclasses import dataclass
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


class _DummyAttachAgent:
    config: object
    context: SimpleNamespace

    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})

    async def run_tuning_loop(self, engine: object, max_iterations: int) -> str:
        _ = engine
        _ = max_iterations
        return "ok"

    def stop(self) -> None:
        return None


class _CaptureEmitter:
    instances: ClassVar[list["_CaptureEmitter"]] = []

    run_id: str
    events: list[str]

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
        _CaptureEmitter.instances.append(self)

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        _ = payload
        _ = event_type
        self.events.append(event_name)
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
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    events = _CaptureEmitter.instances[-1].events
    assert "run.started" in events
    assert "phase.started" in events
    assert "phase.finished" in events
    assert "run.finished" in events


def test_attach_flow_emits_lifecycle_markers() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _DummyAttachAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--pid",
                "1234",
                "--max-iterations",
                "1",
            ],
        )

    assert result.exit_code == 0
    events = _CaptureEmitter.instances[-1].events
    assert "run.started" in events
    assert "phase.started" in events
    assert "phase.finished" in events
    assert "run.finished" in events
