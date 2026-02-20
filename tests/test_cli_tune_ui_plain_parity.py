from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import cli


@dataclass
class _DummyResponse:
    content: str


class _DummyAgent:
    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})

    async def step(self, prompt: str) -> _DummyResponse:
        _ = prompt
        return _DummyResponse('{"summary":"none","manual_citation":"","evidence":[],"actions":[]}')


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


def test_plain_mode_emits_key_markers_for_spawn_flow() -> None:
    runner = CliRunner()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
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
    assert "SysPilot Tuning Engine (Spawn Mode)" in result.output
    assert "Round 0 Step 0: Initial Baseline Run" in result.output
    assert "Agent recommended no actionable tuning" in result.output
    assert "Performance Comparison" in result.output
    assert "Applied actions" in result.output
