from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    def enable_llm_prompt_logging(self, path: Path) -> Path:
        return path


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


def test_default_ui_mode_non_tty_falls_back_to_plain() -> None:
    runner = CliRunner()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.ui.tui_sidecar.is_interactive_tty", lambda: False),
        patch(
            "syspilot.ui.tui_sidecar.resolve_tui_binary_path",
            side_effect=AssertionError("unexpected"),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert "Non-TTY detected" in result.output
    assert "SysPilot Tuning Engine (Spawn Mode)" in result.output
    assert "Performance Comparison" in result.output


def test_missing_tui_binary_falls_back_to_plain() -> None:
    runner = CliRunner()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.ui.tui_sidecar.is_interactive_tty", lambda: True),
        patch(
            "syspilot.ui.tui_sidecar.resolve_tui_binary_path", lambda override: (None, "missing")
        ),
        patch("syspilot.ui.tui_sidecar.TuneTuiSidecar", side_effect=AssertionError("unexpected")),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "tui",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert "missing" in result.output
    assert "SysPilot Tuning Engine (Spawn Mode)" in result.output
    assert "Performance Comparison" in result.output


def test_plain_mode_never_attempts_tui_launch() -> None:
    runner = CliRunner()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.cli.TuneTuiOrchestrator", side_effect=AssertionError("unexpected")),
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


def test_sidecar_disconnect_mid_run_continues_and_prints_completion() -> None:
    runner = CliRunner()

    class _FakeSidecar:
        def __init__(self, *, binary: object) -> None:
            _ = binary
            self.disconnected = False

        def start(self) -> None:
            return None

        def enqueue_jsonl(self, envelope: dict[str, object]) -> None:
            _ = envelope
            self.disconnected = True

        def close(self, *, wait_timeout_s: float = 0.5) -> None:
            _ = wait_timeout_s
            return None

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.ui.tui_sidecar.is_interactive_tty", lambda: True),
        patch(
            "syspilot.ui.tui_sidecar.resolve_tui_binary_path",
            lambda override: (__import__("pathlib").Path("/tmp/syspilot-tui"), None),
        ),
        patch("syspilot.ui.tui_sidecar.TuneTuiSidecar", _FakeSidecar),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "tui",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert "TUI disconnected" in result.output
    assert "SysPilot Tuning Engine (Spawn Mode)" in result.output
    assert "Performance Comparison" in result.output
