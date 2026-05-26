from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from click.testing import CliRunner
from unittest.mock import patch

from syspilot.cli import _wrap_spawn_command_with_memory_high, cli
from syspilot.quick_mode import QuickModeDefaults


class _DummyQuickAgent:
    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})

    def enable_llm_prompt_logging(self, path: Path) -> Path:
        return path


def test_tune_help_includes_ui_mode_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "--help"])
    assert result.exit_code == 0
    assert "--ui-mode" in result.output
    assert "--memory-high" in result.output
    assert "--exclude-config" in result.output
    assert "--quick-mode / --no-quick-mode" in result.output
    assert "--quick-sample-interval-sec" in result.output
    assert "--quick-max-total-short-runs" in result.output
    assert "plain" in result.output
    assert "tui" in result.output
    assert "default: tui" in result.output


def test_tune_rejects_invalid_ui_mode_value() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "--ui-mode", "nope", "--command", "echo 1"])
    assert result.exit_code != 0
    assert "Invalid value for '--ui-mode'" in result.output


def test_top_level_help_no_longer_exposes_monitor_or_analyze_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert " monitor " not in result.output
    assert " analyze " not in result.output


def test_tune_requires_spawn_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "--memory-high", "1536M"])
    assert result.exit_code == 1
    assert "--command is required; tune is spawn-mode only" in result.output


def test_quick_mode_accepts_primary_objective_options() -> None:
    runner = CliRunner()
    calls: list[dict[str, object]] = []

    def _fake_quick(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyQuickAgent),
        patch("syspilot.cli._register_default_skills", lambda *args, **kwargs: None),
        patch("syspilot.cli._register_profiling_tools", lambda *args, **kwargs: None),
        patch("syspilot.cli._register_tuning_tools", lambda *args, **kwargs: None),
        patch("syspilot.cli._run_quick_tune_loop", _fake_quick),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--quick-mode",
                "--ui-mode",
                "plain",
                "--command",
                "echo 1",
                "--primary-objective",
                "throughput",
                "--primary-objective-mode",
                "maximize",
                "--exclude-config",
                "BENCHMARK_BLOSC_CNAME,BENCHMARK_BLOSC_SHUFFLE",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["primary_objective"] == "throughput"
    assert calls[0]["primary_objective_mode"] == "maximize"
    assert calls[0]["excluded_config_names"] == [
        "BENCHMARK_BLOSC_CNAME",
        "BENCHMARK_BLOSC_SHUFFLE",
    ]


def test_quick_mode_forwards_budget_overrides() -> None:
    runner = CliRunner()
    calls: list[dict[str, object]] = []

    def _fake_quick(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyQuickAgent),
        patch("syspilot.cli._register_default_skills", lambda *args, **kwargs: None),
        patch("syspilot.cli._register_profiling_tools", lambda *args, **kwargs: None),
        patch("syspilot.cli._register_tuning_tools", lambda *args, **kwargs: None),
        patch("syspilot.cli._run_quick_tune_loop", _fake_quick),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--quick-mode",
                "--ui-mode",
                "plain",
                "--command",
                "echo 1",
                "--quick-sample-interval-sec",
                "2.5",
                "--quick-short-measurement-sec",
                "15",
                "--quick-full-run-ratio",
                "0.5",
                "--quick-max-full-runs-per-group",
                "2",
                "--quick-max-total-short-runs",
                "7",
                "--quick-max-total-full-runs",
                "4",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert len(calls) == 1
    defaults = calls[0]["defaults"]
    assert isinstance(defaults, QuickModeDefaults)
    assert defaults.sample_interval_sec == 2.5
    assert defaults.short_measurement_sec == 15.0
    assert defaults.full_run_ratio == 0.5
    assert defaults.max_full_runs_per_group == 2
    assert defaults.max_total_short_runs == 7
    assert defaults.max_total_full_runs == 4


def test_wrap_spawn_command_with_memory_high_prefixes_systemd_scope() -> None:
    wrapped = _wrap_spawn_command_with_memory_high(["python3", "bench.py"], "1536M")
    assert wrapped[:5] == ["systemd-run", "--user", "--scope", "-p", "MemoryHigh=1536M"]
    assert wrapped[5:] == ["python3", "bench.py"]


def test_wrap_spawn_command_with_memory_high_wraps_workdir_via_shell() -> None:
    wrapped = _wrap_spawn_command_with_memory_high(["python3", "bench.py"], "1536M", "/tmp/demo")
    assert wrapped[:7] == [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        "MemoryHigh=1536M",
        "/bin/sh",
        "-lc",
    ]
    assert wrapped[7] == "cd /tmp/demo && exec python3 bench.py"


def test_wrap_spawn_command_with_memory_high_is_noop_when_unset() -> None:
    original = ["python3", "bench.py"]
    assert _wrap_spawn_command_with_memory_high(original, None) == original
