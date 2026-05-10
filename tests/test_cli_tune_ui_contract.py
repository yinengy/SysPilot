from __future__ import annotations

from click.testing import CliRunner

from syspilot.cli import _wrap_spawn_command_with_memory_high, cli


def test_tune_help_includes_ui_mode_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "--help"])
    assert result.exit_code == 0
    assert "--ui-mode" in result.output
    assert "--memory-high" in result.output
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
