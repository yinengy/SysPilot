from __future__ import annotations

from click.testing import CliRunner

from syspilot.cli import cli


def test_tune_help_includes_ui_mode_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "--help"])
    assert result.exit_code == 0
    assert "--ui-mode" in result.output
    assert "plain" in result.output
    assert "tui" in result.output
    assert "default: tui" in result.output


def test_tune_rejects_invalid_ui_mode_value() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "--ui-mode", "nope", "--command", "echo 1"])
    assert result.exit_code != 0
    assert "Invalid value for '--ui-mode'" in result.output
