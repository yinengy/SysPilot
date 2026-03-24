from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import cli


def test_auth_login_help_shows_manual_default() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["auth", "login", "--help"])

    assert result.exit_code == 0
    assert "--auto / --manual" in result.output
    assert "[default: manual]" in result.output


def test_auth_login_defaults_to_manual_flow() -> None:
    runner = CliRunner()
    fake_auth = {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}

    with (
        patch(
            "syspilot.cli.create_pkce_verifier_and_challenge",
            return_value=("verifier123", "challenge123"),
        ),
        patch("syspilot.cli.create_state", return_value="state123"),
        patch("syspilot.cli.build_authorization_url", return_value="https://example.com/auth"),
        patch("syspilot.cli.wait_for_callback_code", side_effect=AssertionError("unexpected")),
        patch("syspilot.cli.exchange_authorization_code", return_value=fake_auth) as exchange_mock,
        patch("syspilot.cli.save_auth") as save_mock,
        patch("syspilot.cli.syspilot_openai_auth_file", return_value=Path("/tmp/openai-auth.json")),
    ):
        result = runner.invoke(cli, ["auth", "login"], input="manual-code\n")

    assert result.exit_code == 0
    assert "Open this URL and complete login" in result.output
    exchange_mock.assert_called_once_with("manual-code", "verifier123")
    save_mock.assert_called_once_with(fake_auth, Path("/tmp/openai-auth.json"))


def test_auth_login_auto_mode_uses_callback_flow() -> None:
    runner = CliRunner()
    fake_auth = {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}

    with (
        patch(
            "syspilot.cli.create_pkce_verifier_and_challenge",
            return_value=("verifier123", "challenge123"),
        ),
        patch("syspilot.cli.create_state", return_value="state123"),
        patch("syspilot.cli.build_authorization_url", return_value="https://example.com/auth"),
        patch(
            "syspilot.cli.wait_for_callback_code",
            return_value=("callback-code", True),
        ) as callback_mock,
        patch("syspilot.cli.click.prompt", side_effect=AssertionError("unexpected prompt")),
        patch("syspilot.cli.exchange_authorization_code", return_value=fake_auth) as exchange_mock,
        patch("syspilot.cli.save_auth") as save_mock,
        patch("syspilot.cli.syspilot_openai_auth_file", return_value=Path("/tmp/openai-auth.json")),
    ):
        result = runner.invoke(cli, ["auth", "login", "--auto"])

    assert result.exit_code == 0
    assert "Browser opened for OpenAI login. Waiting for callback..." in result.output
    callback_mock.assert_called_once_with(
        state="state123",
        timeout_seconds=180,
        authorize_url="https://example.com/auth",
    )
    exchange_mock.assert_called_once_with("callback-code", "verifier123")
    save_mock.assert_called_once_with(fake_auth, Path("/tmp/openai-auth.json"))
