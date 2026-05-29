import json

import pytest
from syspilot.core.config import SysPilotConfig, LLMConfig, LLMProvider
from syspilot.core.llm import LLMInterface
from syspilot.cli import (
    _default_llm_config_path,
    _load_cli_config,
    _load_default_llm_config,
    _merge_llm_config_with_repo_defaults,
    _manual_pipeline_llm_config,
    _metric_extraction_llm_config,
)


def test_default_config():
    config = SysPilotConfig()
    assert config.llm.provider == LLMProvider.OPENAI
    assert config.llm.model == "gpt-4o"
    assert config.monitoring.interval_seconds == 5.0
    assert config.tuning.dry_run is True


def test_config_from_dict():
    data = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "metric_extraction_model": "gpt-5.3-codex",
            "timeout_seconds": 12.5,
            "max_retries": 3,
            "debug_conversation": True,
            "debug_markdown_file": "/tmp/llm_debug.md",
        },
        "monitoring": {
            "interval_seconds": 10.0,
        },
        "target_pid": 1234,
    }

    config = SysPilotConfig.from_dict(data)
    assert config.llm.provider == LLMProvider.ANTHROPIC
    assert config.llm.model == "claude-3-opus-20240229"
    assert config.llm.metric_extraction_model == "gpt-5.3-codex"
    assert config.llm.timeout_seconds == 12.5
    assert config.llm.max_retries == 3
    assert config.llm.debug_conversation is True
    assert config.llm.debug_markdown_file == "/tmp/llm_debug.md"
    assert config.monitoring.interval_seconds == 10.0
    assert config.target_pid == 1234


def test_config_to_dict():
    config = SysPilotConfig()
    config.target_pid = 5678
    config.llm.timeout_seconds = 9.0
    config.llm.metric_extraction_model = "gpt-5.3-codex"
    config.llm.max_retries = 2
    config.llm.debug_conversation = True
    config.llm.debug_markdown_file = "/tmp/out.md"

    data = config.to_dict()
    assert data["llm"]["provider"] == "openai"
    assert data["llm"]["metric_extraction_model"] == "gpt-5.3-codex"
    assert data["llm"]["timeout_seconds"] == 9.0
    assert data["llm"]["max_retries"] == 2
    assert data["llm"]["debug_conversation"] is True
    assert data["llm"]["debug_markdown_file"] == "/tmp/out.md"
    assert data["target_pid"] == 5678


def test_openai_auth_config_from_dict():
    llm = LLMConfig.from_dict(
        {
            "provider": "openai_auth",
            "model": "gpt-5.3-codex",
            "openai_auth_file": "~/.syspilot/auth/openai.json",
        }
    )

    assert llm.provider == LLMProvider.OPENAI_AUTH
    assert llm.model == "gpt-5.3-codex"
    assert llm.openai_auth_file == "~/.syspilot/auth/openai.json"


def test_to_dict_includes_openai_auth_file():
    config = SysPilotConfig()
    config.llm.provider = LLMProvider.OPENAI_AUTH
    config.llm.openai_auth_file = "~/.syspilot/auth/openai.json"

    data = config.to_dict()
    assert data["llm"]["provider"] == "openai_auth"
    assert data["llm"]["openai_auth_file"] == "~/.syspilot/auth/openai.json"


def test_llm_interface_selects_openai_auth_provider():
    llm = LLMConfig(provider=LLMProvider.OPENAI_AUTH, model="gpt-5.3-codex")
    provider = LLMInterface(llm).provider
    assert provider.__class__.__name__ == "OpenAIAuthProvider"


def test_openai_auth_default_model_is_latest_codex():
    llm = LLMConfig.from_dict({"provider": "openai_auth"})
    assert llm.model == "gpt-5.3-codex"


def test_openai_auth_rejects_legacy_codex_models():
    llm = LLMConfig(provider=LLMProvider.OPENAI_AUTH, model="gpt-5.1-codex")
    provider = LLMInterface(llm).provider
    with pytest.raises(ValueError):
        getattr(provider, "_resolve_model")()


def test_default_llm_config_path_points_to_repo_config():
    assert _default_llm_config_path().name == "llm_config.json"
    assert _default_llm_config_path().parent.name == "config"


def test_load_default_llm_config_reads_repo_config_file():
    repo_config = json.loads(_default_llm_config_path().read_text(encoding="utf-8"))

    config = _load_default_llm_config()

    assert config.provider.value == repo_config["provider"]
    assert config.model == repo_config["model"]
    assert config.metric_extraction_model == repo_config["metric_extraction_model"]
    assert config.timeout_seconds == repo_config["timeout_seconds"]
    assert config.max_retries == repo_config["max_retries"]


def test_merge_llm_config_with_repo_defaults_uses_repo_values_when_yaml_missing_llm():
    repo_llm = _load_default_llm_config()

    merged = _merge_llm_config_with_repo_defaults({"monitoring": {"interval_seconds": 1.0}})

    assert merged.provider == repo_llm.provider
    assert merged.model == repo_llm.model
    assert merged.metric_extraction_model == repo_llm.metric_extraction_model
    assert merged.base_url == repo_llm.base_url
    assert merged.timeout_seconds == repo_llm.timeout_seconds


def test_merge_llm_config_with_repo_defaults_overlays_partial_yaml_llm():
    repo_llm = _load_default_llm_config()

    merged = _merge_llm_config_with_repo_defaults(
        {"llm": {"timeout_seconds": 120.0, "model": "gpt-5.4"}}
    )

    assert merged.provider == repo_llm.provider
    assert merged.base_url == repo_llm.base_url
    assert merged.timeout_seconds == 120.0
    assert merged.model == "gpt-5.4"
    assert merged.metric_extraction_model == repo_llm.metric_extraction_model


def test_metric_extraction_llm_config_overrides_only_model_and_appends_debug() -> None:
    base = LLMConfig(
        model="gpt-5.4(high)",
        metric_extraction_model="gpt-5.3-codex",
        base_url="https://example.test/v1",
        api_key="test",
        timeout_seconds=12.0,
        debug_conversation=True,
        debug_markdown_file="debug.md",
    )

    metric_config = _metric_extraction_llm_config(base)

    assert metric_config.model == "gpt-5.3-codex"
    assert metric_config.provider == base.provider
    assert metric_config.base_url == base.base_url
    assert metric_config.api_key == base.api_key
    assert metric_config.timeout_seconds == base.timeout_seconds
    assert metric_config.debug_conversation is True
    assert metric_config.debug_markdown_file == "debug.md"
    assert metric_config.debug_markdown_append is True


def test_manual_pipeline_llm_config_enforces_minimum_timeout_and_retry():
    base = LLMConfig(timeout_seconds=30.0, max_retries=0)
    tuned = _manual_pipeline_llm_config(base, minimum_timeout_seconds=180.0)
    assert tuned.timeout_seconds == 180.0
    assert tuned.max_retries == 1
    assert base.timeout_seconds == 30.0
    assert base.max_retries == 0


def test_load_cli_config_with_yaml_keeps_repo_llm_defaults(tmp_path):
    config_path = tmp_path / "syspilot.yaml"
    config_path.write_text("monitoring:\n  interval_seconds: 1.0\n", encoding="utf-8")

    loaded = _load_cli_config(str(config_path))
    repo_llm = _load_default_llm_config()

    assert loaded.monitoring.interval_seconds == 1.0
    assert loaded.llm.provider == repo_llm.provider
    assert loaded.llm.model == repo_llm.model
    assert loaded.llm.base_url == repo_llm.base_url
    assert loaded.llm.timeout_seconds == repo_llm.timeout_seconds
