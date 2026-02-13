import pytest
from syspilot.core.config import SysPilotConfig, LLMConfig, LLMProvider
from syspilot.core.llm import LLMInterface


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
    assert config.llm.debug_conversation is True
    assert config.llm.debug_markdown_file == "/tmp/llm_debug.md"
    assert config.monitoring.interval_seconds == 10.0
    assert config.target_pid == 1234


def test_config_to_dict():
    config = SysPilotConfig()
    config.target_pid = 5678
    config.llm.debug_conversation = True
    config.llm.debug_markdown_file = "/tmp/out.md"
    
    data = config.to_dict()
    assert data["llm"]["provider"] == "openai"
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
