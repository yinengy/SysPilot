import pytest
from syspilot.core.config import SysPilotConfig, LLMConfig, LLMProvider


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
        },
        "monitoring": {
            "interval_seconds": 10.0,
        },
        "target_pid": 1234,
    }
    
    config = SysPilotConfig.from_dict(data)
    assert config.llm.provider == LLMProvider.ANTHROPIC
    assert config.llm.model == "claude-3-opus-20240229"
    assert config.monitoring.interval_seconds == 10.0
    assert config.target_pid == 1234


def test_config_to_dict():
    config = SysPilotConfig()
    config.target_pid = 5678
    
    data = config.to_dict()
    assert data["llm"]["provider"] == "openai"
    assert data["target_pid"] == 5678
