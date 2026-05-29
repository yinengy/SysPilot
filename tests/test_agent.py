from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig
from syspilot.core.llm import LLMResponse, Message


@pytest.mark.asyncio
async def test_process_tool_calls_handles_invalid_json_arguments() -> None:
    agent = SysPilotAgent(SysPilotConfig())
    agent.register_tool(
        "collect_profile",
        lambda **kwargs: {"ok": kwargs},
        "Collect profile",
        {"type": "object"},
    )

    results = await agent._process_tool_calls(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "collect_profile",
                    "arguments": "{not-json}",
                },
            }
        ]
    )

    assert len(results) == 1
    assert results[0].role == "tool"
    assert results[0].tool_call_id == "call_1"
    assert results[0].content.startswith("Error:")


@pytest.mark.asyncio
async def test_step_writes_llm_prompt_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = SysPilotAgent(SysPilotConfig())
    log_path = agent.enable_llm_prompt_logging(tmp_path / "tune_prompts.jsonl")

    async def _fake_chat(*args: object, **kwargs: object) -> LLMResponse:
        _ = args
        _ = kwargs
        return LLMResponse(content='{"summary":"ok","actions":[]}')

    monkeypatch.setattr(agent.llm, "chat", _fake_chat)

    await agent.step("hello tune")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["index"] == 1
    assert payload["state"] == "idle"
    assert any(msg["role"] == "system" for msg in payload["messages"])
    assert any(
        msg["role"] == "user" and msg["content"] == "hello tune" for msg in payload["messages"]
    )
    assert payload["tools"] == []


@pytest.mark.asyncio
async def test_step_stateless_does_not_resend_previous_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = SysPilotAgent(SysPilotConfig())
    captured: list[list[str]] = []

    async def _fake_chat(*args: object, **kwargs: object) -> LLMResponse:
        _ = args
        messages = cast(list[Message], kwargs["messages"])
        captured.append([msg.content for msg in messages])
        return LLMResponse(content='{"summary":"ok","actions":[]}')

    monkeypatch.setattr(agent.llm, "chat", _fake_chat)

    await agent.step("large previous tune prompt")
    await agent.step_stateless("small current tune prompt")

    assert any("large previous tune prompt" in content for content in captured[0])
    assert any("small current tune prompt" in content for content in captured[1])
    assert not any("large previous tune prompt" in content for content in captured[1])


def test_configure_tuning_prompt_omits_inactive_mode_terms() -> None:
    agent = SysPilotAgent(SysPilotConfig())
    agent.context.loaded_manuals["app"] = "SPECIFIC_TUNABLE_FLAG=only-this"
    agent.configure_tuning_prompt(manual_enabled=True)

    prompt = agent._message_history[0].content

    assert "Manual mode context" in prompt
    assert "SPECIFIC_TUNABLE_FLAG" not in prompt
    assert "LPCL" not in prompt
    assert "virtual_probe" not in prompt
    assert "Quick-mode context" not in prompt


def test_configure_tuning_prompt_adds_contract_context_terms() -> None:
    agent = SysPilotAgent(SysPilotConfig())
    agent.configure_tuning_prompt(contract_enabled=True)

    prompt = agent._message_history[0].content

    assert "Contract mode context" in prompt
    assert "A contract is structured LPCL library/runtime tuning knowledge" in prompt
    assert "A hypothesis is a contract-derived explanation" in prompt
    assert "A signal is a named measurable observation" in prompt
    assert "A virtual_probe is the contract's abstract name" in prompt
    assert "Manual mode context" not in prompt


def test_configure_tuning_prompt_adds_quick_window_terms() -> None:
    agent = SysPilotAgent(SysPilotConfig())
    agent.configure_tuning_prompt(contract_enabled=True, quick_mode=True, profiling_enabled=False)

    prompt = agent._message_history[0].content

    assert "Quick-mode context" in prompt
    assert "A window is a bounded time interval" in prompt
    assert "No-profile context" in prompt
    assert "Contract mode context" in prompt
