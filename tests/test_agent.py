from __future__ import annotations

import json
from pathlib import Path

import pytest

from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig
from syspilot.core.llm import LLMResponse


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
