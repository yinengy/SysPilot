from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from syspilot.core.config import LLMConfig, LLMProvider
from syspilot.core.llm import BaseLLMProvider, LLMInterface, LLMResponse, Message, ToolDefinition


class DummyProvider(BaseLLMProvider):
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content="debug response",
            tool_calls=[
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "collect_profile",
                        "arguments": "{}",
                    },
                }
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            if False:
                yield ""

        return _gen()


@pytest.mark.asyncio
async def test_debug_markdown_transcript_written(tmp_path):
    md_path = tmp_path / "llm_debug.md"
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        api_key="test",
        debug_conversation=True,
        debug_markdown_file=str(md_path),
    )

    llm = LLMInterface(llm_config)
    llm._provider = DummyProvider()

    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="user prompt"),
    ]
    tools = [
        ToolDefinition(
            name="collect_profile",
            description="Collect profile",
            parameters={"type": "object"},
        )
    ]

    response = await llm.chat(messages, tools)
    assert response.content == "debug response"

    text = md_path.read_text(encoding="utf-8")
    assert "# SysPilot LLM Debug Transcript" in text
    assert "## Exchange 1" in text
    assert "#### system" in text
    assert "#### user" in text
    assert "### Response" in text
    assert "collect_profile" in text
    assert "debug response" in text
