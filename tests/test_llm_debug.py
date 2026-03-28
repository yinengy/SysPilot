from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from syspilot.core.config import LLMConfig, LLMProvider
from syspilot.core.llm import (
    BaseLLMProvider,
    LLMInterface,
    LLMResponse,
    LLMTimeoutError,
    Message,
    ToolDefinition,
)
from syspilot.core.openai_auth_provider import OpenAIAuthProvider


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


class FlakyProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        _ = messages
        _ = tools
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary outage")
        return LLMResponse(content="recovered")

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        _ = messages
        _ = tools
        _ = kwargs

        async def _gen() -> AsyncIterator[str]:
            yield "recovered"

        return _gen()


class HangingProvider(BaseLLMProvider):
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        _ = messages
        _ = tools
        _ = kwargs
        await asyncio.sleep(60)
        return LLMResponse(content="never")

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        _ = messages
        _ = tools
        _ = kwargs

        async def _gen() -> AsyncIterator[str]:
            await asyncio.sleep(60)
            yield "never"

        return _gen()


class _HangingOpenAIAuthStream:
    async def __aenter__(self) -> "_HangingOpenAIAuthStream":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type
        _ = exc
        _ = tb

    def __aiter__(self) -> "_HangingOpenAIAuthStream":
        return self

    async def __anext__(self) -> object:
        await asyncio.sleep(60)
        raise StopAsyncIteration

    async def get_final_response(self) -> object:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")


class _FakeOpenAIAuthResponses:
    def stream(self, **kwargs: Any) -> _HangingOpenAIAuthStream:
        _ = kwargs
        return _HangingOpenAIAuthStream()


class _FakeOpenAIAuthClient:
    def __init__(self) -> None:
        self.responses = _FakeOpenAIAuthResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


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


@pytest.mark.asyncio
async def test_llm_interface_retries_transient_chat_failure() -> None:
    llm = LLMInterface(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key="test",
            timeout_seconds=0.05,
            max_retries=1,
        )
    )
    provider = FlakyProvider()
    llm._provider = provider

    response = await llm.chat([Message(role="user", content="hello")])

    assert response.content == "recovered"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_llm_interface_times_out_hung_chat_provider() -> None:
    llm = LLMInterface(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key="test",
            timeout_seconds=0.01,
            max_retries=0,
        )
    )
    llm._provider = HangingProvider()

    with pytest.raises(LLMTimeoutError):
        await llm.chat([Message(role="user", content="hello")])


@pytest.mark.asyncio
async def test_openai_auth_chat_times_out_hung_stream() -> None:
    provider = OpenAIAuthProvider(
        LLMConfig(
            provider=LLMProvider.OPENAI_AUTH,
            model="gpt-5.3-codex",
            timeout_seconds=0.01,
            max_retries=0,
        )
    )
    client = _FakeOpenAIAuthClient()

    async def _fake_build_client() -> _FakeOpenAIAuthClient:
        return client

    setattr(provider, "_build_client", _fake_build_client)

    with pytest.raises(LLMTimeoutError):
        await provider.chat([Message(role="user", content="hello")])

    assert client.closed is True


@pytest.mark.asyncio
async def test_openai_auth_stream_chat_times_out_hung_stream() -> None:
    provider = OpenAIAuthProvider(
        LLMConfig(
            provider=LLMProvider.OPENAI_AUTH,
            model="gpt-5.3-codex",
            timeout_seconds=0.01,
            max_retries=0,
        )
    )
    client = _FakeOpenAIAuthClient()

    async def _fake_build_client() -> _FakeOpenAIAuthClient:
        return client

    setattr(provider, "_build_client", _fake_build_client)

    with pytest.raises(LLMTimeoutError):
        async for _ in provider.stream_chat([Message(role="user", content="hello")]):
            pass

    assert client.closed is True
