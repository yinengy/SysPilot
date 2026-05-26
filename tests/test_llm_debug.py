from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from syspilot.core.config import LLMConfig, LLMProvider
from syspilot.core.llm import (
    BaseLLMProvider,
    LLMInterface,
    LLMResponse,
    LLMTimeoutError,
    Message,
    OpenAIProvider,
    ToolDefinition,
    _is_transient_exception,
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


class _FakeOpenAIStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "_FakeOpenAIStream":
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _FakeOpenAIChatCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeOpenAIClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeOpenAIChatCompletions(responses))


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


@pytest.mark.asyncio
async def test_openai_provider_uses_stream_first_for_gpt5_completion() -> None:
    provider = OpenAIProvider(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-5.4",
            api_key="test",
            timeout_seconds=12.5,
        )
    )
    stream = _FakeOpenAIStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="OK", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
    )
    provider._client = _FakeOpenAIClient([stream])

    response = await provider.chat([Message(role="user", content="hello")])

    assert response.content == "OK"
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert len(provider.client.chat.completions.calls) == 1
    assert provider.client.chat.completions.calls[0]["stream"] is True
    assert provider.client.chat.completions.calls[0]["timeout"] == 12.5


@pytest.mark.asyncio
async def test_openai_provider_extracts_text_from_typed_content_parts() -> None:
    provider = OpenAIProvider(
        LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o", api_key="test")
    )
    typed_content_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "output_text", "text": "hello"},
                        {"type": "text", "text": " world"},
                    ],
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    provider._client = _FakeOpenAIClient([typed_content_response])

    response = await provider.chat([Message(role="user", content="hello")])

    assert response.content == "hello world"


@pytest.mark.asyncio
async def test_openai_provider_passes_timeout_to_stream_chat_request() -> None:
    provider = OpenAIProvider(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-5.4",
            api_key="test",
            timeout_seconds=7.25,
        )
    )
    stream = _FakeOpenAIStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hello", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        ]
    )
    provider._client = _FakeOpenAIClient([stream])

    chunks: list[str] = []
    async for chunk in provider.stream_chat([Message(role="user", content="hello")]):
        chunks.append(chunk)

    assert chunks == ["hello"]
    assert provider.client.chat.completions.calls[0]["stream"] is True
    assert provider.client.chat.completions.calls[0]["timeout"] == 7.25


def test_empty_stream_message_is_retryable_provider_failure() -> None:
    assert _is_transient_exception(
        RuntimeError("empty_stream: upstream stream closed before first payload")
    )
