"""
LLM Interface Module

Provides a unified interface for interacting with different LLM providers.
Supports OpenAI, Anthropic, and local models.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from syspilot.core.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMTimeoutError(LLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class LLMProviderError(LLMError):
    pass


def _is_transient_exception(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True

    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "InternalServerError",
        "NetworkError",
        "ProtocolError",
        "RateLimitError",
        "ReadError",
        "RemoteProtocolError",
        "TimeoutException",
        "TransportError",
    }
    return exc.__class__.__name__ in transient_names


def _format_timeout_seconds(timeout_seconds: float) -> str:
    return f"{timeout_seconds:g}s"


@dataclass
class Message:
    """Represents a chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolDefinition:
    """Defines a tool that the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        result = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            result.append(m)
        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert internal tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _request_timeout_seconds(self) -> float:
        return max(float(self.config.timeout_seconds), 0.001)

    def _should_use_stream_fallback(self) -> bool:
        return self.config.model.strip().lower().startswith("gpt-5")

    def _extract_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                text = part.get("text")
            else:
                part_type = getattr(part, "type", None)
                text = getattr(part, "text", None)

            if not isinstance(text, str) or not text:
                continue

            if part_type in (None, "text", "input_text", "output_text"):
                text_parts.append(text)

        return "".join(text_parts)

    def _convert_tool_calls(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for tc in raw_tool_calls or []:
            function = getattr(tc, "function", None)
            if function is None and isinstance(tc, dict):
                function = tc.get("function")

            call_id = getattr(tc, "id", None)
            if call_id is None and isinstance(tc, dict):
                call_id = tc.get("id")

            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments")

            if not isinstance(call_id, str) or not isinstance(name, str):
                continue

            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else "{}",
                    },
                }
            )
        return tool_calls

    def _extract_usage(self, usage: Any) -> dict[str, int] | None:
        if usage is None:
            return None

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        if prompt_tokens is None or completion_tokens is None or total_tokens is None:
            return None

        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

    async def _consume_streamed_chat_completion(
        self, stream: AsyncIterator[Any]
    ) -> tuple[str, list[dict[str, Any]], str, dict[str, int] | None]:
        content_parts: list[str] = []
        partial_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] | None = None

        async for chunk in stream:
            chunk_usage = self._extract_usage(getattr(chunk, "usage", None))
            if chunk_usage is not None:
                usage = chunk_usage

            for choice in getattr(chunk, "choices", []) or []:
                current_finish_reason = getattr(choice, "finish_reason", None)
                if isinstance(current_finish_reason, str) and current_finish_reason:
                    finish_reason = current_finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                delta_content = getattr(delta, "content", None)
                text = self._extract_message_content(delta_content)
                if not text and isinstance(delta_content, str):
                    text = delta_content
                if text:
                    content_parts.append(text)

                for tc in getattr(delta, "tool_calls", None) or []:
                    if isinstance(tc, dict):
                        index = tc.get("index", 0)
                        call_id = tc.get("id")
                        call_type = tc.get("type")
                        function = tc.get("function") or {}
                    else:
                        index = getattr(tc, "index", 0)
                        call_id = getattr(tc, "id", None)
                        call_type = getattr(tc, "type", None)
                        function = getattr(tc, "function", None) or {}

                    existing = partial_tool_calls.setdefault(
                        int(index),
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    if isinstance(call_id, str) and call_id:
                        existing["id"] = call_id
                    if isinstance(call_type, str) and call_type:
                        existing["type"] = call_type

                    if isinstance(function, dict):
                        name = function.get("name")
                        arguments = function.get("arguments")
                    else:
                        name = getattr(function, "name", None)
                        arguments = getattr(function, "arguments", None)

                    if isinstance(name, str) and name:
                        existing["function"]["name"] = name
                    if isinstance(arguments, str) and arguments:
                        existing["function"]["arguments"] += arguments

        tool_calls = [
            item
            for _, item in sorted(partial_tool_calls.items())
            if isinstance(item.get("id"), str)
            and item["id"]
            and isinstance(item.get("function", {}).get("name"), str)
            and item["function"]["name"]
        ]
        return "".join(content_parts), tool_calls, finish_reason, usage

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to OpenAI."""
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }
        request_kwargs.setdefault("timeout", self._request_timeout_seconds())

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        if self._should_use_stream_fallback():
            stream = await self.client.chat.completions.create(**{**request_kwargs, "stream": True})
            (
                streamed_content,
                streamed_tool_calls,
                streamed_finish_reason,
                streamed_usage,
            ) = await self._consume_streamed_chat_completion(stream)
            return LLMResponse(
                content=streamed_content,
                tool_calls=streamed_tool_calls or None,
                finish_reason=streamed_finish_reason,
                usage=streamed_usage,
            )

        response = await self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]

        tool_calls = self._convert_tool_calls(choice.message.tool_calls)
        content = self._extract_message_content(choice.message.content)
        finish_reason = choice.finish_reason or "stop"
        usage = self._extract_usage(response.usage)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response from OpenAI."""
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            **kwargs,
        }
        request_kwargs.setdefault("timeout", self._request_timeout_seconds())

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        async for chunk in await self.client.chat.completions.create(**request_kwargs):
            if not chunk.choices:
                continue
            delta_content = getattr(chunk.choices[0].delta, "content", None)
            text = self._extract_message_content(delta_content)
            if not text and isinstance(delta_content, str):
                text = delta_content
            if text:
                yield text


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-load Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.config.api_key)
        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format."""
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                converted.append({"role": msg.role, "content": msg.content})

        return system_prompt, converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert internal tools to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Anthropic."""
        system_prompt, converted_messages = self._convert_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": converted_messages,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        response = await self.client.messages.create(**request_kwargs)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response from Anthropic."""
        system_prompt, converted_messages = self._convert_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": converted_messages,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                yield text


class LLMInterface:
    """Unified interface for LLM interactions."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._provider: BaseLLMProvider | None = None
        self._debug_markdown_path: Path | None = None
        self._debug_exchange_index = 0
        self._debug_last_message_count = 0

        if self.config.debug_conversation:
            if self.config.debug_markdown_file:
                path = Path(self.config.debug_markdown_file).expanduser()
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path("~/.syspilot/logs").expanduser() / f"llm_debug_{ts}.md"

            path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_markdown_path = path
            path.write_text(
                "# SysPilot LLM Debug Transcript\n\n"
                f"- Started: {datetime.now().isoformat()}\n"
                f"- Provider: {self.config.provider.value}\n"
                f"- Model: {self.config.model}\n\n",
                encoding="utf-8",
            )
            logger.info("LLM debug markdown output: %s", path)

    def _request_timeout_seconds(self) -> float:
        return max(float(self.config.timeout_seconds), 0.001)

    def _retry_attempts(self) -> int:
        return max(int(self.config.max_retries), 0) + 1

    def _retry_delay_seconds(self, attempt: int) -> float:
        return min(0.5 * attempt, 1.5)

    def _normalize_provider_error(self, exc: BaseException) -> LLMError:
        if isinstance(exc, LLMError):
            return exc

        timeout_seconds = self._request_timeout_seconds()
        if isinstance(exc, asyncio.TimeoutError):
            return LLMTimeoutError(
                "LLM request timed out "
                f"after {_format_timeout_seconds(timeout_seconds)} "
                f"({self.config.provider.value}/{self.config.model})"
            )

        message = str(exc).strip() or exc.__class__.__name__
        return LLMProviderError(
            f"LLM provider request failed ({self.config.provider.value}/{self.config.model}): {message}",
            retryable=_is_transient_exception(exc),
        )

    async def _call_chat_once(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        return await asyncio.wait_for(
            self.provider.chat(messages, tools, **kwargs),
            timeout=self._request_timeout_seconds(),
        )

    def _message_to_debug_dict(self, msg: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
        }
        if msg.tool_calls is not None:
            payload["tool_calls"] = msg.tool_calls
        if msg.tool_call_id is not None:
            payload["tool_call_id"] = msg.tool_call_id
        return payload

    def _log_debug_request(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        kwargs: dict[str, Any],
    ) -> None:
        payload = {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "tools": [t.name for t in tools] if tools else [],
            "kwargs": kwargs,
            "messages": [self._message_to_debug_dict(m) for m in messages],
        }
        logger.info("LLM_DEBUG_REQUEST\n%s", json.dumps(payload, indent=2))

    def _log_debug_response(self, response: LLMResponse) -> None:
        payload = {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }
        logger.info("LLM_DEBUG_RESPONSE\n%s", json.dumps(payload, indent=2))

    def _append_markdown_debug(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        kwargs: dict[str, Any],
        response: LLMResponse,
    ) -> None:
        if self._debug_markdown_path is None:
            return

        self._debug_exchange_index += 1

        if len(messages) >= self._debug_last_message_count:
            delta_messages = messages[self._debug_last_message_count :]
        else:
            delta_messages = messages

        self._debug_last_message_count = len(messages)

        lines: list[str] = []
        lines.append(f"## Exchange {self._debug_exchange_index}")
        lines.append("")
        lines.append(f"- Provider: `{self.config.provider.value}`")
        lines.append(f"- Model: `{self.config.model}`")
        lines.append(f"- Temperature: `{self.config.temperature}`")
        lines.append(f"- Max Tokens: `{self.config.max_tokens}`")
        lines.append(
            f"- Tools Available: `{', '.join(t.name for t in tools) if tools else '(none)'}`"
        )
        lines.append("")
        lines.append("### Request Arguments")
        lines.append("````json")
        lines.append(json.dumps(kwargs, indent=2))
        lines.append("````")
        lines.append("")
        lines.append("### New Messages Sent")

        if not delta_messages:
            lines.append("(no new messages)")
            lines.append("")
        else:
            for msg in delta_messages:
                lines.append(f"#### {msg.role}")
                lines.append("````text")
                lines.append(msg.content)
                lines.append("````")
                if msg.tool_calls is not None:
                    lines.append("Tool Calls:")
                    lines.append("````json")
                    lines.append(json.dumps(msg.tool_calls, indent=2))
                    lines.append("````")
                if msg.tool_call_id is not None:
                    lines.append(f"Tool Call ID: `{msg.tool_call_id}`")
                lines.append("")

        lines.append("### Response")
        lines.append(f"- Finish Reason: `{response.finish_reason}`")
        lines.append("- Usage:")
        lines.append("````json")
        lines.append(json.dumps(response.usage or {}, indent=2))
        lines.append("````")
        lines.append("")
        lines.append("Content:")
        lines.append("````text")
        lines.append(response.content)
        lines.append("````")
        lines.append("")

        if response.tool_calls is not None:
            lines.append("Tool Calls:")
            lines.append("````json")
            lines.append(json.dumps(response.tool_calls, indent=2))
            lines.append("````")
            lines.append("")

        with self._debug_markdown_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @property
    def provider(self) -> BaseLLMProvider:
        """Get the appropriate LLM provider."""
        if self._provider is None:
            if self.config.provider == LLMProvider.OPENAI:
                self._provider = OpenAIProvider(self.config)
            elif self.config.provider == LLMProvider.OPENAI_AUTH:
                from syspilot.core.openai_auth_provider import OpenAIAuthProvider

                self._provider = OpenAIAuthProvider(self.config)
            elif self.config.provider == LLMProvider.ANTHROPIC:
                self._provider = AnthropicProvider(self.config)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        return self._provider

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        logger.debug(f"Sending chat request with {len(messages)} messages")
        if self.config.debug_conversation:
            self._log_debug_request(messages, tools, kwargs)

        last_error: LLMError | None = None
        total_attempts = self._retry_attempts()

        for attempt in range(1, total_attempts + 1):
            try:
                response = await self._call_chat_once(messages, tools, kwargs)
                break
            except BaseException as exc:
                last_error = self._normalize_provider_error(exc)
                should_retry = last_error.retryable and attempt < total_attempts
                if not should_retry:
                    raise last_error from exc

                logger.warning(
                    "Retrying LLM request after attempt %s/%s failed: %s",
                    attempt,
                    total_attempts,
                    last_error,
                )
                await asyncio.sleep(self._retry_delay_seconds(attempt))
        else:
            assert last_error is not None
            raise last_error

        if self.config.debug_conversation:
            self._log_debug_response(response)
            self._append_markdown_debug(messages, tools, kwargs, response)

        return response

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        logger.debug(f"Streaming chat request with {len(messages)} messages")
        provider_stream = self.provider.stream_chat(messages, tools, **kwargs)

        while True:
            try:
                chunk = await asyncio.wait_for(
                    anext(provider_stream),
                    timeout=self._request_timeout_seconds(),
                )
            except StopAsyncIteration:
                break
            except BaseException as exc:
                raise self._normalize_provider_error(exc) from exc
            else:
                yield chunk
