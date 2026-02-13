"""
LLM Interface Module

Provides a unified interface for interacting with different LLM providers.
Supports OpenAI, Anthropic, and local models.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from syspilot.core.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


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
        
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)
        
        response = await self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
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
        
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)
        
        async for chunk in await self.client.chat.completions.create(**request_kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


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
    
    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
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
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        
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
            delta_messages = messages[self._debug_last_message_count:]
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

        response = await self.provider.chat(messages, tools, **kwargs)

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
        async for chunk in self.provider.stream_chat(messages, tools, **kwargs):
            yield chunk
