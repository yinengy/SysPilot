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
        return await self.provider.chat(messages, tools, **kwargs)
    
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
