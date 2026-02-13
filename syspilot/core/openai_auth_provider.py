from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

from syspilot.core.config import LLMConfig
from syspilot.core.llm import BaseLLMProvider, LLMResponse, Message, ToolDefinition
from syspilot.core.openai_auth import (
    ACCOUNT_CLAIM,
    CLIENT_ID,
    CODEX_BASE_URL,
    SUPPORTED_CODEX_MODELS,
    TOKEN_URL,
    opencode_openai_auth_file,
    syspilot_openai_auth_file,
)


class OpenAIAuthProvider(BaseLLMProvider):
    TOKEN_URL = TOKEN_URL
    CLIENT_ID = CLIENT_ID
    CODEX_BASE_URL = CODEX_BASE_URL
    ACCOUNT_CLAIM = ACCOUNT_CLAIM

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _resolve_model(self) -> str:
        normalized = (self.config.model or "").strip().lower()
        if not normalized:
            return "gpt-5.3-codex"

        aliases = {
            "gpt-5.2": "gpt-5.2-codex",
            "gpt-5.2-codex": "gpt-5.2-codex",
            "gpt-5.3": "gpt-5.3-codex",
            "gpt-5.3-codex": "gpt-5.3-codex",
        }
        if normalized in aliases:
            return aliases[normalized]

        raise ValueError(
            "OpenAI auth provider supports only "
            f"{SUPPORTED_CODEX_MODELS[0]} or {SUPPORTED_CODEX_MODELS[1]}"
        )

    @property
    def auth_path(self) -> Path:
        if self.config.openai_auth_file:
            return Path(self.config.openai_auth_file).expanduser()

        syspilot_path = syspilot_openai_auth_file()
        if syspilot_path.exists():
            return syspilot_path

        legacy_opencode = opencode_openai_auth_file()
        if legacy_opencode.exists():
            return legacy_opencode

        return syspilot_path

    def _load_auth(self) -> dict[str, Any]:
        if not self.auth_path.exists():
            raise ValueError(
                f"OpenAI auth file not found at {self.auth_path}. "
                "Run 'syspilot auth login' first."
            )

        with open(self.auth_path, "r") as f:
            data = json.load(f)

        if data.get("type") != "oauth":
            raise ValueError(
                f"Unsupported auth type in {self.auth_path}: {data.get('type')}"
            )

        access = data.get("access")
        refresh = data.get("refresh")
        expires = data.get("expires")
        if not access or not refresh or expires is None:
            raise ValueError(
                f"Invalid OpenAI auth file at {self.auth_path}: "
                "expected access, refresh, and expires fields"
            )

        return {
            "type": "oauth",
            "access": str(access),
            "refresh": str(refresh),
            "expires": int(expires),
        }

    def _write_auth(self, auth: dict[str, Any]) -> None:
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.auth_path, "w") as f:
            json.dump(auth, f, indent=2)
        os.chmod(self.auth_path, 0o600)

    def _is_expired(self, auth: dict[str, Any]) -> bool:
        refresh_buffer_ms = 60_000
        return int(auth["expires"]) <= int(time.time() * 1000) + refresh_buffer_ms

    async def _refresh_auth(self, auth: dict[str, Any]) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": auth["refresh"],
                    "client_id": self.CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code >= 400:
            raise ValueError(
                "Failed to refresh OpenAI token "
                f"(status {response.status_code}): {response.text}"
            )

        payload = response.json()
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        expires_in_raw = payload.get("expires_in")

        try:
            expires_in = int(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = None

        if not access or not refresh or expires_in is None:
            raise ValueError("OpenAI refresh response missing required fields")

        updated = {
            "type": "oauth",
            "access": access,
            "refresh": refresh,
            "expires": int(time.time() * 1000) + expires_in * 1000,
        }
        self._write_auth(updated)
        return updated

    def _extract_account_id(self, access_token: str) -> str | None:
        try:
            token_parts = access_token.split(".")
            if len(token_parts) != 3:
                return None

            payload_raw = token_parts[1]
            payload_raw += "=" * ((4 - len(payload_raw) % 4) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_raw).decode("utf-8"))
            claim = payload.get(self.ACCOUNT_CLAIM)
            if isinstance(claim, dict):
                account_id = claim.get("chatgpt_account_id")
                return str(account_id) if account_id else None
        except Exception:
            return None
        return None

    async def _resolve_auth(self) -> dict[str, Any]:
        auth = self._load_auth()
        if self._is_expired(auth):
            auth = await self._refresh_auth(auth)
        return auth

    async def _build_client(self) -> Any:
        from openai import AsyncOpenAI

        auth = await self._resolve_auth()
        account_id = self._extract_account_id(auth["access"])
        if not account_id:
            raise ValueError(
                "Could not extract chatgpt_account_id from OpenAI token. "
                "Re-run 'syspilot auth login'."
            )

        base_url = self.config.base_url or self.CODEX_BASE_URL
        return AsyncOpenAI(
            api_key=auth["access"],
            base_url=base_url,
            default_headers={
                "OpenAI-Beta": "responses=experimental",
                "chatgpt-account-id": account_id,
                "originator": "codex_cli_rs",
                "accept": "text/event-stream",
            },
        )

    def _prepare_input(self, messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
        converted: list[dict[str, Any]] = []
        instructions_parts: list[str] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    instructions_parts.append(msg.content)
                continue

            if msg.role == "tool":
                if msg.tool_call_id:
                    converted.append(
                        {
                            "type": "function_call_output",
                            "call_id": msg.tool_call_id,
                            "output": msg.content,
                        }
                    )
                elif msg.content:
                    converted.append({"role": "assistant", "content": msg.content})
                continue

            if msg.content:
                converted.append({"role": msg.role, "content": msg.content})

            if msg.role == "assistant" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    function = tool_call.get("function", {})
                    call_id = tool_call.get("id")
                    name = function.get("name")
                    arguments = function.get("arguments", "{}")

                    if call_id and name:
                        converted.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments,
                            }
                        )

        instructions = "\n\n".join(instructions_parts).strip()
        if not instructions:
            instructions = "You are a helpful assistant."
        return converted, instructions

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]

    def _extract_content(self, response: Any) -> str:
        content = getattr(response, "output_text", None)
        if isinstance(content, str):
            return content

        text_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    text = getattr(part, "text", "")
                    if text:
                        text_parts.append(text)
        return "".join(text_parts)

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue

            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            name = getattr(item, "name", None)
            arguments = getattr(item, "arguments", "{}")
            if call_id and name:
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                )
        return tool_calls

    def _extract_usage(self, response: Any) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if prompt_tokens is None or completion_tokens is None or total_tokens is None:
            return None

        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

    def _resolve_include(self, kwargs: dict[str, Any]) -> list[str]:
        include = kwargs.pop("include", [])
        include_values = include if isinstance(include, list) else []
        normalized = [str(item) for item in include_values if item]
        if "reasoning.encrypted_content" not in normalized:
            normalized.append("reasoning.encrypted_content")
        return normalized

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._build_client()
        try:
            input_items, instructions = self._prepare_input(messages)
            request_kwargs: dict[str, Any] = {
                "model": self._resolve_model(),
                "input": input_items,
                "instructions": instructions,
                "store": False,
                "include": self._resolve_include(kwargs),
                **kwargs,
            }

            if tools:
                request_kwargs["tools"] = self._convert_tools(tools)

            async with client.responses.stream(**request_kwargs) as stream:
                async for _ in stream:
                    pass
                response = await stream.get_final_response()

            tool_calls = self._extract_tool_calls(response)
            return LLMResponse(
                content=self._extract_content(response),
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=getattr(response, "status", "stop") or "stop",
                usage=self._extract_usage(response),
            )
        finally:
            await client.close()

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = await self._build_client()
        try:
            input_items, instructions = self._prepare_input(messages)
            request_kwargs: dict[str, Any] = {
                "model": self._resolve_model(),
                "input": input_items,
                "instructions": instructions,
                "store": False,
                "include": self._resolve_include(kwargs),
                **kwargs,
            }

            if tools:
                request_kwargs["tools"] = self._convert_tools(tools)

            async with client.responses.stream(**request_kwargs) as stream:
                async for event in stream:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            yield delta
        finally:
            await client.close()
