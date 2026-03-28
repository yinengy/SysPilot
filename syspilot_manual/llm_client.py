from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from syspilot.core.config import LLMConfig
from syspilot.core.llm import LLMError, LLMInterface, Message


class ManualLLMProtocol(Protocol):
    async def complete_prompt(self, prompt_name: str, payload: dict[str, Any]) -> str: ...


class ManualLLMRequestError(RuntimeError):
    pass


class ManualLLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        prompt_dir: Path | None = None,
        interface: LLMInterface | None = None,
    ) -> None:
        self._interface = interface or LLMInterface(config or LLMConfig())
        self._prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"

    def load_prompt(self, prompt_name: str) -> str:
        return (self._prompt_dir / prompt_name).read_text(encoding="utf-8")

    async def complete_prompt(self, prompt_name: str, payload: dict[str, Any]) -> str:
        prompt = self.load_prompt(prompt_name)
        messages = [
            Message(role="system", content=prompt),
            Message(
                role="user",
                content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            ),
        ]
        try:
            response = await self._interface.chat(messages)
        except LLMError as exc:
            raise ManualLLMRequestError(f"Manual prompt '{prompt_name}' failed: {exc}") from exc
        return response.content


def extract_json_object(raw_text: str) -> Any:
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("Empty LLM response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end_char = "}" if stripped[start] == "{" else "]"
        end = stripped.rfind(end_char)
        if end < start:
            raise
        return json.loads(stripped[start : end + 1])
