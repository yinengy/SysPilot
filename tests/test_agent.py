from __future__ import annotations

import pytest

from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig


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
