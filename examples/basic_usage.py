#!/usr/bin/env python3
"""
Example: Basic usage of SysPilot agent.

This example demonstrates how to:
1. Create and configure a SysPilot agent
2. Register custom tools
3. Collect a local profile snapshot that can inform spawn-mode tuning
"""

import asyncio
import os

from syspilot import SysPilotAgent, SysPilotConfig
from syspilot.core.config import LLMProvider, LLMConfig
from syspilot.profiling import ProfilingManager
from syspilot.skills import SkillRegistry
from syspilot.skills.builtin import DynamicScriptSkill, StraceAnalyzerSkill


async def main() -> None:
    config = SysPilotConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        target_pid=os.getpid(),  # Monitor ourselves for demo
    )
    
    agent = SysPilotAgent(config)
    manager = ProfilingManager()
    
    registry = SkillRegistry()
    registry.register(DynamicScriptSkill())
    registry.register(StraceAnalyzerSkill())
    
    async def collect_profile(collectors: list[str] | None = None, pid: int | None = None):
        results = await manager.collect(collectors, pid)
        return {name: result.summarize() for name, result in results.items()}
    
    agent.register_tool(
        name="collect_profile",
        func=collect_profile,
        description="Collect system profiling data",
        parameters={
            "type": "object",
            "properties": {
                "collectors": {"type": "array", "items": {"type": "string"}},
                "pid": {"type": "integer"},
            },
        },
    )
    
    print("Collecting initial profile...")
    results = await manager.collect_all(config.target_pid)
    print(manager.summarize_results(results))
    
    print("\nSpawn-mode tuning is the supported workflow.")
    print('Use `syspilot tune --command "..."` to run iterative benchmark tuning.')


if __name__ == "__main__":
    asyncio.run(main())
