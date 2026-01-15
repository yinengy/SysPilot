"""
SysPilot Agent - Core Agent Framework

The main agent that orchestrates system monitoring, profiling, and tuning.
Uses an LLM to dynamically generate scripts and make tuning decisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from syspilot.core.config import SysPilotConfig
from syspilot.core.llm import LLMInterface, Message, ToolDefinition, LLMResponse

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent operational states."""
    IDLE = "idle"
    MONITORING = "monitoring"
    PROFILING = "profiling"
    ANALYZING = "analyzing"
    TUNING = "tuning"
    ERROR = "error"


@dataclass
class ProfileResult:
    """Result from a profiling operation."""
    timestamp: datetime
    source: str  # e.g., "cpu_collector", "ebpf_script", "custom_script"
    data: dict[str, Any]
    summary: str  # Condensed summary for LLM context
    raw_size_bytes: int
    summary_size_bytes: int


@dataclass
class TuningRecommendation:
    """A tuning recommendation from the agent."""
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float  # 0.0 to 1.0
    impact: str  # "low", "medium", "high"


@dataclass
class AgentContext:
    """Context maintained by the agent across interactions."""
    target_pid: int | None = None
    target_name: str | None = None
    profile_history: list[ProfileResult] = field(default_factory=list)
    tuning_history: list[TuningRecommendation] = field(default_factory=list)
    active_scripts: dict[str, str] = field(default_factory=dict)  # name -> script content
    system_baseline: dict[str, Any] = field(default_factory=dict)
    
    def get_context_summary(self, max_profiles: int = 5) -> str:
        """Generate a condensed context summary for LLM."""
        lines = []
        
        if self.target_pid:
            lines.append(f"Target Process: PID {self.target_pid}")
        if self.target_name:
            lines.append(f"Target Name: {self.target_name}")
        
        if self.system_baseline:
            lines.append(f"System Baseline: {json.dumps(self.system_baseline, indent=2)}")
        
        # Include recent profile summaries (not raw data)
        recent_profiles = self.profile_history[-max_profiles:]
        if recent_profiles:
            lines.append("\nRecent Profile Summaries:")
            for p in recent_profiles:
                lines.append(f"  [{p.timestamp.isoformat()}] {p.source}: {p.summary}")
        
        return "\n".join(lines)


class SysPilotAgent:
    """
    Main SysPilot Agent.
    
    Orchestrates system monitoring and application tuning using an LLM
    to dynamically generate profiling scripts and make tuning decisions.
    """
    
    SYSTEM_PROMPT = """You are SysPilot, an AI agent specialized in system monitoring and application performance tuning.

Your capabilities:
1. Monitor system resources (CPU, memory, cache, I/O)
2. Profile target applications to understand their resource usage patterns
3. Write custom scripts (Python, Bash, eBPF) to gather specific profiling data
4. Analyze profiling results and identify performance bottlenecks
5. Recommend and apply configuration tuning

Key principles:
- MINIMIZE CONTEXT: Always summarize large profiling outputs into essential insights
- DYNAMIC SCRIPTING: Write scripts to process raw data (strace, perf, etc.) and extract key metrics
- eBPF PROFICIENCY: Use eBPF for low-overhead, precise system profiling when needed
- CONSERVATIVE TUNING: Recommend changes with clear reasoning and confidence levels

When writing profiling scripts:
- Process raw data and output only essential metrics
- Include error handling and timeouts
- For eBPF: use BCC Python bindings or bpftrace

Available tools will be provided for each interaction."""

    def __init__(self, config: SysPilotConfig) -> None:
        self.config = config
        self.llm = LLMInterface(config.llm)
        self.state = AgentState.IDLE
        self.context = AgentContext(
            target_pid=config.target_pid,
            target_name=config.target_name,
        )
        self._tools: dict[str, Callable[..., Any]] = {}
        self._tool_definitions: list[ToolDefinition] = []
        self._message_history: list[Message] = []
        self._running = False
        
        # Initialize with system prompt
        self._message_history.append(Message(role="system", content=self.SYSTEM_PROMPT))
    
    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool that the agent can use."""
        self._tools[name] = func
        self._tool_definitions.append(ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        ))
        logger.info(f"Registered tool: {name}")
    
    def _build_context_message(self) -> str:
        """Build current context for the LLM."""
        return f"""Current Agent State: {self.state.value}

{self.context.get_context_summary()}

Respond with your analysis and any tool calls needed."""

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        
        func = self._tools[name]
        logger.debug(f"Executing tool: {name} with args: {arguments}")
        
        # Handle both sync and async functions
        result = func(**arguments)
        if asyncio.iscoroutine(result):
            result = await result
        
        return result

    async def _process_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[Message]:
        """Process tool calls from LLM response."""
        results = []
        
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            arguments = json.loads(tc["function"]["arguments"])
            
            try:
                result = await self._execute_tool(tool_name, arguments)
                result_str = json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                result_str = f"Error: {str(e)}"
            
            results.append(Message(
                role="tool",
                content=result_str,
                tool_call_id=tc["id"],
            ))
        
        return results

    async def step(self, user_input: str | None = None) -> LLMResponse:
        """Execute one step of the agent loop."""
        # Add user input if provided
        if user_input:
            self._message_history.append(Message(role="user", content=user_input))
        else:
            # Add context update
            context_msg = self._build_context_message()
            self._message_history.append(Message(role="user", content=context_msg))
        
        # Get LLM response
        response = await self.llm.chat(
            messages=self._message_history,
            tools=self._tool_definitions if self._tool_definitions else None,
        )
        
        # Add assistant response to history
        self._message_history.append(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        ))
        
        # Process tool calls if any
        if response.tool_calls:
            tool_results = await self._process_tool_calls(response.tool_calls)
            self._message_history.extend(tool_results)
            
            # Continue conversation after tool execution
            return await self.step()
        
        return response

    async def run_monitoring_loop(self) -> None:
        """Run the main monitoring loop."""
        self._running = True
        self.state = AgentState.MONITORING
        logger.info("Starting monitoring loop")
        
        while self._running:
            try:
                # Trigger a monitoring step
                await self.step("Check system status and profile target if needed.")
                
                # Wait for next interval
                await asyncio.sleep(self.config.monitoring.interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.state = AgentState.ERROR
                await asyncio.sleep(5)  # Brief pause before retry
                self.state = AgentState.MONITORING

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        self.state = AgentState.IDLE
        logger.info("Stopping monitoring loop")

    async def analyze_target(self, pid: int | None = None, name: str | None = None) -> str:
        """Analyze a target application."""
        if pid:
            self.context.target_pid = pid
        if name:
            self.context.target_name = name
        
        self.state = AgentState.ANALYZING
        
        prompt = f"""Analyze the target application:
- PID: {self.context.target_pid or 'Not specified'}
- Name: {self.context.target_name or 'Not specified'}

1. First, gather basic process information
2. Profile CPU and memory usage
3. Identify any obvious performance issues
4. Suggest profiling scripts if deeper analysis is needed

Provide a summary of findings."""

        response = await self.step(prompt)
        self.state = AgentState.IDLE
        return response.content

    async def tune_application(self, recommendations: list[TuningRecommendation] | None = None) -> str:
        """Apply tuning recommendations."""
        self.state = AgentState.TUNING
        
        if recommendations:
            rec_text = "\n".join([
                f"- {r.parameter}: {r.current_value} -> {r.recommended_value} ({r.reason})"
                for r in recommendations
            ])
            prompt = f"""Apply the following tuning recommendations:
{rec_text}

Dry run mode: {self.config.tuning.dry_run}
Backup configs: {self.config.tuning.backup_configs}

Execute the tuning and report results."""
        else:
            prompt = """Based on the current profiling data, suggest and apply tuning recommendations.
Be conservative and explain each change."""

        response = await self.step(prompt)
        self.state = AgentState.IDLE
        return response.content

    def add_profile_result(self, result: ProfileResult) -> None:
        """Add a profiling result to context."""
        self.context.profile_history.append(result)
        logger.debug(f"Added profile result from {result.source}: {result.summary[:100]}...")
