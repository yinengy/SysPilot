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
from pathlib import Path
from typing import Any, Callable

from syspilot.core.config import SysPilotConfig
from syspilot.core.llm import LLMError, LLMInterface, Message, ToolDefinition, LLMResponse

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
    loaded_manuals: dict[str, str] = field(default_factory=dict)

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
1. Monitor system resources (CPU, memory, cache, I/O, NUMA topology)
2. Profile target applications to understand their resource usage patterns
3. Write custom scripts (Python, Bash, eBPF) to gather specific profiling data
4. Analyze profiling results and identify performance bottlenecks
5. Recommend and apply configuration tuning based on loaded manuals
6. Iteratively tune applications: profile → apply → re-profile → compare → iterate

Key principles:
- MINIMIZE CONTEXT: Always summarize large profiling outputs into essential insights
- DYNAMIC SCRIPTING: Write scripts to process raw data (strace, perf, etc.) and extract key metrics
- eBPF PROFICIENCY: Use eBPF for low-overhead, precise system profiling when needed
- CONSERVATIVE TUNING: Recommend changes with clear reasoning and confidence levels
- MANUAL-GUIDED TUNING: When manuals are loaded, use them to identify tuning opportunities
- ITERATIVE TUNING: Run profile-apply-measure loops and roll back regressions automatically

When to use eBPF:
- You need low-overhead kernel-level tracing (syscall patterns, page faults, context switches)
- Standard profiling tools don't provide sufficient granularity
- You need to trace NUMA page migrations, cache misses, or memory allocation patterns
- The target application is latency-sensitive and cannot tolerate strace overhead

Dynamic Script Generation pipeline:
1. CAPTURE: Run a command and save its raw output to a temp file
2. SAMPLE: Read the first few lines to understand the output format
3. GENERATE: Write a Python processing script to compress the raw output into key insights
4. EXECUTE: Run the processing script against the captured data
5. RETURN: Provide the compressed result to minimize LLM context usage

When writing profiling scripts:
- Process raw data and output only essential metrics
- Include error handling and timeouts
- For eBPF: use BCC Python bindings or bpftrace

When tuning:
- Detect which libraries/tools the application uses (check loaded .so files, imports, process name)
- Reference the loaded manual sections for detection heuristics and tuning patterns
- Before recommending configuration changes for a workload, run a profiling pipeline first:
  1) collect_profile for process/system metrics
  2) eBPF profiling when available (cache misses, page faults, or NUMA migrations)
  3) dynamic script refinement (capture/process) to compress raw profiling output
- Base recommendations on measured profiling evidence, not assumptions
- Use the apply_tuning tool with config_type: sysctl | prlimit | command | env_var | numactl
- Follow the manual's decision trees and recommended configurations
- After applying changes, re-profile to verify improvement
- Roll back any change that causes >5% performance regression
- Stop iterating when 2 consecutive iterations show <1% improvement

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
        self._llm_prompt_log_path: Path | None = None
        self._llm_prompt_log_index = 0

        # Robust manual directory discovery
        manual_candidates = [
            Path(__file__).resolve().parents[2] / "manual",
            # Current directory
            Path("manual"),
            # User config: ~/.syspilot/manual
            Path("~/.syspilot/manual").expanduser(),
        ]

        self._manual_dir = manual_candidates[0]  # Default
        for p in manual_candidates:
            if p.exists() and p.is_dir():
                self._manual_dir = p
                logger.debug(f"Using manual directory: {p}")
                break

        # Initialize with system prompt (manuals will be loaded separately)
        self._update_system_prompt()

    def enable_llm_prompt_logging(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._llm_prompt_log_path = path
        self._llm_prompt_log_index = 0
        return path

    @staticmethod
    def _message_to_prompt_dict(message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls is not None:
            payload["tool_calls"] = message.tool_calls
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        return payload

    @staticmethod
    def _tool_to_prompt_dict(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    def _record_llm_prompt_messages(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> None:
        if self._llm_prompt_log_path is None:
            return
        self._llm_prompt_log_index += 1
        payload = {
            "index": self._llm_prompt_log_index,
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "state": self.state.value,
            "messages": [self._message_to_prompt_dict(msg) for msg in messages],
            "tools": [self._tool_to_prompt_dict(tool) for tool in tools or []],
        }
        with self._llm_prompt_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _record_llm_prompt(self) -> None:
        self._record_llm_prompt_messages(self._message_history, self._tool_definitions)

    def _update_system_prompt(self) -> None:
        prompt_parts = [self.SYSTEM_PROMPT]

        if self.context.loaded_manuals:
            prompt_parts.append("\n\n## LOADED MANUALS")
            prompt_parts.append("You have access to the following technical manuals for tuning:")
            for manual_name, manual_content in self.context.loaded_manuals.items():
                prompt_parts.append(f"\n### Manual: {manual_name}")
                prompt_parts.append(manual_content)

        full_prompt = "\n".join(prompt_parts)

        if self._message_history and self._message_history[0].role == "system":
            self._message_history[0] = Message(role="system", content=full_prompt)
        else:
            self._message_history.insert(0, Message(role="system", content=full_prompt))

    def _manual_candidate_paths(self, manual_name: str) -> list[Path]:
        return [
            self._manual_dir / f"{manual_name}.md",
            self._manual_dir / manual_name / "manual.md",
        ]

    def load_manual(self, manual_name: str) -> bool:
        manual_path = next(
            (path for path in self._manual_candidate_paths(manual_name) if path.exists()),
            None,
        )

        if manual_path is None:
            logger.warning(
                "Manual not found for %s (checked %s)",
                manual_name,
                ", ".join(str(path) for path in self._manual_candidate_paths(manual_name)),
            )
            return False

        try:
            with open(manual_path, "r") as f:
                content = f.read()

            self.context.loaded_manuals[manual_name] = content
            self._update_system_prompt()
            logger.info(f"Loaded manual: {manual_name} ({len(content)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to load manual {manual_name}: {e}")
            return False

    def load_manuals(self, manual_names: list[str]) -> dict[str, bool]:
        results = {}
        for name in manual_names:
            results[name] = self.load_manual(name)
        return results

    def list_available_manuals(self) -> list[str]:
        if not self._manual_dir.exists():
            return []
        names = {p.stem for p in self._manual_dir.glob("*.md")}
        for bundle_dir in (path for path in self._manual_dir.iterdir() if path.is_dir()):
            if any((bundle_dir / name).exists() for name in ("manual.md", "manual.json")):
                names.add(bundle_dir.name)
        return sorted(names)

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool that the agent can use."""
        self._tools[name] = func
        self._tool_definitions.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
            )
        )
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

            try:
                arguments = json.loads(tc["function"]["arguments"])
                result = await self._execute_tool(tool_name, arguments)
                result_str = json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                result_str = f"Error: {str(e)}"

            results.append(
                Message(
                    role="tool",
                    content=result_str,
                    tool_call_id=tc["id"],
                )
            )

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
        self._record_llm_prompt()
        response = await self.llm.chat(
            messages=self._message_history,
            tools=self._tool_definitions if self._tool_definitions else None,
        )

        # Add assistant response to history
        self._message_history.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

        # Process tool calls if any
        if response.tool_calls:
            tool_results = await self._process_tool_calls(response.tool_calls)
            self._message_history.extend(tool_results)

            # Continue conversation after tool execution
            return await self.step()

        return response

    async def step_stateless(self, user_input: str, *, include_tools: bool = False) -> LLMResponse:
        """Send one prompt without retaining prior tuning-analysis history."""
        self._update_system_prompt()
        messages: list[Message] = []
        if self._message_history and self._message_history[0].role == "system":
            messages.append(self._message_history[0])
        messages.append(Message(role="user", content=user_input))
        tools = self._tool_definitions if include_tools and self._tool_definitions else None
        self._record_llm_prompt_messages(messages, tools)
        return await self.llm.chat(messages=messages, tools=tools)

    async def run_monitoring_loop(self) -> None:
        self._running = True
        self.state = AgentState.MONITORING
        logger.info("Starting monitoring loop")

        iteration = 0
        while self._running:
            try:
                iteration += 1
                prompt = (
                    f"Monitoring iteration {iteration}.\n"
                    f"Target PID: {self.context.target_pid or 'system-wide'}\n\n"
                    "Steps:\n"
                    "1. Collect profiling data using collect_profile\n"
                    "2. Analyze the results for anomalies or bottlenecks\n"
                    "3. If a tuning manual is loaded, check if any recommended optimizations apply\n"
                    "4. If issues are found, suggest specific tuning actions with config_type and parameters\n"
                    "5. Summarize findings in 3 sentences or fewer"
                )
                await self.step(prompt)
                await asyncio.sleep(self.config.monitoring.interval_seconds)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.state = AgentState.ERROR
                await asyncio.sleep(5)
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
- PID: {self.context.target_pid or "Not specified"}
- Name: {self.context.target_name or "Not specified"}

1. First, gather basic process information
2. Profile CPU and memory usage
3. Identify any obvious performance issues
4. Suggest profiling scripts if deeper analysis is needed

Provide a summary of findings."""

        try:
            response = await self.step(prompt)
            return response.content
        except LLMError as exc:
            logger.warning("Analyze target LLM request failed: %s", exc)
            return f"Analysis unavailable: {exc}"
        finally:
            self.state = AgentState.IDLE

    async def tune_application(
        self, recommendations: list[TuningRecommendation] | None = None
    ) -> str:
        """Apply tuning recommendations."""
        self.state = AgentState.TUNING

        if recommendations:
            rec_text = "\n".join(
                [
                    f"- {r.parameter}: {r.current_value} -> {r.recommended_value} ({r.reason})"
                    for r in recommendations
                ]
            )
            prompt = f"""Apply the following tuning recommendations:
{rec_text}

Dry run mode: {self.config.tuning.dry_run}
Backup configs: {self.config.tuning.backup_configs}

Execute the tuning and report results."""
        else:
            prompt = """Based on the current profiling data, suggest and apply tuning recommendations.
Be conservative and explain each change."""

        try:
            response = await self.step(prompt)
            return response.content
        except LLMError as exc:
            logger.warning("Tune application LLM request failed: %s", exc)
            return f"Tuning analysis unavailable: {exc}"
        finally:
            self.state = AgentState.IDLE

    async def run_tuning_loop(
        self,
        tuning_engine: Any,
        max_iterations: int | None = None,
    ) -> str:
        from syspilot.tuning.engine import TuningEngine

        if not isinstance(tuning_engine, TuningEngine):
            raise TypeError("tuning_engine must be a TuningEngine instance")

        self.state = AgentState.TUNING
        self._running = True
        pid = self.context.target_pid
        if not pid:
            self.state = AgentState.IDLE
            return "No target PID specified for tuning."

        iterations = max_iterations or tuning_engine.max_iterations

        for i in range(1, iterations + 1):
            if not self._running:
                break

            prompt = (
                f"Tuning iteration {i}/{iterations} for PID {pid}.\n\n"
                "Steps:\n"
                "1. Collect current profiling data\n"
                "2. Analyze metrics and loaded manuals\n"
                "3. Recommend tuning actions as a JSON list of objects with keys: "
                "config_type (sysctl|prlimit|command|env_var|numactl), parameter, "
                "current_value, new_value, reason\n"
                "4. If no further optimizations are possible, respond with an empty list []"
            )

            try:
                response = await self.step(prompt)
            except LLMError as exc:
                logger.warning("Tuning iteration %s LLM request failed: %s", i, exc)
                break

            try:
                import re as _re

                json_match = _re.search(r"\[.*\]", response.content, _re.DOTALL)
                if json_match:
                    recommendations = json.loads(json_match.group())
                else:
                    recommendations = []
            except (json.JSONDecodeError, ValueError):
                recommendations = []

            if not recommendations:
                logger.info(f"No more recommendations at iteration {i}, stopping")
                break

            result = await tuning_engine.run_iteration(pid, recommendations, i)

            if result.rolled_back:
                logger.warning(f"Iteration {i} rolled back, stopping tuning loop")
                break

            if tuning_engine.is_converged():
                logger.info("Tuning converged, stopping")
                break

        self.state = AgentState.IDLE
        summary = tuning_engine.get_summary()
        return json.dumps(summary, indent=2)

    def add_profile_result(self, result: ProfileResult) -> None:
        """Add a profiling result to context."""
        self.context.profile_history.append(result)
        logger.debug(f"Added profile result from {result.source}: {result.summary[:100]}...")
