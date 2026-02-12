from __future__ import annotations

import os
import tempfile
import time
import uuid
from typing import Any

from syspilot.skills.base import BaseSkill, SkillResult
from syspilot.skills.executor import ScriptExecutor


class DynamicScriptSkill(BaseSkill):
    """Skill for executing dynamically generated scripts."""
    
    name = "dynamic_script"
    description = "Execute dynamically generated Python or Bash scripts for custom profiling"
    
    def __init__(self) -> None:
        self.executor = ScriptExecutor()
    
    async def execute(self, **kwargs: Any) -> SkillResult:
        script_type = kwargs.get("script_type", "python")
        script = kwargs.get("script", "")
        args = kwargs.get("args", [])
        
        if not script:
            return SkillResult(
                success=False,
                output="",
                summary="No script provided",
                execution_time_seconds=0,
                error="Empty script",
            )
        
        if script_type == "python":
            return await self.executor.execute_python(script, args)
        elif script_type == "bash":
            return await self.executor.execute_bash(script, args)
        else:
            return SkillResult(
                success=False,
                output="",
                summary=f"Unknown script type: {script_type}",
                execution_time_seconds=0,
                error=f"Unsupported: {script_type}",
            )
    
    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "script_type": {
                    "type": "string",
                    "enum": ["python", "bash"],
                    "description": "Type of script to execute",
                },
                "script": {
                    "type": "string",
                    "description": "The script content to execute",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command line arguments",
                },
            },
            "required": ["script_type", "script"],
        }


class StraceAnalyzerSkill(BaseSkill):
    """Skill for running and analyzing strace output."""
    
    name = "strace_analyzer"
    description = "Run strace on a process and summarize syscall patterns"
    
    def __init__(self) -> None:
        self.executor = ScriptExecutor()
    
    async def execute(self, **kwargs: Any) -> SkillResult:
        pid = kwargs.get("pid")
        duration = kwargs.get("duration", 5)
        
        if not pid:
            return SkillResult(
                success=False,
                output="",
                summary="PID required",
                execution_time_seconds=0,
                error="Missing PID",
            )
        
        script = f'''#!/bin/bash
set -e
timeout {duration} strace -c -p {pid} 2>&1 || true
'''
        return await self.executor.execute_bash(script)
    
    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pid": {"type": "integer", "description": "Process ID to trace"},
                "duration": {"type": "integer", "description": "Duration in seconds", "default": 5},
            },
            "required": ["pid"],
        }


class PerfAnalyzerSkill(BaseSkill):
    """Skill for running perf and analyzing performance data."""
    
    name = "perf_analyzer"
    description = "Run perf on a process and summarize CPU hotspots"
    
    def __init__(self) -> None:
        self.executor = ScriptExecutor()
    
    async def execute(self, **kwargs: Any) -> SkillResult:
        pid = kwargs.get("pid")
        duration = kwargs.get("duration", 5)
        event = kwargs.get("event", "cycles")
        
        if not pid:
            return SkillResult(
                success=False,
                output="",
                summary="PID required",
                execution_time_seconds=0,
                error="Missing PID",
            )
        
        script = f'''#!/bin/bash
set -e
sudo perf record -e {event} -p {pid} -g -- sleep {duration} 2>/dev/null
sudo perf report --stdio --no-children 2>/dev/null | head -50
rm -f perf.data perf.data.old 2>/dev/null || true
'''
        return await self.executor.execute_bash(script)
    
    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pid": {"type": "integer", "description": "Process ID to profile"},
                "duration": {"type": "integer", "description": "Duration in seconds", "default": 5},
                "event": {"type": "string", "description": "Perf event to record", "default": "cycles"},
            },
            "required": ["pid"],
        }


class DataProcessorSkill(BaseSkill):
    name = "data_processor"
    description = (
        "Two-phase data processing: 'capture' runs a command and stores raw output, "
        "'process' runs a script against the captured data to produce compressed insights"
    )

    PREVIEW_LINES = 15

    def __init__(self) -> None:
        self.executor = ScriptExecutor()
        self._captures: dict[str, str] = {}

    async def execute(self, **kwargs: Any) -> SkillResult:
        action = kwargs.get("action", "capture")

        if action == "capture":
            return await self._capture(kwargs.get("command", ""))
        elif action == "process":
            return await self._process(
                capture_id=kwargs.get("capture_id", ""),
                script=kwargs.get("script", ""),
            )
        else:
            return SkillResult(
                success=False,
                output="",
                summary=f"Unknown action: {action}",
                execution_time_seconds=0,
                error=f"Unknown action: {action}",
            )

    async def _capture(self, command: str) -> SkillResult:
        if not command:
            return SkillResult(
                success=False, output="", summary="No command provided",
                execution_time_seconds=0, error="Empty command",
            )

        result = await self.executor.execute_bash(command)
        if not result.success:
            return result

        capture_id = uuid.uuid4().hex[:12]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".capture", delete=False, prefix=f"syspilot_{capture_id}_",
        )
        tmp.write(result.output)
        tmp.close()
        self._captures[capture_id] = tmp.name

        lines = result.output.splitlines()
        preview = "\n".join(lines[:self.PREVIEW_LINES])

        return SkillResult(
            success=True,
            output=preview,
            summary=f"Captured {len(lines)} lines ({len(result.output)} bytes) as {capture_id}",
            execution_time_seconds=result.execution_time_seconds,
            data={"capture_id": capture_id, "total_lines": len(lines), "total_bytes": len(result.output)},
        )

    async def _process(self, capture_id: str, script: str) -> SkillResult:
        if not capture_id or capture_id not in self._captures:
            return SkillResult(
                success=False, output="", summary=f"Unknown capture_id: {capture_id}",
                execution_time_seconds=0, error="Invalid capture_id",
            )
        if not script:
            return SkillResult(
                success=False, output="", summary="No processing script provided",
                execution_time_seconds=0, error="Empty script",
            )

        filepath = self._captures[capture_id]
        injected = f'raw_data = open("{filepath}").read()\n' + script

        result = await self.executor.execute_python(injected)

        try:
            os.unlink(filepath)
        except OSError:
            pass
        self._captures.pop(capture_id, None)

        return result

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["capture", "process"],
                    "description": (
                        "'capture': run a shell command and store raw output. "
                        "'process': run a Python script against previously captured data."
                    ),
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to capture (used with action=capture)",
                },
                "capture_id": {
                    "type": "string",
                    "description": "ID returned from a previous capture (used with action=process)",
                },
                "script": {
                    "type": "string",
                    "description": (
                        "Python script to process captured data. The variable 'raw_data' "
                        "is pre-injected with the captured output string."
                    ),
                },
            },
            "required": ["action"],
        }
