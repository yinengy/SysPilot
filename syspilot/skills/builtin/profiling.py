from __future__ import annotations

import time
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
