from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syspilot.skills.base import SkillResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    timeout_seconds: float = 60.0
    sandbox_enabled: bool = True
    allowed_commands: list[str] | None = None
    working_dir: Path | None = None
    env_vars: dict[str, str] | None = None


class ScriptExecutor:
    """Executes dynamically generated scripts safely."""
    
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()
    
    async def execute_python(
        self,
        script: str,
        args: list[str] | None = None,
    ) -> SkillResult:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(script)
            script_path = f.name
        
        try:
            cmd = ["python3", script_path] + (args or [])
            return await self._run_command(cmd, script)
        finally:
            os.unlink(script_path)
    
    async def execute_bash(
        self,
        script: str,
        args: list[str] | None = None,
    ) -> SkillResult:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
        ) as f:
            f.write(script)
            script_path = f.name
        
        try:
            os.chmod(script_path, 0o755)
            cmd = ["bash", script_path] + (args or [])
            return await self._run_command(cmd, script)
        finally:
            os.unlink(script_path)
    
    async def execute_bpftrace(
        self,
        script: str,
        duration_seconds: float = 5.0,
    ) -> SkillResult:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".bt",
            delete=False,
        ) as f:
            f.write(script)
            script_path = f.name
        
        try:
            cmd = ["sudo", "bpftrace", script_path]
            return await self._run_command(
                cmd,
                script,
                timeout_override=duration_seconds + 5,
            )
        finally:
            os.unlink(script_path)
    
    async def _run_command(
        self,
        cmd: list[str],
        original_script: str,
        timeout_override: float | None = None,
    ) -> SkillResult:
        start_time = time.time()
        timeout = timeout_override or self.config.timeout_seconds
        
        env = os.environ.copy()
        if self.config.env_vars:
            env.update(self.config.env_vars)
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.working_dir,
                env=env,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            
            execution_time = time.time() - start_time
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            
            if proc.returncode != 0:
                return SkillResult(
                    success=False,
                    output=output,
                    summary=f"Script failed with exit code {proc.returncode}",
                    execution_time_seconds=execution_time,
                    error=error_output,
                )
            
            summary = self._generate_summary(output)
            
            return SkillResult(
                success=True,
                output=output,
                summary=summary,
                execution_time_seconds=execution_time,
            )
            
        except asyncio.TimeoutError:
            return SkillResult(
                success=False,
                output="",
                summary=f"Script timed out after {timeout}s",
                execution_time_seconds=timeout,
                error="Execution timeout",
            )
        except Exception as e:
            return SkillResult(
                success=False,
                output="",
                summary=f"Execution error: {str(e)}",
                execution_time_seconds=time.time() - start_time,
                error=str(e),
            )
    
    def _generate_summary(self, output: str, max_lines: int = 20) -> str:
        lines = output.strip().split("\n")
        if len(lines) <= max_lines:
            return output.strip()
        
        head = lines[:max_lines // 2]
        tail = lines[-(max_lines // 2):]
        return "\n".join(head + [f"... ({len(lines) - max_lines} lines omitted) ..."] + tail)
