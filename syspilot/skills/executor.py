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


BLOCKED_COMMANDS: set[str] = {
    "rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "poweroff",
    "halt", "init", "systemctl", "format", "fdisk", "parted",
    ":(){:|:&};:", "fork", "wget", "curl",
}

SUDO_ALLOWED: set[str] = {"bpftrace", "perf", "sysctl", "prlimit", "numactl"}


class ScriptExecutor:
    """Executes dynamically generated scripts safely."""
    
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()
    
    def _validate_command(self, cmd: list[str]) -> None:
        """Raise ValueError if the command violates sandbox policy.
        
        Checks:
        1. The base binary must not be in the BLOCKED_COMMANDS set.
        2. If sandbox_enabled, the base binary must be in allowed_commands
           (when an allow-list is configured).
        3. sudo is only permitted for a known subset of tools.
        """
        if not cmd:
            raise ValueError("Empty command")

        base_idx = 0
        if cmd[0] == "sudo":
            base_idx = 1
            while base_idx < len(cmd) and cmd[base_idx].startswith("-"):
                base_idx += 1
            if base_idx >= len(cmd):
                raise ValueError("sudo command missing target binary")
        base_cmd = Path(cmd[base_idx]).name  # handle absolute paths

        # Block dangerous commands unconditionally
        if base_cmd in BLOCKED_COMMANDS:
            raise ValueError(
                f"Command '{base_cmd}' is blocked by sandbox policy"
            )

        # Enforce allow-list when sandbox is on and list is provided
        if self.config.sandbox_enabled and self.config.allowed_commands is not None:
            if base_cmd not in self.config.allowed_commands:
                raise ValueError(
                    f"Command '{base_cmd}' is not in the allowed commands list: "
                    f"{self.config.allowed_commands}"
                )

        # Restrict sudo usage
        if cmd[0] == "sudo" and base_cmd not in SUDO_ALLOWED:
            raise ValueError(
                f"sudo is not permitted for '{base_cmd}'. "
                f"Allowed sudo commands: {SUDO_ALLOWED}"
            )
    
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
            cmd = ["sudo", "-n", "bpftrace", script_path]
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
        
        try:
            self._validate_command(cmd)
        except ValueError as e:
            return SkillResult(
                success=False,
                output="",
                summary=f"Sandbox violation: {e}",
                execution_time_seconds=0,
                error=str(e),
            )
        
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
