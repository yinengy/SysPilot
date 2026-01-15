from __future__ import annotations

import shutil
import time
from typing import Any

from syspilot.skills.base import BaseSkill, SkillResult
from syspilot.skills.executor import ScriptExecutor
from syspilot.skills.ebpf.templates import EBPFTemplates


class EBPFSkill(BaseSkill):
    name = "ebpf"
    description = "Execute eBPF programs for low-overhead system profiling"
    
    def __init__(self) -> None:
        self.executor = ScriptExecutor()
        self._bpftrace_available: bool | None = None
        self._bcc_available: bool | None = None
    
    def _check_bpftrace(self) -> bool:
        if self._bpftrace_available is None:
            self._bpftrace_available = shutil.which("bpftrace") is not None
        return self._bpftrace_available
    
    def _check_bcc(self) -> bool:
        if self._bcc_available is None:
            try:
                import bcc
                self._bcc_available = True
            except ImportError:
                self._bcc_available = False
        return self._bcc_available
    
    async def execute(self, **kwargs: Any) -> SkillResult:
        action = kwargs.get("action", "run_template")
        
        if action == "list_templates":
            return SkillResult(
                success=True,
                output=str(EBPFTemplates.list_templates()),
                summary="Listed available eBPF templates",
                execution_time_seconds=0,
                data={"templates": EBPFTemplates.list_templates()},
            )
        
        if action == "check_availability":
            return self._check_availability()
        
        if action == "run_template":
            return await self._run_template(
                template_name=kwargs.get("template_name", "syscall_count"),
                pid=kwargs.get("pid"),
                duration=kwargs.get("duration", 5),
            )
        
        if action == "run_custom":
            return await self._run_custom(
                script=kwargs.get("script", ""),
                script_type=kwargs.get("script_type", "bpftrace"),
                duration=kwargs.get("duration", 5),
            )
        
        return SkillResult(
            success=False,
            output="",
            summary=f"Unknown action: {action}",
            execution_time_seconds=0,
            error=f"Unknown action: {action}",
        )
    
    def _check_availability(self) -> SkillResult:
        bpftrace = self._check_bpftrace()
        bcc = self._check_bcc()
        
        data = {
            "bpftrace_available": bpftrace,
            "bcc_available": bcc,
            "any_available": bpftrace or bcc,
        }
        
        if not data["any_available"]:
            return SkillResult(
                success=False,
                output="",
                summary="No eBPF tools available. Install bpftrace or bcc.",
                execution_time_seconds=0,
                error="eBPF tools not found",
                data=data,
            )
        
        return SkillResult(
            success=True,
            output=f"bpftrace: {bpftrace}, bcc: {bcc}",
            summary=f"eBPF available (bpftrace={bpftrace}, bcc={bcc})",
            execution_time_seconds=0,
            data=data,
        )
    
    async def _run_template(
        self,
        template_name: str,
        pid: int | None,
        duration: int,
    ) -> SkillResult:
        if not pid:
            return SkillResult(
                success=False,
                output="",
                summary="PID required for eBPF profiling",
                execution_time_seconds=0,
                error="Missing PID",
            )
        
        script = EBPFTemplates.render(template_name, pid, duration)
        if not script:
            return SkillResult(
                success=False,
                output="",
                summary=f"Unknown template: {template_name}",
                execution_time_seconds=0,
                error=f"Template not found: {template_name}",
            )
        
        if template_name.startswith("bcc_"):
            return await self.executor.execute_python(script)
        else:
            return await self.executor.execute_bpftrace(script, duration)
    
    async def _run_custom(
        self,
        script: str,
        script_type: str,
        duration: int,
    ) -> SkillResult:
        if not script:
            return SkillResult(
                success=False,
                output="",
                summary="No script provided",
                execution_time_seconds=0,
                error="Empty script",
            )
        
        if script_type == "bpftrace":
            if not self._check_bpftrace():
                return SkillResult(
                    success=False,
                    output="",
                    summary="bpftrace not available",
                    execution_time_seconds=0,
                    error="bpftrace not installed",
                )
            return await self.executor.execute_bpftrace(script, duration)
        
        elif script_type == "bcc":
            if not self._check_bcc():
                return SkillResult(
                    success=False,
                    output="",
                    summary="BCC not available",
                    execution_time_seconds=0,
                    error="BCC not installed",
                )
            return await self.executor.execute_python(script)
        
        return SkillResult(
            success=False,
            output="",
            summary=f"Unknown script type: {script_type}",
            execution_time_seconds=0,
            error=f"Unsupported script type: {script_type}",
        )
    
    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run_template", "run_custom", "list_templates", "check_availability"],
                    "description": "Action to perform",
                },
                "template_name": {
                    "type": "string",
                    "description": "Name of the eBPF template to run",
                },
                "pid": {
                    "type": "integer",
                    "description": "Target process ID",
                },
                "duration": {
                    "type": "integer",
                    "description": "Profiling duration in seconds",
                    "default": 5,
                },
                "script": {
                    "type": "string",
                    "description": "Custom eBPF script content",
                },
                "script_type": {
                    "type": "string",
                    "enum": ["bpftrace", "bcc"],
                    "description": "Type of eBPF script",
                },
            },
            "required": ["action"],
        }
