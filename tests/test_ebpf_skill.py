from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from syspilot.skills.base import SkillResult
from syspilot.skills.ebpf.skill import EBPFSkill


def _ok_result(summary: str = "ok") -> SkillResult:
    return SkillResult(
        success=True,
        output="output",
        summary=summary,
        execution_time_seconds=0.01,
    )


@pytest.mark.asyncio
async def test_run_template_requires_pid() -> None:
    skill = EBPFSkill()

    result = await skill.execute(action="run_template", template_name="syscall_count")

    assert not result.success
    assert result.error == "Missing PID"


@pytest.mark.asyncio
async def test_run_template_uses_bpftrace_when_available() -> None:
    skill = EBPFSkill()
    skill._check_bpftrace = lambda: True
    skill._check_bcc = lambda: False
    run_mock = AsyncMock(return_value=_ok_result("ran bpftrace"))
    skill.executor.execute_bpftrace = run_mock

    result = await skill.execute(
        action="run_template",
        template_name="syscall_count",
        pid=123,
        duration=3,
    )

    assert result.success
    run_mock.assert_awaited_once()
    awaited_call = run_mock.await_args
    assert awaited_call is not None
    rendered_script = awaited_call.args[0]
    assert "tracepoint:raw_syscalls:sys_enter" in rendered_script
    assert awaited_call.args[1] == 3


@pytest.mark.asyncio
async def test_run_template_uses_bcc_when_bcc_template_requested() -> None:
    skill = EBPFSkill()
    skill._check_bpftrace = lambda: False
    skill._check_bcc = lambda: True
    run_mock = AsyncMock(return_value=_ok_result("ran bcc fallback"))
    skill.executor.execute_python = run_mock

    result = await skill.execute(
        action="run_template",
        template_name="bcc_syscall_latency",
        pid=55,
        duration=2,
    )

    assert result.success
    run_mock.assert_awaited_once()
    awaited_call = run_mock.await_args
    assert awaited_call is not None
    fallback_script = awaited_call.args[0]
    assert "from bcc import BPF" in fallback_script
    assert "pid != 55" in fallback_script


@pytest.mark.asyncio
async def test_run_template_requires_bpftrace_for_non_bcc_templates() -> None:
    skill = EBPFSkill()
    skill._check_bpftrace = lambda: False
    skill._check_bcc = lambda: True

    result = await skill.execute(
        action="run_template",
        template_name="cache_misses",
        pid=77,
        duration=2,
    )

    assert not result.success
    assert result.error == "bpftrace not installed"


def test_check_availability_reports_missing_tools() -> None:
    skill = EBPFSkill()
    skill._check_bpftrace = lambda: False
    skill._check_bcc = lambda: False

    result = skill._check_availability()

    assert not result.success
    assert result.error == "eBPF tools not found"
    assert result.data["any_available"] is False


@pytest.mark.asyncio
async def test_run_custom_rejects_dangerous_script() -> None:
    skill = EBPFSkill()

    result = await skill.execute(
        action="run_custom",
        script_type="bpftrace",
        script="tracepoint:syscalls:sys_enter_openat { @x = count(); }; rm -rf /",
        duration=1,
    )

    assert not result.success
    assert result.error is not None
    assert "dangerous pattern" in result.error


@pytest.mark.asyncio
async def test_run_custom_rejects_probe_less_bpftrace_script() -> None:
    skill = EBPFSkill()

    result = await skill.execute(
        action="run_custom",
        script_type="bpftrace",
        script="BEGIN { printf(\"hello\"); }",
        duration=1,
    )

    assert not result.success
    assert result.error == "bpftrace script must contain at least one probe definition"
