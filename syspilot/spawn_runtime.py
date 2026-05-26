from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import psutil

from syspilot.core.agent import SysPilotAgent
from syspilot.toolbox_runtime import extract_declared_virtual_probes


@dataclass
class SharedRunArtifact:
    pid: int
    launch_command: list[str]
    wall_time: float
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    started_at: str
    finished_at: str
    profiling_text: str
    profiling_json: dict[str, Any]
    tool_calls_used: int
    stop_reason: str = "completed"
    stopped_intentionally: bool = False
    stop_after_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ContractValidationRunner = Callable[
    [
        SysPilotAgent,
        list[dict[str, Any]],
        dict[str, Any],
        str,
        str,
        dict[str, str],
        dict[str, str],
    ],
    Awaitable[tuple[dict[str, Any], int, list[str]]],
]
StartProbeSessions = Callable[[list[str], dict[str, Any]], Awaitable[dict[str, Any]]]
FinishProbeSessions = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, dict[str, Any]]]]
SampleProbeSessions = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, dict[str, Any]]]]


def _read_proc_status_map(pid: int) -> dict[str, Any]:
    try:
        raw = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return {}

    result: dict[str, Any] = {}
    try:
        with raw.oneshot():
            status_path = f"/proc/{pid}/status"
            with open(status_path, encoding="utf-8") as handle:
                for line in handle:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    result[key.strip()] = value.strip()
    except (FileNotFoundError, OSError, psutil.NoSuchProcess, psutil.AccessDenied):
        return {}
    return result


def _live_collect_profile_payload(runtime_map: dict[str, Any]) -> dict[str, Any]:
    threads = runtime_map.get("Threads")
    rss_raw = runtime_map.get("VmRSS")
    thread_count: int | None = None
    rss_mb: float | None = None

    if isinstance(threads, str) and threads.isdigit():
        thread_count = int(threads)
    elif isinstance(threads, int):
        thread_count = threads

    if isinstance(rss_raw, str):
        parts = rss_raw.split()
        if parts and parts[0].isdigit():
            rss_mb = int(parts[0]) / 1024.0

    return {
        "cpu": {"data": {"num_threads": thread_count}},
        "memory": {"data": {"rss_mb": rss_mb, "peak_memory_mb": rss_mb}},
    }


def _combined_stream_text(chunks: list[str]) -> str:
    return "".join(chunks)


def _append_virtual_probe_timeseries_samples(
    periodic_samples: list[dict[str, Any]],
    virtual_probe_results: dict[str, Any],
) -> None:
    for probe_name, probe_result in virtual_probe_results.items():
        if not isinstance(probe_result, dict):
            continue
        structured = probe_result.get("structured", {})
        if not isinstance(structured, dict):
            continue
        timeseries = structured.get("timeseries", [])
        if not isinstance(timeseries, list):
            continue
        for point in timeseries:
            if not isinstance(point, dict):
                continue
            raw_time = point.get("time_offset_s")
            raw_value = point.get("value")
            if not isinstance(raw_time, (int, float)) or not isinstance(raw_value, (int, float)):
                continue
            sample_result = {
                **probe_result,
                "value": raw_value,
                "flat": {probe_name: raw_value},
                "structured": {**structured, "timeseries_point": point},
            }
            periodic_samples.append(
                {
                    "time_offset_s": max(float(raw_time), 0.0),
                    "results": {probe_name: sample_result},
                    "source": "finish_probe_timeseries",
                }
            )


async def execute_profiled_spawn_run(
    agent: SysPilotAgent,
    command_parts: list[str],
    timeout: int,
    *,
    workdir: str | None,
    memory_high: str | None,
    env_overrides: dict[str, str] | None,
    hypothesis_profiles: list[dict[str, Any]] | None,
    live_output_callback: Callable[[str, str], None] | None,
    build_spawn_profiling_env: Callable[[dict[str, str] | None, dict[str, str] | None], dict[str, str] | None],
    wrap_spawn_command_with_memory_high: Callable[[list[str], str | None, str | None], list[str]],
    discover_related_processes: Callable[[int, int], dict[str, Any]],
    summarize_spawn_runtime_output: Callable[[str, str], dict[str, object]],
    collect_contract_report_env: Callable[[list[dict[str, Any]]], dict[str, str]],
    run_contract_hypothesis_validation: ContractValidationRunner,
    stringify_env_value: Callable[[Any], str],
    format_bytes_human: Callable[[int | None], str],
    start_probe_sessions: StartProbeSessions,
    finish_probe_sessions: FinishProbeSessions,
    sample_probe_sessions: SampleProbeSessions,
    stop_after_seconds: float | None = None,
    intentional_stop_reason: str = "screen_budget_reached",
    periodic_sample_interval_s: float | None = None,
) -> SharedRunArtifact:
    started_at = time.monotonic()
    started_at_wall = datetime.now(timezone.utc).isoformat()
    selected_hypothesis_profiles = hypothesis_profiles or []
    report_env_overrides = collect_contract_report_env(selected_hypothesis_profiles)
    env = build_spawn_profiling_env(env_overrides, report_env_overrides)
    launch_command = wrap_spawn_command_with_memory_high(command_parts, memory_high, workdir)
    launch_cwd = None if memory_high else workdir

    proc = await asyncio.create_subprocess_exec(
        *launch_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=launch_cwd,
        env=env,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    async def _drain_stream(
        stream_name: str,
        stream: asyncio.StreamReader | None,
        chunks: list[str],
    ) -> str:
        if stream is None:
            return ""
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            if live_output_callback is not None:
                live_output_callback(stream_name, text)
        return "".join(chunks)

    stdout_task = asyncio.create_task(_drain_stream("stdout", proc.stdout, stdout_chunks))
    stderr_task = asyncio.create_task(_drain_stream("stderr", proc.stderr, stderr_chunks))

    pid = proc.pid
    prev_target_pid = getattr(agent.context, "target_pid", None)
    agent.context.target_pid = pid

    await asyncio.sleep(1.0)

    process_targets = discover_related_processes(pid, 4)
    target_pids = process_targets["target_pids"]
    primary_pid = process_targets["primary_pid"]
    requested_virtual_probes = extract_declared_virtual_probes(
        selected_hypothesis_profiles,
        memory_high=memory_high,
    )
    probe_sessions = await start_probe_sessions(
        requested_virtual_probes,
        {
            "target_pids": target_pids,
            "primary_pid": primary_pid,
            "memory_high": memory_high,
        },
    )

    periodic_samples: list[dict[str, Any]] = []
    periodic_sampler_stop = asyncio.Event()

    def _build_live_probe_payload() -> dict[str, Any]:
        per_pid_runtime_maps = {
            str(target_pid): runtime_map
            for target_pid in target_pids
            if (runtime_map := _read_proc_status_map(target_pid))
        }
        primary_runtime_map = per_pid_runtime_maps.get(str(primary_pid), {})
        stdout_live = _combined_stream_text(stdout_chunks)
        stderr_live = _combined_stream_text(stderr_chunks)
        return {
            "target_pids": target_pids,
            "primary_pid": primary_pid,
            "memory_high": memory_high,
            "stdout": stdout_live,
            "stderr": stderr_live,
            "runtime_output": summarize_spawn_runtime_output(stdout_live, stderr_live),
            "refined_runtime_maps": primary_runtime_map,
            "per_pid_runtime_maps": per_pid_runtime_maps,
            "collect_profile_data": _live_collect_profile_payload(primary_runtime_map),
            "per_pid_collect_profile": {
                str(primary_pid): _live_collect_profile_payload(primary_runtime_map)
            }
            if primary_runtime_map
            else {},
            "process_tree_targets": process_targets,
            "command_parts": command_parts,
            "workdir": workdir,
            "launch_env": env,
        }

    async def _sample_periodic_probes() -> None:
        if not periodic_sample_interval_s or not probe_sessions:
            return
        while not periodic_sampler_stop.is_set():
            payload = _build_live_probe_payload()
            sample_results = await sample_probe_sessions(probe_sessions, payload)
            periodic_samples.append(
                {
                    "time_offset_s": max(time.monotonic() - started_at, 0.0),
                    "results": sample_results,
                }
            )
            try:
                await asyncio.wait_for(
                    periodic_sampler_stop.wait(),
                    timeout=periodic_sample_interval_s,
                )
            except asyncio.TimeoutError:
                continue

    periodic_sampler_task = asyncio.create_task(_sample_periodic_probes())

    tools_used: list[str] = []
    key_findings: list[str] = []
    per_pid_runtime_maps: dict[str, dict[str, Any]] = {}
    per_pid_collect_profile: dict[str, dict[str, Any]] = {}
    collect_summaries: list[str] = []
    agent_tool_calls = 0

    for target_pid in target_pids:
        try:
            collect_result = await agent._execute_tool(
                "collect_profile",
                {
                    "collectors": ["cpu", "memory", "process", "numa"],
                    "pid": target_pid,
                },
            )
            agent_tool_calls += 1
            tools_used.append(f"collect_profile:{target_pid}")

            if isinstance(collect_result, dict):
                per_pid_collect_profile[str(target_pid)] = collect_result
                summaries: list[str] = []
                for collector_name, payload in collect_result.items():
                    if isinstance(payload, dict) and "summary" in payload:
                        summaries.append(f"{collector_name}: {payload['summary']}")
                if summaries:
                    collect_summaries.append(f"pid {target_pid}: " + " | ".join(summaries))
        except Exception as exc:
            key_findings.append(f"collect_profile failed for pid {target_pid}: {exc}")

    ebpf_summary = ""
    ebpf_data: dict[str, Any] = {}
    try:
        ebpf_check = await agent._execute_tool("skill_ebpf", {"action": "check_availability"})
        agent_tool_calls += 1
        tools_used.append("skill_ebpf.check_availability")
        if isinstance(ebpf_check, dict):
            ebpf_summary = str(ebpf_check.get("summary", ""))
            raw_data = ebpf_check.get("data", {})
            if isinstance(raw_data, dict):
                ebpf_data = raw_data

        template_name = ""
        if ebpf_data.get("bpftrace_available"):
            template_name = "syscall_count"
        elif ebpf_data.get("bcc_available"):
            template_name = "bcc_syscall_latency"

        if template_name:
            ebpf_profile = await agent._execute_tool(
                "skill_ebpf",
                {
                    "action": "run_template",
                    "template_name": template_name,
                    "pid": primary_pid,
                    "duration": 2,
                },
            )
            agent_tool_calls += 1
            tools_used.append(f"skill_ebpf.run_template:{template_name}:{primary_pid}")
            if isinstance(ebpf_profile, dict):
                profile_summary = str(ebpf_profile.get("summary", "")).strip()
                profile_error = str(ebpf_profile.get("error", "")).strip()
                if profile_summary:
                    ebpf_summary = profile_summary
                if profile_error:
                    ebpf_summary = (
                        f"{ebpf_summary} ({profile_error})" if ebpf_summary else profile_error
                    )

                profile_success = bool(ebpf_profile.get("success", False))
                if (
                    not profile_success
                    and template_name == "syscall_count"
                    and ebpf_data.get("bcc_available")
                ):
                    fallback_profile = await agent._execute_tool(
                        "skill_ebpf",
                        {
                            "action": "run_template",
                            "template_name": "bcc_syscall_latency",
                            "pid": primary_pid,
                            "duration": 2,
                        },
                    )
                    agent_tool_calls += 1
                    tools_used.append(f"skill_ebpf.run_template:bcc_syscall_latency:{primary_pid}")
                    if isinstance(fallback_profile, dict):
                        fallback_summary = str(fallback_profile.get("summary", "")).strip()
                        fallback_error = str(fallback_profile.get("error", "")).strip()
                        fallback_text = fallback_summary
                        if fallback_error:
                            fallback_text = (
                                f"{fallback_text} ({fallback_error})"
                                if fallback_text
                                else fallback_error
                            )
                        if fallback_text:
                            ebpf_summary = f"{ebpf_summary} | fallback: {fallback_text}"
    except Exception as exc:
        key_findings.append(f"ebpf profiling failed: {exc}")

    processor_script = (
        "import json\n"
        "data = {}\n"
        "line_count = 0\n"
        "for line in raw_data.splitlines():\n"
        "    line_count += 1\n"
        "    if ':' not in line:\n"
        "        continue\n"
        "    key, value = line.split(':', 1)\n"
        "    key = key.strip()\n"
        "    value = value.strip()\n"
        "    if key in {'Name', 'State', 'Threads', 'Cpus_allowed_list', 'Mems_allowed_list', 'VmRSS', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'}:\n"
        "        data[key] = value\n"
        "data['line_count'] = line_count\n"
        "print(json.dumps(data))\n"
    )

    for target_pid in target_pids:
        try:
            capture_result = await agent._execute_tool(
                "skill_data_processor",
                {
                    "action": "capture",
                    "command": f"cat /proc/{target_pid}/status",
                },
            )
            agent_tool_calls += 1
            tools_used.append(f"skill_data_processor.capture:{target_pid}")

            capture_id = None
            if isinstance(capture_result, dict):
                capture_data = capture_result.get("data", {})
                if isinstance(capture_data, dict):
                    capture_id = capture_data.get("capture_id")

            if capture_id:
                process_result = await agent._execute_tool(
                    "skill_data_processor",
                    {
                        "action": "process",
                        "capture_id": capture_id,
                        "script": processor_script,
                    },
                )
                agent_tool_calls += 1
                tools_used.append(f"skill_data_processor.process:{target_pid}")

                if isinstance(process_result, dict):
                    output = process_result.get("output", "")
                    try:
                        refined = json.loads(output.strip()) if output else {}
                        if isinstance(refined, dict):
                            per_pid_runtime_maps[str(target_pid)] = refined
                    except json.JSONDecodeError:
                        key_findings.append(
                            f"status parsing returned invalid JSON for pid {target_pid}"
                        )
        except Exception as exc:
            key_findings.append(f"dynamic status pipeline failed for pid {target_pid}: {exc}")

    timed_out = False
    stopped_intentionally = False
    stop_reason = "completed"
    wait_budget = float(timeout)
    if stop_after_seconds is not None:
        wait_budget = min(wait_budget, max(stop_after_seconds, 0.01))

    try:
        await asyncio.wait_for(proc.wait(), timeout=wait_budget)
        rc = proc.returncode if proc.returncode is not None else 0
    except asyncio.TimeoutError:
        if stop_after_seconds is not None and wait_budget <= stop_after_seconds + 1e-9:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()
                try:
                    await proc.wait()
                except ProcessLookupError:
                    pass
            rc = proc.returncode if proc.returncode is not None else -15
            stopped_intentionally = True
            stop_reason = intentional_stop_reason
        else:
            proc.kill()
            try:
                await proc.wait()
            except ProcessLookupError:
                pass
            rc = -1
            timed_out = True
            stop_reason = "timeout"

    stdout = await stdout_task
    stderr = await stderr_task
    periodic_sampler_stop.set()
    try:
        await periodic_sampler_task
    except Exception:
        pass

    if timed_out:
        timeout_message = "Command timed out during profiling pipeline"
        stderr = f"{stderr}\n{timeout_message}".strip() if stderr else timeout_message
    elif stopped_intentionally:
        stop_message = f"Command stopped intentionally: {intentional_stop_reason}"
        stderr = f"{stderr}\n{stop_message}".strip() if stderr else stop_message

    wall = time.monotonic() - started_at
    runtime_output = summarize_spawn_runtime_output(stdout, stderr)
    workers = process_targets.get("workers", [])
    if workers:
        heavy_workers = ", ".join(
            f"pid {worker.get('pid')} cpu={worker.get('cpu_percent')}% rss={worker.get('rss_mb')}MB"
            for worker in workers[:3]
        )
        key_findings.append(f"worker-heavy profiling targets: {heavy_workers}")

    if not key_findings:
        key_findings.append("profiling pipeline executed")

    refined_runtime_maps = per_pid_runtime_maps.get(str(primary_pid), {})
    if not refined_runtime_maps and per_pid_runtime_maps:
        first_key = next(iter(per_pid_runtime_maps))
        refined_runtime_maps = per_pid_runtime_maps[first_key]

    primary_collect_profile = per_pid_collect_profile.get(str(primary_pid), {})
    if not primary_collect_profile and per_pid_collect_profile:
        first_key = next(iter(per_pid_collect_profile))
        primary_collect_profile = per_pid_collect_profile[first_key]

    virtual_probe_payload = {
        "target_pids": target_pids,
        "primary_pid": primary_pid,
        "memory_high": memory_high,
        "stdout": stdout,
        "stderr": stderr,
        "runtime_output": runtime_output,
        "refined_runtime_maps": refined_runtime_maps,
        "per_pid_runtime_maps": per_pid_runtime_maps,
        "collect_profile_data": primary_collect_profile,
        "per_pid_collect_profile": per_pid_collect_profile,
        "process_tree_targets": process_targets,
        "command_parts": command_parts,
        "workdir": workdir,
        "launch_env": env,
    }
    virtual_probe_results = await finish_probe_sessions(probe_sessions, virtual_probe_payload)
    if virtual_probe_results and periodic_sample_interval_s:
        _append_virtual_probe_timeseries_samples(periodic_samples, virtual_probe_results)
        periodic_samples.append(
            {
                "time_offset_s": max(wall, 0.0),
                "results": virtual_probe_results,
                "source": "finish_probe_sessions",
            }
        )
    for probe_result in virtual_probe_results.values():
        flat = probe_result.get("flat", {}) if isinstance(probe_result, dict) else {}
        if isinstance(flat, dict) and flat:
            refined_runtime_maps = {**refined_runtime_maps, **flat}

    cgroup_probe_result = virtual_probe_results.get("cgroup.memory.events.high", {})
    cgroup_memory_evidence = (
        cgroup_probe_result.get("structured", {})
        if isinstance(cgroup_probe_result, dict)
        else {}
    )
    if isinstance(cgroup_memory_evidence, dict) and cgroup_memory_evidence.get("control_group"):
        key_findings.append(
            "cgroup memory envelope: "
            f"status={cgroup_memory_evidence.get('status', 'unknown')} "
            f"peak={format_bytes_human(cgroup_memory_evidence.get('peak_bytes'))} "
            f"headroom={format_bytes_human(cgroup_memory_evidence.get('min_headroom_bytes'))}"
        )

    profiling_json: dict[str, Any] = {
        "pipeline_used": True,
        "tools_used": tools_used,
        "target_pids": target_pids,
        "primary_pid": primary_pid,
        "collect_profile_summary": " | ".join(collect_summaries)[:1800],
        "collect_profile_data": primary_collect_profile,
        "ebpf_summary": ebpf_summary,
        "refined_runtime_maps": refined_runtime_maps,
        "per_pid_runtime_maps": per_pid_runtime_maps,
        "cgroup_memory_evidence": cgroup_memory_evidence,
        "virtual_probe_results": virtual_probe_results,
        "runtime_output": runtime_output,
        "process_tree_targets": process_targets,
        "key_findings": key_findings,
        "periodic_probe_samples": periodic_samples,
        "requested_virtual_probes": requested_virtual_probes,
        "stop_reason": stop_reason,
        "stopped_intentionally": stopped_intentionally,
        "stop_after_seconds": stop_after_seconds,
    }

    if selected_hypothesis_profiles:
        validation_result, validation_tool_calls, validation_tools = await run_contract_hypothesis_validation(
            agent,
            selected_hypothesis_profiles,
            profiling_json,
            stdout,
            stderr,
            report_env_overrides,
            {
                **(
                    {key: stringify_env_value(value) for key, value in env_overrides.items()}
                    if env_overrides
                    else {}
                ),
                **report_env_overrides,
            },
        )
        agent_tool_calls += validation_tool_calls
        tools_used.extend(validation_tools)
        profiling_json["tools_used"] = tools_used
        profiling_json["contract_hypothesis_validation"] = validation_result
        if validation_result.get("profiles"):
            key_findings.append(
                f"contract-guided validation evaluated {len(validation_result.get('profiles', []))} hypothesis profile(s)"
            )
        elif validation_result.get("error"):
            key_findings.append(
                f"contract-guided validation unavailable: {validation_result.get('error')}"
            )
        profiling_json["key_findings"] = key_findings
    profiling_text = json.dumps(profiling_json)
    finished_at_wall = datetime.now(timezone.utc).isoformat()

    agent.context.target_pid = prev_target_pid
    return SharedRunArtifact(
        pid=pid,
        launch_command=launch_command,
        wall_time=wall,
        stdout=stdout,
        stderr=stderr,
        returncode=rc,
        timed_out=timed_out,
        started_at=started_at_wall,
        finished_at=finished_at_wall,
        profiling_text=profiling_text,
        profiling_json=profiling_json,
        tool_calls_used=agent_tool_calls,
        stop_reason=stop_reason,
        stopped_intentionally=stopped_intentionally,
        stop_after_seconds=stop_after_seconds,
    )
