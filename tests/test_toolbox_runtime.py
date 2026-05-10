from __future__ import annotations

import asyncio

from syspilot.toolbox_runtime import (
    extract_declared_virtual_probes,
    finish_toolbox_probe_sessions,
    resolve_toolbox_probe,
    start_toolbox_probe_sessions,
)


def test_resolve_toolbox_probe_reports_repo_scripts() -> None:
    available = resolve_toolbox_probe("process.cpu.thread_count")
    missing = resolve_toolbox_probe("missing.probe.example")

    assert available["availability"] == "available"
    assert str(available.get("script_path", "")).endswith("toolbox/probes/process/cpu/thread_count.py")
    assert missing["availability"] == "unavailable"


def test_extract_declared_virtual_probes_adds_memoryhigh_probe() -> None:
    probes = extract_declared_virtual_probes(
        [
            {
                "validation_signals": [
                    {"virtual_probe": "process.cpu.thread_count"},
                    {"signal": {"virtual_probe": "benchmark.context_switch_rate"}},
                ]
            }
        ],
        memory_high="1G",
    )

    assert probes == [
        "process.cpu.thread_count",
        "benchmark.context_switch_rate",
        "cgroup.memory.events.high",
    ]


def test_toolbox_probe_sessions_collect_runtime_output_matches() -> None:
    async def _run() -> None:
        sessions = await start_toolbox_probe_sessions(
            ["runtime.config.startup_properties"],
            {"primary_pid": 1234, "target_pids": [1234]},
        )
        results = await finish_toolbox_probe_sessions(
            sessions,
            {
                "primary_pid": 1234,
                "target_pids": [1234],
                "stdout": "effective env: BENCHMARK_NUM_STREAM_THREADS=2\n",
                "stderr": "",
                "runtime_output": {},
                "refined_runtime_maps": {},
                "per_pid_runtime_maps": {},
                "collect_profile_data": {},
                "per_pid_collect_profile": {},
                "launch_env": {},
            },
        )

        probe_result = results["runtime.config.startup_properties"]
        assert probe_result["observed"] is True
        assert probe_result["source"] == "runtime_output"
        assert probe_result["excerpt"] == ["effective env: BENCHMARK_NUM_STREAM_THREADS=2"]

    asyncio.run(_run())
