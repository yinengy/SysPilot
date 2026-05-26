from __future__ import annotations

import asyncio
from unittest.mock import patch

from syspilot.toolbox_runtime import (
    extract_declared_virtual_probes,
    finish_toolbox_probe_sessions,
    resolve_toolbox_probe,
    sample_toolbox_probe_sessions,
    start_toolbox_probe_sessions,
)
from toolbox.probes.benchmark.completion_count import collect as collect_completion_count_probe
from toolbox.probes.benchmark.log.message_passing_report import collect as collect_message_passing_report_probe
from toolbox.probes.benchmark.log.section_presence import collect as collect_section_presence_probe
from toolbox.probes.benchmark.log.timing_report import collect as collect_timing_report_probe
from toolbox.probes.cp2k.log.message_passing_summary import collect as collect_cp2k_message_passing_summary_probe
from toolbox.probes.cp2k.log.rank_max_self_time import collect as collect_cp2k_rank_max_self_time_probe
from toolbox.probes.cp2k.log.timing_summary import collect as collect_cp2k_timing_summary_probe
from toolbox.probes.benchmark.throughput import collect as collect_throughput_probe
from toolbox.probes.process.cpu.thread_count import collect as collect_process_thread_count_probe
from toolbox.probes.process.io.syscall_rate import collect as collect_process_io_syscall_rate_probe
from toolbox.probes.runtime.api.openblas_get_config import collect as collect_openblas_get_config_probe
from toolbox.probes.runtime.api.openblas_get_corename import collect as collect_openblas_get_corename_probe
from toolbox.probes.runtime.api.openblas_get_num_procs import collect as collect_openblas_get_num_procs_probe
from toolbox.probes.runtime.api.openblas_get_num_threads import collect as collect_openblas_get_num_threads_probe
from toolbox.probes.runtime.api.openblas_get_parallel import collect as collect_openblas_get_parallel_probe
from toolbox.probes.runtime.launch.mpi_process_count import collect as collect_mpi_process_count_probe
from toolbox.probes.threadpoolctl.threadpool_info import collect as collect_threadpool_info_probe


def test_resolve_toolbox_probe_reports_repo_scripts() -> None:
    available = resolve_toolbox_probe("process.cpu.thread_count")
    alias = resolve_toolbox_probe("runtime.python.threadpoolctl.info")
    native_alias = resolve_toolbox_probe("runtime.native.threadpool_info")
    native_threadpools_alias = resolve_toolbox_probe("runtime.native_threadpools.info")
    threadpool = resolve_toolbox_probe("threadpoolctl.threadpool_info")
    mpi_process_count = resolve_toolbox_probe("runtime.launch.mpi_process_count")
    section_presence = resolve_toolbox_probe("benchmark.log.section_presence")
    timing_report = resolve_toolbox_probe("benchmark.log.timing_report")
    message_passing_report = resolve_toolbox_probe("benchmark.log.message_passing_report")
    cp2k_timing_summary = resolve_toolbox_probe("cp2k.log.timing_summary")
    cp2k_rank_max_self_time = resolve_toolbox_probe("cp2k.log.rank_max_self_time")
    cp2k_message_passing_summary = resolve_toolbox_probe("cp2k.log.message_passing_summary")
    openblas_get_parallel = resolve_toolbox_probe("runtime.api.openblas_get_parallel")
    openblas_get_num_threads = resolve_toolbox_probe("runtime.api.openblas_get_num_threads")
    openblas_get_num_procs = resolve_toolbox_probe("runtime.api.openblas_get_num_procs")
    openblas_get_config = resolve_toolbox_probe("runtime.api.openblas_get_config")
    openblas_get_corename = resolve_toolbox_probe("runtime.api.openblas_get_corename")
    missing = resolve_toolbox_probe("missing.probe.example")

    assert available["availability"] == "available"
    assert str(available.get("script_path", "")).endswith("toolbox/probes/process/cpu/thread_count.py")
    assert alias["availability"] == "available"
    assert str(alias.get("script_path", "")).endswith(
        "toolbox/probes/runtime/python/threadpoolctl/info.py"
    )
    assert native_alias["availability"] == "available"
    assert str(native_alias.get("script_path", "")).endswith(
        "toolbox/probes/runtime/native/threadpool_info.py"
    )
    assert native_threadpools_alias["availability"] == "available"
    assert str(native_threadpools_alias.get("script_path", "")).endswith(
        "toolbox/probes/runtime/native_threadpools/info.py"
    )
    assert threadpool["availability"] == "available"
    assert str(threadpool.get("script_path", "")).endswith(
        "toolbox/probes/threadpoolctl/threadpool_info.py"
    )
    assert mpi_process_count["availability"] == "available"
    assert str(mpi_process_count.get("script_path", "")).endswith(
        "toolbox/probes/runtime/launch/mpi_process_count.py"
    )
    assert section_presence["availability"] == "available"
    assert str(section_presence.get("script_path", "")).endswith(
        "toolbox/probes/benchmark/log/section_presence.py"
    )
    assert timing_report["availability"] == "available"
    assert str(timing_report.get("script_path", "")).endswith(
        "toolbox/probes/benchmark/log/timing_report.py"
    )
    assert message_passing_report["availability"] == "available"
    assert str(message_passing_report.get("script_path", "")).endswith(
        "toolbox/probes/benchmark/log/message_passing_report.py"
    )
    assert cp2k_timing_summary["availability"] == "available"
    assert str(cp2k_timing_summary.get("script_path", "")).endswith(
        "toolbox/probes/cp2k/log/timing_summary.py"
    )
    assert cp2k_rank_max_self_time["availability"] == "available"
    assert str(cp2k_rank_max_self_time.get("script_path", "")).endswith(
        "toolbox/probes/cp2k/log/rank_max_self_time.py"
    )
    assert cp2k_message_passing_summary["availability"] == "available"
    assert str(cp2k_message_passing_summary.get("script_path", "")).endswith(
        "toolbox/probes/cp2k/log/message_passing_summary.py"
    )
    assert openblas_get_parallel["availability"] == "available"
    assert str(openblas_get_parallel.get("script_path", "")).endswith(
        "toolbox/probes/runtime/api/openblas_get_parallel.py"
    )
    assert openblas_get_num_threads["availability"] == "available"
    assert str(openblas_get_num_threads.get("script_path", "")).endswith(
        "toolbox/probes/runtime/api/openblas_get_num_threads.py"
    )
    assert openblas_get_num_procs["availability"] == "available"
    assert str(openblas_get_num_procs.get("script_path", "")).endswith(
        "toolbox/probes/runtime/api/openblas_get_num_procs.py"
    )
    assert openblas_get_config["availability"] == "available"
    assert str(openblas_get_config.get("script_path", "")).endswith(
        "toolbox/probes/runtime/api/openblas_get_config.py"
    )
    assert openblas_get_corename["availability"] == "available"
    assert str(openblas_get_corename.get("script_path", "")).endswith(
        "toolbox/probes/runtime/api/openblas_get_corename.py"
    )
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


def test_toolbox_probe_sessions_support_periodic_sampling_via_collect_fallback() -> None:
    async def _run() -> None:
        sessions = await start_toolbox_probe_sessions(
            ["process.cpu.thread_count"],
            {"primary_pid": 1234, "target_pids": [1234]},
        )
        results = await sample_toolbox_probe_sessions(
            sessions,
            {
                "primary_pid": 1234,
                "target_pids": [1234],
                "runtime_output": {},
                "per_pid_runtime_maps": {"1234": {"Threads": "7"}},
                "collect_profile_data": {},
            },
        )

        probe_result = results["process.cpu.thread_count"]
        assert probe_result["observed"] is True
        assert probe_result["value"] == 7
        assert probe_result["source"] == "proc_status.Threads"

    asyncio.run(_run())


def test_toolbox_probe_sessions_support_periodic_sampling_for_cgroup_memory_high() -> None:
    async def _run() -> None:
        snapshot = {
            "control_group": "/user.slice/demo.scope",
            "cgroup_root": "/sys/fs/cgroup/user.slice/demo.scope",
            "exists": True,
            "memory_high_bytes": 1024,
            "memory_max_bytes": 4096,
            "memory_current_bytes": 900,
            "memory_peak_bytes": 950,
            "memory_swap_current_bytes": 0,
            "memory_events": {"high": 7},
            "memory_swap_events": {},
            "memory_pressure": {},
            "memory_stat": {},
        }

        async def _fake_sample_cgroup_memory_evidence(
            control_group: str,
            stop_event: asyncio.Event,
            *,
            sample_interval_s: float = 0.25,
        ) -> dict[str, object]:
            _ = control_group
            _ = stop_event
            _ = sample_interval_s
            return {
                "latest_snapshot": snapshot,
                "sampled_peak_bytes": 950,
                "min_headroom_bytes": 124,
                "sample_count": 3,
            }

        with (
            patch(
                "toolbox.shared.cgroup_v2.resolve_cgroup_v2_context",
                return_value={
                    "selected_path": "/user.slice/demo.scope",
                    "selected_pid": 1234,
                    "selected_target_pids": [1234],
                    "per_pid_paths": {"1234": "/user.slice/demo.scope"},
                },
            ),
            patch(
                "toolbox.shared.cgroup_v2.sample_cgroup_memory_evidence",
                new=_fake_sample_cgroup_memory_evidence,
            ),
            patch("toolbox.shared.cgroup_v2.read_cgroup_memory_snapshot", return_value=snapshot),
        ):
            payload = {"primary_pid": 1234, "target_pids": [1234], "memory_high": "1G"}
            sessions = await start_toolbox_probe_sessions(["cgroup.memory.events.high"], payload)
            results = await sample_toolbox_probe_sessions(sessions, payload)

            probe_result = results["cgroup.memory.events.high"]
            assert probe_result["observed"] is True
            assert probe_result["value"] == 7
            assert probe_result["source"] == "live_cgroup_v2"
            assert probe_result["flat"]["memory.events.high"] == 7
            assert probe_result["flat"]["cgroup.memory.events.high"] == 7

            await finish_toolbox_probe_sessions(sessions, payload)

    asyncio.run(_run())


def test_benchmark_throughput_probe_prefers_throughput_value_over_first_number() -> None:
    result = collect_throughput_probe(
        {
            "stdout": "threads=8 throughput=123.4 ops/s\n",
            "stderr": "",
        }
    )

    assert result["observed"] is True
    assert result["value"] == 123.4
    assert result["excerpt"] == ["threads=8 throughput=123.4 ops/s"]


def test_completion_count_probe_reads_libcontentbench_json_rows() -> None:
    result = collect_completion_count_probe(
        {
            "stdout": '{"rows":[{"status":"ok","success":true},{"status":"error","success":false}]}',
            "stderr": "",
        }
    )

    assert result["observed"] is True
    assert result["source"] == "runtime_output.json.rows"
    assert result["value"] == 1
    flat = result["flat"]
    assert isinstance(flat, dict)
    assert flat["benchmark.completion_count"] == 1


def test_process_thread_count_probe_reads_libcontentbench_sampler_summary() -> None:
    result = collect_process_thread_count_probe(
        {
            "stdout": '{"rows":[{"config_summary":"outer_jobs=6, blas_threads=6","sampler_summary":{"peak_threads":42}}]}',
            "stderr": "",
            "runtime_output": {},
            "collect_profile_data": {},
            "per_pid_runtime_maps": {},
        }
    )

    assert result["observed"] is True
    assert result["source"] == "runtime_output.json.sampler_summary.peak_threads"
    assert result["value"] == 42
    flat = result["flat"]
    assert isinstance(flat, dict)
    assert flat["process.thread_count"] == 42


def test_process_thread_count_probe_reads_libcontentbench_metrics_timeseries(tmp_path) -> None:
    output_dir = tmp_path / "case_01"
    output_dir.mkdir()
    (output_dir / "metrics_timeseries.csv").write_text(
        "ts,processes,threads,rss_bytes,voluntary_ctx,involuntary_ctx,read_count,write_count,read_bytes,write_bytes\n"
        "10.0,1,3,100,1,1,0,0,0,0\n"
        "10.2,2,9,120,2,1,0,0,0,0\n",
        encoding="utf-8",
    )

    result = collect_process_thread_count_probe(
        {
            "stdout": '{"rows":[{"output_dir":"' + str(output_dir) + '","sampler_summary":{"peak_threads":9}}]}',
            "stderr": "",
            "runtime_output": {},
            "collect_profile_data": {},
            "per_pid_runtime_maps": {},
        }
    )

    assert result["observed"] is True
    assert result["source"] == "runtime_output.json.metrics_timeseries.threads"
    assert result["value"] == 9
    structured = result["structured"]
    assert isinstance(structured, dict)
    timeseries = structured["timeseries"]
    assert isinstance(timeseries, list)
    assert len(timeseries) == 2
    assert timeseries[0]["value"] == 3.0
    assert timeseries[1]["value"] == 9.0


def test_process_io_syscall_rate_probe_reads_libcontentbench_sampler_summary() -> None:
    result = collect_process_io_syscall_rate_probe(
        {
            "stdout": '{"rows":[{"wall_time_sec":2.0,"sampler_summary":{"write_count_delta":8}}]}',
            "stderr": "",
            "primary_pid": 999999999,
            "target_pids": [999999999],
        }
    )

    assert result["observed"] is True
    assert result["source"] == "runtime_output.json.sampler_summary.write_count_delta"
    assert result["value"] == 4.0
    flat = result["flat"]
    assert isinstance(flat, dict)
    assert flat["process.io.syscall_rate"] == 4.0


def test_process_io_syscall_rate_probe_reads_libcontentbench_metrics_timeseries(tmp_path) -> None:
    output_dir = tmp_path / "case_01"
    output_dir.mkdir()
    (output_dir / "metrics_timeseries.csv").write_text(
        "ts,processes,threads,rss_bytes,voluntary_ctx,involuntary_ctx,read_count,write_count,read_bytes,write_bytes\n"
        "10.0,1,3,100,1,1,0,2,0,0\n"
        "10.5,1,3,100,1,1,0,8,0,0\n"
        "11.0,1,3,100,1,1,0,10,0,0\n",
        encoding="utf-8",
    )

    result = collect_process_io_syscall_rate_probe(
        {
            "stdout": '{"rows":[{"output_dir":"' + str(output_dir) + '"}]}',
            "stderr": "",
            "primary_pid": 999999999,
            "target_pids": [999999999],
        }
    )

    assert result["observed"] is True
    assert result["source"] == "runtime_output.json.metrics_timeseries.write_count"
    assert result["value"] == 12.0
    structured = result["structured"]
    assert isinstance(structured, dict)
    timeseries = structured["timeseries"]
    assert isinstance(timeseries, list)
    assert len(timeseries) == 2
    assert timeseries[0]["value"] == 12.0
    assert timeseries[1]["value"] == 4.0


def test_threadpoolctl_probe_extracts_threadpool_info_from_json_stdout() -> None:
    result = collect_threadpool_info_probe(
        {
            "stdout": '{"threadpools":[{"internal_api":"openmp","num_threads":4},{"internal_api":"openblas","num_threads":1}]}'
        }
    )

    assert result["observed"] is True
    assert result["value"] == 4
    assert result["flat"]["threadpoolctl.threadpool_count"] == 2
    assert result["flat"]["native_threadpool.loaded_runtime_count"] == 2
    assert result["flat"]["native_threadpool.num_threads_per_pool"] == 4
    assert result["excerpt"] == ["openmp threads=4", "openblas threads=1"]


def test_runtime_launch_mpi_process_count_probe_reads_cp2k_output(tmp_path) -> None:
    (tmp_path / "demo.out").write_text(
        "GLOBAL| Total number of message passing processes                             12\n",
        encoding="utf-8",
    )

    result = collect_mpi_process_count_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 12
    assert result["flat"]["runtime.launch.mpi_process_count"] == 12


def test_benchmark_log_timing_report_probe_extracts_ot_step_time(tmp_path) -> None:
    (tmp_path / "cp2k.out").write_text(
        "  1 OT DIIS     0.15E+00    7.0     0.10146221       -78.5965414196 -7.86E+01\n"
        "  2 OT DIIS     0.15E+00    8.4     0.09097050       -83.7787919349 -5.18E+00\n",
        encoding="utf-8",
    )

    result = collect_timing_report_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 8.4
    assert result["flat"]["benchmark.log.timing_report"] == 8.4


def test_benchmark_log_section_presence_probe_counts_matching_lines(tmp_path) -> None:
    (tmp_path / "cp2k.out").write_text(
        "SCF WAVEFUNCTION OPTIMIZATION\n"
        "outer SCF iter =    2 RMS gradient =   0.17E-07 energy =        -86.7177803603\n"
        "ENERGY| Total FORCE_EVAL ( QMMM ) energy [a.u.]:            -86.717780360275896\n",
        encoding="utf-8",
    )

    result = collect_section_presence_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 3
    assert result["flat"]["benchmark.log.section_presence"] == 3


def test_benchmark_log_message_passing_probe_extracts_numeric_value(tmp_path) -> None:
    (tmp_path / "cp2k.out").write_text(
        "MESSAGE PASSING total time [s] 12.5\nGLOBAL| Total number of message passing processes 4\n",
        encoding="utf-8",
    )

    result = collect_message_passing_report_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 12.5
    assert result["flat"]["benchmark.log.message_passing_report"] == 12.5


def test_cp2k_log_timing_summary_probe_counts_cp2k_timing_entries(tmp_path) -> None:
    (tmp_path / "cp2k.out").write_text(
        "  1 OT DIIS     0.15E+00    7.0     0.10146221       -78.5965414196 -7.86E+01\n"
        "outer SCF iter =    2 RMS gradient =   0.17E-07 energy =        -86.7177803603\n"
        "*** SCF run converged in     9 steps ***\n",
        encoding="utf-8",
    )

    result = collect_cp2k_timing_summary_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 3
    assert result["flat"]["cp2k.log.timing_summary"] == 3


def test_cp2k_log_rank_max_self_time_probe_reuses_timing_report_value(tmp_path) -> None:
    (tmp_path / "cp2k.out").write_text(
        "  1 OT DIIS     0.15E+00    6.5     0.00136275       -86.7182029340 -8.67E+01\n"
        "  2 OT DIIS     0.15E+00    7.1     0.00069515       -86.7195541612 -1.35E-03\n",
        encoding="utf-8",
    )

    result = collect_cp2k_rank_max_self_time_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 7.1
    assert result["flat"]["cp2k.log.rank_max_self_time"] == 7.1


def test_cp2k_log_message_passing_summary_probe_reuses_message_passing_value(tmp_path) -> None:
    (tmp_path / "cp2k.out").write_text(
        "MESSAGE PASSING total time [s] 9.75\n",
        encoding="utf-8",
    )

    result = collect_cp2k_message_passing_summary_probe({"workdir": str(tmp_path), "launch_env": {}})

    assert result["observed"] is True
    assert result["value"] == 9.75
    assert result["flat"]["cp2k.log.message_passing_summary"] == 9.75


def test_openblas_runtime_api_probes_extract_threadpool_and_runtime_values() -> None:
    payload = {
        "stdout": '{"threadpools":[{"internal_api":"openblas","user_api":"blas","num_threads":6,"threading_layer":"pthreads","architecture":"HASWELL"}]}',
        "stderr": "",
        "per_pid_runtime_maps": {"1234": {"Cpus_allowed_list": "0-23"}},
        "primary_pid": 1234,
    }

    num_threads = collect_openblas_get_num_threads_probe(payload)
    num_procs = collect_openblas_get_num_procs_probe(payload)
    config = collect_openblas_get_config_probe(payload)
    corename = collect_openblas_get_corename_probe(payload)
    parallel = collect_openblas_get_parallel_probe(payload)

    assert num_threads["observed"] is True
    assert num_threads["value"] == 6
    assert num_threads["flat"]["runtime.api.openblas_get_num_threads"] == 6
    assert num_procs["observed"] is True
    assert num_procs["value"] == 24
    assert num_procs["flat"]["runtime.api.openblas_get_num_procs"] == 24
    assert config["observed"] is True
    assert "openblas" in str(config["value"]).lower()
    assert corename["observed"] is True
    assert corename["value"] == "HASWELL"
    assert parallel["observed"] is True
    assert parallel["value"] == "pthreads"
