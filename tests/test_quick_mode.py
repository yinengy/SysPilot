from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections.abc import Sequence
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import _run_live_profiling_pipeline, cli
from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig
from syspilot.core.llm import LLMProviderError
from syspilot.quick_mode import (
    QuickBaselineWindow,
    QuickModeDefaults,
    QuickRunMetadata,
    QuickShortResult,
    QuickSignalSpec,
    SampledSignalPoint,
    _build_signal_window_groups,
    _extract_signal_specs,
    _format_objective_speedup,
    _interleave_windows_by_signal,
    _objective_speedup_ratio,
    _plain_live_output_callback,
    _materialize_result,
    classify_signal_usefulness,
    detect_baseline_window,
    rank_abnormal_windows,
    select_promotions,
)
from syspilot.ui.output_sink import PlainOutputSink
from rich.console import Console


class _DummyQuickAgent:
    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})

    async def step(self, prompt: str) -> SimpleNamespace:
        _ = prompt
        return SimpleNamespace(
            content=(
                '{"summary":"screen env candidates","manual_citation":"rule","evidence":["signal"],'
                '"actions":['
                '{"config_type":"env_var","parameter":"OMP_PROC_BIND","new_value":"close","reason":"locality","manual_citation":"rule"},'
                '{"config_type":"env_var","parameter":"OMP_WAIT_POLICY","new_value":"ACTIVE","reason":"latency","manual_citation":"rule"}'
                "]}"
            )
        )

    async def _execute_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        _ = name
        _ = arguments
        return {}

    def enable_llm_prompt_logging(self, path: Path) -> Path:
        return path


class _LiveProfileAgent(SysPilotAgent):
    def __init__(self) -> None:
        super().__init__(SysPilotConfig())
        self.context = SimpleNamespace(loaded_manuals={}, target_pid=None)

    async def _execute_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        _ = name
        _ = arguments
        return {}


class _CaptureEmitter:
    def __init__(self, run_id: str | None = None, capacity: int = 256, critical_reserve: int = 16) -> None:
        _ = capacity
        _ = critical_reserve
        self.run_id = run_id or "quick-test-run"
        self.events: list[dict[str, object]] = []

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        self.events.append(
            {"event": event_name, "payload": payload or {}, "event_type": event_type or "event"}
        )
        return True


class _RecordingTuiSink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def is_tui_active(self) -> bool:
        return True

    def print(self, message: str = "") -> None:
        self.messages.append(message)

    def panel(self, body: str, *, title: str) -> None:
        _ = body
        _ = title

    def table(
        self,
        *,
        title: str,
        columns: Sequence[tuple[str, str | None]],
        rows: Sequence[Sequence[str]],
    ) -> None:
        _ = title
        _ = columns
        _ = rows


def _noop(*args: object, **kwargs: object) -> None:
    _ = args
    _ = kwargs


def _sample_entries(values: Sequence[float | int], probe_name: str = "process.memory.rss") -> list[dict[str, object]]:
    return [
        {
            "time_offset_s": idx * 5.0,
            "results": {
                probe_name: {
                    "observed": True,
                    "source": "proc_status",
                    "value": value,
                    "flat": {"value": value},
                    "structured": {"value": value},
                }
            },
        }
        for idx, value in enumerate(values)
    ]


def _multi_probe_sample_entries(probe_values: dict[str, Sequence[float | int]]) -> list[dict[str, object]]:
    sample_count = max(len(values) for values in probe_values.values()) if probe_values else 0
    entries: list[dict[str, object]] = []
    for idx in range(sample_count):
        results: dict[str, object] = {}
        for probe_name, values in probe_values.items():
            if idx >= len(values):
                continue
            value = values[idx]
            results[probe_name] = {
                "observed": True,
                "source": "proc_status",
                "value": value,
                "flat": {"value": value},
                "structured": {"value": value},
            }
        entries.append({"time_offset_s": idx * 5.0, "results": results})
    return entries


def test_quick_mode_defaults_match_requested_behavior() -> None:
    defaults = QuickModeDefaults()

    assert defaults.sample_interval_sec == 5.0
    assert defaults.short_measurement_sec == 30.0
    assert defaults.full_run_ratio == 0.20
    assert defaults.max_full_runs_per_group == 3
    assert defaults.short_measurement_sec >= 3 * defaults.sample_interval_sec
    assert defaults.window_samples == 6


def test_detect_baseline_window_prefers_decrease_burst_window_over_late_plateau() -> None:
    points = [
        SampledSignalPoint(0.0, 10.0),
        SampledSignalPoint(5.0, 10.0),
        SampledSignalPoint(10.0, 20.0),
        SampledSignalPoint(15.0, 21.0),
        SampledSignalPoint(20.0, 20.0),
        SampledSignalPoint(25.0, 19.0),
    ]

    window = detect_baseline_window("sig", points, "decrease", 4)

    assert window is not None
    assert abs(window.start_offset_s - 5.0) < 1e-9
    assert window.window_summary["directional_delta_sum"] == 11.0


def test_detect_baseline_window_prefers_increase_burst_window_over_late_plateau() -> None:
    points = [
        SampledSignalPoint(0.0, 20.0),
        SampledSignalPoint(5.0, 20.0),
        SampledSignalPoint(10.0, 10.0),
        SampledSignalPoint(15.0, 9.0),
        SampledSignalPoint(20.0, 10.0),
        SampledSignalPoint(25.0, 11.0),
    ]

    window = detect_baseline_window("sig", points, "increase", 4)

    assert window is not None
    assert abs(window.start_offset_s - 5.0) < 1e-9
    assert window.window_summary["directional_delta_sum"] == 11.0


def test_detect_baseline_window_falls_back_to_level_when_no_burst_exists() -> None:
    points = [
        SampledSignalPoint(0.0, 20.0),
        SampledSignalPoint(5.0, 20.0),
        SampledSignalPoint(10.0, 20.0),
        SampledSignalPoint(15.0, 10.0),
        SampledSignalPoint(20.0, 10.0),
        SampledSignalPoint(25.0, 10.0),
    ]

    window = detect_baseline_window("sig", points, "decrease", 4)

    assert window is not None
    assert abs(window.start_offset_s - 0.0) < 1e-9
    assert window.window_summary["directional_delta_sum"] == 0.0


def test_rank_abnormal_windows_orders_multiple_signals() -> None:
    defaults = QuickModeDefaults()
    specs = [
        QuickSignalSpec("sig.a", "a", "rss_mb", "process.memory.rss", "decrease", "resource_signal"),
        QuickSignalSpec("sig.b", "b", "queue_depth", "queue.depth", "decrease", "resource_signal"),
    ]
    artifact = {
        "profiling_json": {
            "periodic_probe_samples": [
                {
                    "time_offset_s": idx * 5.0,
                    "results": {
                        "process.memory.rss": {"observed": True, "value": value_a, "flat": {"value": value_a}},
                        "queue.depth": {"observed": True, "value": value_b, "flat": {"value": value_b}},
                    },
                }
                for idx, (value_a, value_b) in enumerate(
                    zip(
                        [10, 10, 11, 30, 30, 30, 30, 10],
                        [5, 5, 5, 8, 8, 8, 8, 5],
                    )
                )
            ]
        }
    }

    ranked, usefulness, profiling_available = rank_abnormal_windows(specs, artifact, defaults)

    assert profiling_available is True
    assert ranked
    assert ranked[0].signal_ref == "sig.a"
    assert any(entry.status == "checking" for entry in usefulness)


def test_interleave_windows_by_signal_rotates_first_pass_across_distinct_signals() -> None:
    def _window(window_id: str, signal_ref: str, severity: float, abnormality_score: float) -> QuickBaselineWindow:
        return QuickBaselineWindow(
            window_id=window_id,
            signal_name=signal_ref,
            signal_ref=signal_ref,
            hypothesis_name="",
            start_offset_s=0.0,
            end_offset_s=25.0,
            duration_s=25.0,
            sample_count=6,
            expected_direction="decrease",
            severity=severity,
            abnormality_score=abnormality_score,
            candidate_controls=[],
            window_summary={},
            global_summary={},
            reason="",
        )

    ordered = _interleave_windows_by_signal(
        [
            _window("rss-w1", "sig.rss", 44.0, 5030.0),
            _window("rss-w2", "sig.rss", 4.9, 565.0),
            _window("cgroup-w1", "sig.cgroup", 0.0, 0.0),
            _window("cgroup-w2", "sig.cgroup", 0.0, 0.0),
        ]
    )

    assert [window.window_id for window in ordered] == [
        "rss-w1",
        "cgroup-w1",
        "rss-w2",
        "cgroup-w2",
    ]


def test_build_signal_window_groups_keeps_multiple_windows_under_one_signal() -> None:
    def _window(window_id: str, signal_ref: str, signal_name: str, start: float, end: float) -> QuickBaselineWindow:
        return QuickBaselineWindow(
            window_id=window_id,
            signal_name=signal_name,
            signal_ref=signal_ref,
            hypothesis_name="",
            start_offset_s=start,
            end_offset_s=end,
            duration_s=end - start,
            sample_count=6,
            expected_direction="decrease",
            severity=10.0,
            abnormality_score=1.0,
            candidate_controls=[],
            window_summary={},
            global_summary={},
            reason="",
        )

    groups = _build_signal_window_groups(
        [
            _window("rss-w1", "sig.rss", "process_rss_peak", 0.0, 25.0),
            _window("cgroup-w1", "sig.cgroup", "cgroup_memory_events_high", 25.0, 50.0),
            _window("rss-w2", "sig.rss", "process_rss_peak", 250.0, 275.0),
        ],
        checked_window_ids=["rss-w1", "cgroup-w1"],
    )

    assert [group.signal_name for group in groups] == [
        "process_rss_peak",
        "cgroup_memory_events_high",
    ]
    assert [window.window_id for window in groups[0].windows] == ["rss-w1", "rss-w2"]
    assert [window.checked for window in groups[0].windows] == [True, False]
    assert [window.window_id for window in groups[1].windows] == ["cgroup-w1"]


def test_objective_speedup_ratio_formats_better_and_slower() -> None:
    assert _objective_speedup_ratio(100.0, 80.0, "minimize") == 1.25
    assert _format_objective_speedup(1.25) == "1.25x (better)"
    assert _objective_speedup_ratio(100.0, 125.0, "minimize") == 0.8
    assert _format_objective_speedup(0.8) == "0.80x (slower)"
    assert _objective_speedup_ratio(100.0, 130.0, "maximize") == 1.3
    assert _format_objective_speedup(1.3) == "1.30x (better)"


def test_plain_live_output_callback_keeps_short_run_footer_lines_when_tui_active() -> None:
    sink = _RecordingTuiSink()

    _plain_live_output_callback(
        sink,
        "stdout",
        "RUN_LABEL=short-1\n",
        run_kind="short",
    )

    assert sink.messages == ["  RUN_LABEL=short-1"]


def test_plain_live_output_callback_suppresses_short_run_footer_lines_for_plain_sink() -> None:
    console = Console(record=True, width=200)
    sink = PlainOutputSink(console=console)

    _plain_live_output_callback(
        sink,
        "stdout",
        "RUN_LABEL=short-1\n",
        run_kind="short",
    )

    assert console.export_text() == ""


def test_extract_signal_specs_merges_semantic_duplicates_and_skips_outcomes() -> None:
    specs = _extract_signal_specs(
        {
            "profile_first": [
                {
                    "signal_ref": "kafka.signal_cgroup_memory_events_high",
                    "role": "resource_signal",
                    "hypothesis_name": "kafka-memory",
                    "candidate_controls": ["KAFKA_CACHE_BYTES"],
                    "signal": {
                        "id": "signal_cgroup_memory_events_high",
                        "metric": "cgroup_memory_events_high",
                        "virtual_probe": "cgroup.memory.events.high",
                        "expected_direction_when_mitigated": "decrease",
                        "role": "resource_signal",
                    },
                },
                {
                    "signal_ref": "kafka.signal_completion_elapsed_time",
                    "role": "outcome",
                    "hypothesis_name": "kafka-memory",
                    "candidate_controls": [],
                    "signal": {
                        "id": "signal_completion_elapsed_time",
                        "metric": "benchmark_elapsed_time",
                        "virtual_probe": "benchmark.elapsed_time",
                        "expected_direction_when_mitigated": "decrease",
                        "role": "outcome",
                    },
                },
            ]
        },
        [
            {
                "hypothesis_name": "rocksdb-memory",
                "validation_signals": [
                    {
                        "ref": "rocksdb.sig_memory_events_high",
                        "role": "resource_signal",
                        "candidate_controls": ["ROCKSDB_BLOCK_CACHE_BYTES"],
                        "signal": {
                            "id": "sig_memory_events_high",
                            "metric": "cgroup_memory_events_high",
                            "virtual_probe": "cgroup.memory.events.high",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "resource_signal",
                        },
                    },
                    {
                        "ref": "rocksdb.sig_rss_peak",
                        "role": "guardrail",
                        "candidate_controls": ["ROCKSDB_TOTAL_MEMTABLE_BYTES"],
                        "signal": {
                            "id": "sig_rss_peak",
                            "metric": "process_rss_peak",
                            "virtual_probe": "process.memory.rss_peak",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "guardrail",
                        },
                    },
                    {
                        "ref": "kafka.signal_process_rss_peak",
                        "role": "guardrail",
                        "candidate_controls": ["KAFKA_COMMIT_INTERVAL_MS"],
                        "signal": {
                            "id": "signal_process_rss_peak",
                            "metric": "process_rss_peak",
                            "virtual_probe": "process.memory.rss_peak",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "guardrail",
                        },
                    },
                ],
            }
        ],
    )

    assert [spec.metric for spec in specs] == ["cgroup_memory_events_high", "process_rss_peak"]
    cgroup_spec = next(spec for spec in specs if spec.metric == "cgroup_memory_events_high")
    assert cgroup_spec.candidate_controls == [
        "KAFKA_CACHE_BYTES",
        "ROCKSDB_BLOCK_CACHE_BYTES",
    ]
    rss_spec = next(spec for spec in specs if spec.metric == "process_rss_peak")
    assert rss_spec.candidate_controls == [
        "ROCKSDB_TOTAL_MEMTABLE_BYTES",
        "KAFKA_COMMIT_INTERVAL_MS",
    ]


def test_rank_abnormal_windows_accepts_cgroup_memory_high_samples() -> None:
    defaults = QuickModeDefaults()
    specs = [
        QuickSignalSpec(
            "sig.cgroup",
            "cgroup_high",
            "cgroup_memory_events_high",
            "cgroup.memory.events.high",
            "decrease",
            "resource_signal",
        )
    ]
    artifact = {
        "profiling_json": {
            "periodic_probe_samples": _sample_entries(
                [0, 0, 0, 2, 4, 4, 4, 4, 4, 4],
                probe_name="cgroup.memory.events.high",
            )
        }
    }

    ranked, usefulness, profiling_available = rank_abnormal_windows(specs, artifact, defaults)

    assert profiling_available is True
    assert ranked
    assert ranked[0].signal_ref == "sig.cgroup"
    assert abs(ranked[0].start_offset_s - 10.0) < 1e-9
    assert ranked[0].window_summary["directional_delta_sum"] == 4.0
    assert any(entry.status == "checking" for entry in usefulness)


def test_select_promotions_caps_fraction() -> None:
    results = [
        QuickShortResult(1, "c1", "short-1", "sig", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 1.0, 10.0, "screened"),
        QuickShortResult(2, "c2", "short-2", "sig", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 1.0, 8.0, "screened"),
        QuickShortResult(3, "c3", "short-3", "sig", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 1.0, 3.0, "screened"),
        QuickShortResult(4, "c4", "short-4", "sig", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, None, None, "unavailable"),
    ]

    decisions = select_promotions(results, full_run_ratio=0.34, max_full_runs_per_group=2)

    promoted = [item.candidate_id for item in decisions if item.promoted]
    assert promoted == [1, 2]


def test_classify_signal_usefulness_reports_predictive() -> None:
    spec = QuickSignalSpec("sig.ref", "sig", "rss", "process.memory.rss", "decrease", "resource_signal")
    defaults = QuickModeDefaults()
    short_results = [
        QuickShortResult(1, "c1", "short-1", "sig.ref", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 10.0, 15.0, "screened"),
        QuickShortResult(2, "c2", "short-2", "sig.ref", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 11.0, 5.0, "screened"),
    ]
    full_results = [
        QuickShortResult(1, "c1", "full-1", "sig.ref", "rss", "w1", ["cmd"], {}, "full", 3.0, 0, "completed", False, 9.0, 12.0, "screened", objective_metric="throughput", objective_mode="maximize", objective_value=120.0, objective_improvement_pct=20.0, objective_improved=True),
        QuickShortResult(2, "c2", "full-2", "sig.ref", "rss", "w1", ["cmd"], {}, "full", 3.0, 0, "completed", False, 10.5, 4.0, "screened", objective_metric="throughput", objective_mode="maximize", objective_value=104.0, objective_improvement_pct=4.0, objective_improved=True),
    ]

    usefulness = classify_signal_usefulness(
        target_spec=spec,
        short_results=short_results,
        full_results=full_results,
        defaults=defaults,
    )

    assert usefulness.status == "predictive"


def test_classify_signal_usefulness_uses_full_run_outcome() -> None:
    spec = QuickSignalSpec("sig.ref", "sig", "rss", "process.memory.rss", "decrease", "resource_signal")
    defaults = QuickModeDefaults()
    short_results = [
        QuickShortResult(1, "c1", "short-1", "sig.ref", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 10.0, 15.0, "screened"),
        QuickShortResult(2, "c2", "short-2", "sig.ref", "rss", "w1", ["cmd"], {}, "short", 1.0, 0, "screen_budget_reached", True, 11.0, 10.0, "screened"),
    ]
    full_results = [
        QuickShortResult(1, "c1", "full-1", "sig.ref", "rss", "w1", ["cmd"], {}, "full", 3.0, 0, "completed", False, 9.0, 13.0, "screened", objective_metric="throughput", objective_mode="maximize", objective_value=95.0, objective_improvement_pct=-5.0, objective_improved=False),
        QuickShortResult(2, "c2", "full-2", "sig.ref", "rss", "w1", ["cmd"], {}, "full", 3.0, 0, "completed", False, 10.0, 6.0, "screened", objective_metric="throughput", objective_mode="maximize", objective_value=112.0, objective_improvement_pct=12.0, objective_improved=True),
    ]

    usefulness = classify_signal_usefulness(
        target_spec=spec,
        short_results=short_results,
        full_results=full_results,
        defaults=defaults,
    )

    assert usefulness.status == "weak"


def test_materialize_result_uses_exact_baseline_window_end_offset() -> None:
    defaults = QuickModeDefaults()
    metadata = QuickRunMetadata(
        run_id="run-1",
        config_id="short-1",
        run_kind="short",
        candidate_id=1,
        candidate_label="candidate-1",
        signal_ref="sig.ref",
        signal_name="rss",
        window_id="sig-w1",
        command=["echo", "hi"],
        env_overrides={},
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:35+00:00",
        wall_time_s=35.0,
        returncode=-15,
        stop_reason="screen_budget_reached",
        stopped_intentionally=True,
        artifact_path="/tmp/run_artifact.json",
        stdout_path="/tmp/stdout.txt",
        stderr_path="/tmp/stderr.txt",
    )
    baseline_window = QuickBaselineWindow(
        window_id="sig-w1",
        signal_name="rss",
        signal_ref="sig.ref",
        hypothesis_name="demo",
        start_offset_s=10.0,
        end_offset_s=35.0,
        duration_s=25.0,
        sample_count=6,
        expected_direction="decrease",
        severity=50.0,
        abnormality_score=10.0,
        candidate_controls=[],
        window_summary={"mean": 20.0, "min": 20.0, "max": 20.0, "delta_from_global_mean": 10.0},
        global_summary={"mean": 10.0, "min": 10.0, "max": 20.0, "range": 10.0},
        reason="demo",
    )
    points = [
        SampledSignalPoint(10.0, 10.0),
        SampledSignalPoint(15.0, 10.0),
        SampledSignalPoint(20.0, 10.0),
        SampledSignalPoint(25.0, 10.0),
        SampledSignalPoint(30.0, 10.0),
        SampledSignalPoint(35.0, 10.0),
        SampledSignalPoint(40.0, 100.0),
    ]

    result = _materialize_result(
        candidate_id=1,
        candidate_label="candidate-1",
        config_id="short-1",
        signal_ref="sig.ref",
        signal_name="rss",
        window_id="sig-w1",
        command=["echo", "hi"],
        env_overrides={},
        run_kind="short",
        artifact={
            "wall_time": 35.0,
            "returncode": -15,
            "stop_reason": "screen_budget_reached",
            "stopped_intentionally": True,
        },
        metadata=metadata,
        target_points=points,
        baseline_window=baseline_window,
        defaults=defaults,
    )

    assert result.target_signal_mean == 10.0
    assert result.target_signal_improvement_pct == 50.0


def test_run_live_profiling_pipeline_marks_intentional_short_stop() -> None:
    agent = _LiveProfileAgent()

    async def _run() -> dict[str, Any]:
        return await _run_live_profiling_pipeline(
            agent,
            ["python3", "-c", "import time; time.sleep(5)"],
            timeout=10,
            stop_after_seconds=0.2,
            periodic_sample_interval_s=0.1,
        )

    artifact = asyncio.run(_run())

    assert artifact["timed_out"] is False
    assert artifact["stopped_intentionally"] is True
    assert artifact["stop_reason"] == "screen_budget_reached"


def test_quick_mode_dispatches_from_cli() -> None:
    runner = CliRunner()
    called = {"quick": 0, "default": 0}

    def _fake_quick(**kwargs: object) -> None:
        _ = kwargs
        called["quick"] += 1

    def _fake_default(*args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs
        called["default"] += 1

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyQuickAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_quick_tune_loop", _fake_quick),
        patch("syspilot.cli._run_spawn_tune_loop", _fake_default),
    ):
        result = runner.invoke(
            cli,
            ["tune", "--quick-mode", "--ui-mode", "plain", "--command", "echo hi"],
        )

    assert result.exit_code == 0
    assert called == {"quick": 1, "default": 0}


def test_quick_mode_end_to_end_writes_structured_artifacts_and_objective_winner() -> None:
    runner = CliRunner()

    class _ObjectiveQuickAgent(_DummyQuickAgent):
        async def step(self, prompt: str) -> SimpleNamespace:
            _ = prompt
            actions = [
                {
                    "config_type": "env_var",
                    "parameter": f"VAR_{idx}",
                    "new_value": str(idx),
                    "reason": f"candidate {idx}",
                    "manual_citation": "rule",
                }
                for idx in range(1, 11)
            ]
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "summary": "screen many env candidates",
                        "manual_citation": "rule",
                        "evidence": ["signal"],
                        "actions": actions,
                    }
                )
            )

    structured_bundle = {
        "profile_first": [
            {
                "signal_ref": "demo.sig",
                "role": "resource_signal",
                "hypothesis_name": "demo-hypothesis",
                "candidate_controls": ["OMP_PROC_BIND", "OMP_WAIT_POLICY"],
                "signal": {
                    "id": "sig",
                    "metric": "rss_mb",
                    "virtual_probe": "process.memory.rss",
                    "expected_direction_when_mitigated": "decrease",
                    "role": "resource_signal",
                },
            }
        ],
        "hypothesis_profiles": [
            {
                "hypothesis_id": "demo-hypothesis",
                "validation_signals": [
                    {
                        "ref": "demo.sig",
                        "role": "resource_signal",
                        "candidate_controls": ["OMP_PROC_BIND", "OMP_WAIT_POLICY"],
                        "signal": {
                            "id": "sig",
                            "metric": "rss_mb",
                            "virtual_probe": "process.memory.rss",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "resource_signal",
                        },
                    }
                ],
            }
        ],
    }

    baseline = {
        "pid": 1000,
        "launch_command": ["echo", "hi"],
        "wall_time": 5.0,
        "stdout": "throughput=100\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "profiling_text": "{}",
        "profiling_json": {"periodic_probe_samples": _sample_entries([10, 10, 10, 20, 20, 20, 20, 10, 10, 10])},
        "tool_calls_used": 0,
        "stop_reason": "completed",
        "stopped_intentionally": False,
    }

    short_runs: list[dict[str, Any]] = []
    short_signal_sets = [
        [10, 10, 10, 14, 14, 14, 14, 10, 10, 10],
        [10, 10, 10, 16, 16, 16, 16, 10, 10, 10],
        [10, 10, 10, 17, 17, 17, 17, 10, 10, 10],
        [10, 10, 10, 18, 18, 18, 18, 10, 10, 10],
        [10, 10, 10, 18.5, 18.5, 18.5, 18.5, 10, 10, 10],
        [10, 10, 10, 19, 19, 19, 19, 10, 10, 10],
        [10, 10, 10, 19.2, 19.2, 19.2, 19.2, 10, 10, 10],
        [10, 10, 10, 19.4, 19.4, 19.4, 19.4, 10, 10, 10],
        [10, 10, 10, 19.6, 19.6, 19.6, 19.6, 10, 10, 10],
        [10, 10, 10, 19.8, 19.8, 19.8, 19.8, 10, 10, 10],
    ]
    for idx, values in enumerate(short_signal_sets, start=1):
        short_runs.append(
            {
                "pid": 1000 + idx,
                "launch_command": ["echo", "hi"],
                "wall_time": 2.0,
                "stdout": (
                    f"[kafka-runner] Progress poll 1: completes {idx}/10 (10.0%)\n"
                    f"RUN_LABEL=short-{idx}\n"
                    f"RESULT_JSON=/tmp/short-{idx}/result.json\n"
                    "RESULT_STATUS=timeout\n"
                ),
                "stderr": "short-run warning\n" if idx == 1 else "",
                "returncode": -15,
                "timed_out": False,
                "profiling_text": "{}",
                "profiling_json": {"periodic_probe_samples": _sample_entries(values)},
                "tool_calls_used": 0,
                "stop_reason": "screen_budget_reached",
                "stopped_intentionally": True,
            }
        )

    full_runs = [
        {
            "pid": 2001,
            "launch_command": ["echo", "hi"],
            "wall_time": 4.0,
            "stdout": "throughput=90\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "{}",
            "profiling_json": {"periodic_probe_samples": _sample_entries([10, 10, 10, 14, 14, 14, 14, 10, 10, 10])},
            "tool_calls_used": 0,
            "stop_reason": "completed",
            "stopped_intentionally": False,
        },
        {
            "pid": 2002,
            "launch_command": ["echo", "hi"],
            "wall_time": 4.5,
            "stdout": "throughput=130\nRUN_LABEL=full-2\nRESULT_STATUS=success\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "{}",
            "profiling_json": {"periodic_probe_samples": _sample_entries([10, 10, 10, 16, 16, 16, 16, 10, 10, 10])},
            "tool_calls_used": 0,
            "stop_reason": "completed",
            "stopped_intentionally": False,
        },
    ]
    artifacts = [baseline, *short_runs, *full_runs]

    async def _fake_profile(*args: object, **kwargs: object) -> dict[str, Any]:
        _ = args
        artifact = artifacts.pop(0)
        callback = kwargs.get("live_output_callback")
        if callable(callback):
            for line in str(artifact.get("stdout", "")).splitlines(True):
                callback("stdout", line)
            for line in str(artifact.get("stderr", "")).splitlines(True):
                callback("stderr", line)
        return artifact

    async def _fake_extract_metrics(agent: SysPilotAgent, output: str, context: str) -> dict[str, Any]:
        _ = agent
        _ = context
        for line in output.splitlines():
            if line.startswith("throughput="):
                return {"throughput": float(line.split("=", 1)[1])}
        return {"no_metrics": True}

    with runner.isolated_filesystem():
        with (
            patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
            patch("syspilot.cli.SysPilotAgent", _ObjectiveQuickAgent),
            patch("syspilot.cli._register_default_skills", _noop),
            patch("syspilot.cli._register_profiling_tools", _noop),
            patch("syspilot.cli._register_tuning_tools", _noop),
            patch("syspilot.cli._build_contract_context_bundle", lambda *args, **kwargs: structured_bundle),
            patch("syspilot.cli._run_live_profiling_pipeline", _fake_profile),
            patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
        ):
            result = runner.invoke(
                cli,
                [
                    "tune",
                    "--quick-mode",
                    "--ui-mode",
                    "plain",
                    "--command",
                    "echo hi",
                    "--primary-objective",
                    "throughput",
                    "--primary-objective-mode",
                    "maximize",
                    "--contract",
                    "demo",
                    "--max-iterations",
                    "10",
                    "--timeout",
                    "5",
                ],
            )

        run_dir = Path("artifacts") / "quick_tune_runs" / "quick-test-run"
        report_path = run_dir / "quick_report.json"
        assert result.exit_code == 0
        assert report_path.exists()
        assert (run_dir / "signal_samples.jsonl").exists()
        assert (run_dir / "signal_samples.csv").exists()
        assert (run_dir / "baseline_run_metadata.jsonl").exists()
        assert (run_dir / "short_run_metadata.jsonl").exists()
        assert (run_dir / "full_run_metadata.jsonl").exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["selected_candidate_id"] == 2
        assert report["primary_objective_metric"] == "throughput"
        assert report["final_selection_basis"] == "throughput"
        assert report["conclusion"] == "significant_improvement_found"
        assert report["abnormal_windows"]
        assert report["signal_windows"]
        assert report["signal_windows"][0]["windows"]
        assert len(report["short_results"]) == 10
        assert len(report["promotions"]) == 10
        assert len(report["full_results"]) == 2
        assert report["full_results"][0]["objective_speedup_ratio"] == 0.9
        assert report["full_results"][0]["objective_speedup_label"] == "slower"
        assert report["full_results"][1]["objective_speedup_ratio"] == 1.3
        assert report["full_results"][1]["objective_speedup_label"] == "better"
        assert any(entry["status"] == "weak" for entry in report["signal_usefulness"])
        assert Path(report["baseline_summary"]["baseline_run"]["artifact_path"]).exists()
        assert Path(report["baseline_summary"]["baseline_run"]["stdout_path"]).exists()
        assert report["short_results"][0]["wall_time_s"] == 2.0
        assert "RUN_LABEL=short-1" not in result.output
        assert "RESULT_JSON=/tmp/short-1/result.json" not in result.output
        assert "[kafka-runner] Progress poll 1:" in result.output
        assert "stderr: short-run warning" in result.output
        assert "RUN_LABEL=full-2" in result.output
        assert "RESULT_STATUS=success" in result.output
        assert "0.90x (slower)" in result.output
        assert "1.30x (better)" in result.output
        assert "Screen Time" in result.output
        assert "2.00s" in result.output
        short_stdout = (run_dir / "runs" / "short-1" / "stdout.txt").read_text(encoding="utf-8")
        assert "RUN_LABEL=short-1" in short_stdout
        assert "RESULT_JSON=/tmp/short-1/result.json" in short_stdout


def test_quick_mode_uses_completed_short_run_as_validated_objective_evidence() -> None:
    runner = CliRunner()

    class _CompletedShortAgent(_DummyQuickAgent):
        async def step(self, prompt: str) -> SimpleNamespace:
            _ = prompt
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "summary": "screen one env candidate",
                        "manual_citation": "rule",
                        "evidence": ["signal"],
                        "actions": [
                            {
                                "config_type": "env_var",
                                "parameter": "OMP_NUM_THREADS",
                                "new_value": "1",
                                "reason": "reduce contention",
                                "manual_citation": "rule",
                            }
                        ],
                    }
                )
            )

    structured_bundle = {
        "profile_first": [
            {
                "signal_ref": "demo.sig",
                "role": "resource_signal",
                "hypothesis_name": "demo-hypothesis",
                "candidate_controls": ["OMP_NUM_THREADS"],
                "signal": {
                    "id": "sig",
                    "metric": "rss_mb",
                    "virtual_probe": "process.memory.rss",
                    "expected_direction_when_mitigated": "decrease",
                    "role": "resource_signal",
                },
            }
        ],
        "hypothesis_profiles": [
            {
                "hypothesis_id": "demo-hypothesis",
                "validation_signals": [
                    {
                        "ref": "demo.sig",
                        "role": "resource_signal",
                        "candidate_controls": ["OMP_NUM_THREADS"],
                        "signal": {
                            "id": "sig",
                            "metric": "rss_mb",
                            "virtual_probe": "process.memory.rss",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "resource_signal",
                        },
                    }
                ],
            }
        ],
    }

    baseline = {
        "pid": 1000,
        "launch_command": ["echo", "hi"],
        "wall_time": 5.0,
        "stdout": "throughput=100\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "profiling_text": "{}",
        "profiling_json": {"periodic_probe_samples": _sample_entries([10, 10, 10, 20, 20, 20, 20, 10, 10, 10])},
        "tool_calls_used": 0,
        "stop_reason": "completed",
        "stopped_intentionally": False,
    }
    completed_short = {
        "pid": 1001,
        "launch_command": ["echo", "hi"],
        "wall_time": 2.0,
        "stdout": "throughput=160\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "profiling_text": "{}",
        "profiling_json": {"periodic_probe_samples": _sample_entries([10])},
        "tool_calls_used": 0,
        "stop_reason": "completed",
        "stopped_intentionally": False,
    }
    artifacts = [baseline, completed_short]

    async def _fake_profile(*args: object, **kwargs: object) -> dict[str, Any]:
        _ = args
        artifact = artifacts.pop(0)
        callback = kwargs.get("live_output_callback")
        if callable(callback):
            for line in str(artifact.get("stdout", "")).splitlines(True):
                callback("stdout", line)
            for line in str(artifact.get("stderr", "")).splitlines(True):
                callback("stderr", line)
        return artifact

    async def _fake_extract_metrics(agent: SysPilotAgent, output: str, context: str) -> dict[str, Any]:
        _ = agent
        _ = context
        for line in output.splitlines():
            if line.startswith("throughput="):
                return {"throughput": float(line.split("=", 1)[1])}
        return {"no_metrics": True}

    with runner.isolated_filesystem():
        with (
            patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
            patch("syspilot.cli.SysPilotAgent", _CompletedShortAgent),
            patch("syspilot.cli._register_default_skills", _noop),
            patch("syspilot.cli._register_profiling_tools", _noop),
            patch("syspilot.cli._register_tuning_tools", _noop),
            patch("syspilot.cli._build_contract_context_bundle", lambda *args, **kwargs: structured_bundle),
            patch("syspilot.cli._run_live_profiling_pipeline", _fake_profile),
            patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
        ):
            result = runner.invoke(
                cli,
                [
                    "tune",
                    "--quick-mode",
                    "--ui-mode",
                    "plain",
                    "--command",
                    "echo hi",
                    "--primary-objective",
                    "throughput",
                    "--primary-objective-mode",
                    "maximize",
                    "--contract",
                    "demo",
                    "--max-iterations",
                    "1",
                    "--timeout",
                    "5",
                ],
            )

        run_dir = Path("artifacts") / "quick_tune_runs" / "quick-test-run"
        report = json.loads((run_dir / "quick_report.json").read_text(encoding="utf-8"))

        assert result.exit_code == 0
        assert report["selected_candidate_id"] == 1
        assert "OMP_NUM_THREADS=1" in report["selected_candidate_label"]
        assert report["final_selection_basis"] == "throughput"
        assert report["conclusion"] == "significant_improvement_found"
        assert report["full_results"] == []
        assert len(report["short_results"]) == 1
        assert report["short_results"][0]["status"] == "unavailable"
        assert report["short_results"][0]["objective_value"] == 160.0
        assert report["short_results"][0]["objective_speedup_ratio"] == 1.6
        assert report["promotions"][0]["reason"] == "already completed full workload during short screen"
        assert "Completed before the quick-screen budget" in result.output
        assert "Quick mode selected" in result.output
        assert "1.60x (better)" in result.output


def test_quick_mode_continues_after_significant_full_improvement() -> None:
    runner = CliRunner()

    class _TwoWindowQuickAgent(_DummyQuickAgent):
        def __init__(self, config: object) -> None:
            super().__init__(config)
            self.window_call_count = 0

        async def step(self, prompt: str) -> SimpleNamespace:
            _ = prompt
            self.window_call_count += 1
            prefix = "RSS" if self.window_call_count == 1 else "QUEUE"
            actions = [
                {
                    "config_type": "env_var",
                    "parameter": f"{prefix}_VAR_{idx}",
                    "new_value": str(idx),
                    "reason": f"{prefix.lower()} candidate {idx}",
                    "manual_citation": "rule",
                }
                for idx in range(1, 4)
            ]
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "summary": f"screen {prefix.lower()} candidates",
                        "manual_citation": "rule",
                        "evidence": [prefix.lower()],
                        "actions": actions,
                    }
                )
            )

    structured_bundle = {
        "profile_first": [
            {
                "signal_ref": "demo.rss",
                "role": "resource_signal",
                "hypothesis_name": "rss-hypothesis",
                "candidate_controls": ["RSS_VAR_1"],
                "signal": {
                    "id": "rss",
                    "metric": "rss_mb",
                    "virtual_probe": "process.memory.rss",
                    "expected_direction_when_mitigated": "decrease",
                    "role": "resource_signal",
                },
            },
            {
                "signal_ref": "demo.queue",
                "role": "resource_signal",
                "hypothesis_name": "queue-hypothesis",
                "candidate_controls": ["QUEUE_VAR_1"],
                "signal": {
                    "id": "queue",
                    "metric": "queue_depth",
                    "virtual_probe": "queue.depth",
                    "expected_direction_when_mitigated": "decrease",
                    "role": "resource_signal",
                },
            },
        ],
        "hypothesis_profiles": [
            {
                "hypothesis_id": "rss-hypothesis",
                "validation_signals": [
                    {
                        "ref": "demo.rss",
                        "role": "resource_signal",
                        "candidate_controls": ["RSS_VAR_1"],
                        "signal": {
                            "id": "rss",
                            "metric": "rss_mb",
                            "virtual_probe": "process.memory.rss",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "resource_signal",
                        },
                    }
                ],
            },
            {
                "hypothesis_id": "queue-hypothesis",
                "validation_signals": [
                    {
                        "ref": "demo.queue",
                        "role": "resource_signal",
                        "candidate_controls": ["QUEUE_VAR_1"],
                        "signal": {
                            "id": "queue",
                            "metric": "queue_depth",
                            "virtual_probe": "queue.depth",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "resource_signal",
                        },
                    }
                ],
            },
        ],
    }

    baseline = {
        "pid": 1000,
        "launch_command": ["echo", "hi"],
        "wall_time": 5.0,
        "stdout": "throughput=100\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "profiling_text": "{}",
        "profiling_json": {
            "periodic_probe_samples": _multi_probe_sample_entries(
                {
                    "process.memory.rss": [10, 10, 10, 20, 20, 20, 20, 10, 10, 10],
                    "queue.depth": [5, 5, 5, 5, 5, 12, 12, 12, 12, 5],
                }
            )
        },
        "tool_calls_used": 0,
        "stop_reason": "completed",
        "stopped_intentionally": False,
    }

    def _short_artifact(label: str, probe_name: str, values: Sequence[float | int]) -> dict[str, Any]:
        return {
            "pid": 1100,
            "launch_command": ["echo", "hi"],
            "wall_time": 2.0,
            "stdout": f"{label}\n",
            "stderr": "",
            "returncode": -15,
            "timed_out": False,
            "profiling_text": "{}",
            "profiling_json": {"periodic_probe_samples": _sample_entries(values, probe_name=probe_name)},
            "tool_calls_used": 0,
            "stop_reason": "screen_budget_reached",
            "stopped_intentionally": True,
        }

    def _full_artifact(label: str, throughput: float, probe_name: str, values: Sequence[float | int]) -> dict[str, Any]:
        return {
            "pid": 2100,
            "launch_command": ["echo", "hi"],
            "wall_time": 4.0,
            "stdout": f"throughput={throughput}\n{label}\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "{}",
            "profiling_json": {"periodic_probe_samples": _sample_entries(values, probe_name=probe_name)},
            "tool_calls_used": 0,
            "stop_reason": "completed",
            "stopped_intentionally": False,
        }

    artifacts = [
        baseline,
        _short_artifact("rss-short-1", "process.memory.rss", [10, 10, 10, 16, 16, 16, 16, 10, 10, 10]),
        _short_artifact("rss-short-2", "process.memory.rss", [10, 10, 10, 18, 18, 18, 18, 10, 10, 10]),
        _short_artifact("rss-short-3", "process.memory.rss", [10, 10, 10, 19, 19, 19, 19, 10, 10, 10]),
        _full_artifact("rss-full", 110.0, "process.memory.rss", [10, 10, 10, 18, 18, 18, 18, 10, 10, 10]),
        _short_artifact("queue-short-1", "queue.depth", [5, 5, 5, 5, 5, 10, 10, 10, 10, 5]),
        _short_artifact("queue-short-2", "queue.depth", [5, 5, 5, 5, 5, 8, 8, 8, 8, 5]),
        _short_artifact("queue-short-3", "queue.depth", [5, 5, 5, 5, 5, 7, 7, 7, 7, 5]),
        _full_artifact("queue-full", 130.0, "queue.depth", [5, 5, 5, 5, 5, 8, 8, 8, 8, 5]),
    ]

    async def _fake_profile(*args: object, **kwargs: object) -> dict[str, Any]:
        _ = args
        artifact = artifacts.pop(0)
        callback = kwargs.get("live_output_callback")
        if callable(callback):
            for line in str(artifact.get("stdout", "")).splitlines(True):
                callback("stdout", line)
            for line in str(artifact.get("stderr", "")).splitlines(True):
                callback("stderr", line)
        return artifact

    async def _fake_extract_metrics(agent: SysPilotAgent, output: str, context: str) -> dict[str, Any]:
        _ = agent
        _ = context
        for line in output.splitlines():
            if line.startswith("throughput="):
                return {"throughput": float(line.split("=", 1)[1])}
        return {"no_metrics": True}

    with runner.isolated_filesystem():
        with (
            patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
            patch("syspilot.cli.SysPilotAgent", _TwoWindowQuickAgent),
            patch("syspilot.cli._register_default_skills", _noop),
            patch("syspilot.cli._register_profiling_tools", _noop),
            patch("syspilot.cli._register_tuning_tools", _noop),
            patch("syspilot.cli._build_contract_context_bundle", lambda *args, **kwargs: structured_bundle),
            patch("syspilot.cli._run_live_profiling_pipeline", _fake_profile),
            patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
        ):
            result = runner.invoke(
                cli,
                [
                    "tune",
                    "--quick-mode",
                    "--ui-mode",
                    "plain",
                    "--command",
                    "echo hi",
                    "--primary-objective",
                    "throughput",
                    "--primary-objective-mode",
                    "maximize",
                    "--contract",
                    "demo",
                    "--max-iterations",
                    "3",
                    "--timeout",
                    "5",
                ],
            )

        run_dir = Path("artifacts") / "quick_tune_runs" / "quick-test-run"
        report = json.loads((run_dir / "quick_report.json").read_text(encoding="utf-8"))
        assert result.exit_code == 0
        assert len(report["checked_window_ids"]) == 2
        assert len(report["full_results"]) == 2
        assert [group["signal_name"] for group in report["signal_windows"]] == ["rss_mb", "queue_depth"]
        assert report["full_results"][0]["objective_speedup_ratio"] == 1.1
        assert report["full_results"][1]["objective_speedup_ratio"] == 1.3
        assert "1.30x (better)" in result.output
        assert report["selected_candidate_id"] == 6
        assert "QUEUE_VAR_1=1" in report["selected_candidate_label"]


def test_rank_abnormal_windows_reports_missing_virtual_probe() -> None:
    spec = QuickSignalSpec(
        signal_ref="demo.missing",
        signal_id="missing",
        metric="missing_metric",
        virtual_probe="runtime.missing.probe",
        expected_direction_when_mitigated="decrease",
        role="resource_signal",
    )
    baseline = {
        "profiling_json": {
            "periodic_probe_samples": _sample_entries([1, 2, 3, 4], probe_name="process.memory.rss")
        }
    }

    windows, usefulness, profiling_available = rank_abnormal_windows(
        [spec],
        baseline,
        QuickModeDefaults(),
    )

    assert windows == []
    assert profiling_available is False
    assert usefulness[0].status == "unavailable"
    assert "runtime.missing.probe" in usefulness[0].notes
    assert "toolbox/probes" in usefulness[0].notes


def test_quick_mode_preserves_report_when_recommendation_provider_fails() -> None:
    runner = CliRunner()

    class _FailingRecommendationAgent(_DummyQuickAgent):
        async def step(self, prompt: str) -> SimpleNamespace:
            _ = prompt
            raise LLMProviderError("empty_stream: upstream stream closed before first payload")

    structured_bundle = {
        "profile_first": [
            {
                "signal_ref": "demo.sig",
                "role": "resource_signal",
                "hypothesis_name": "demo-hypothesis",
                "candidate_controls": ["OMP_PROC_BIND"],
                "signal": {
                    "id": "sig",
                    "metric": "rss_mb",
                    "virtual_probe": "process.memory.rss",
                    "expected_direction_when_mitigated": "decrease",
                    "role": "resource_signal",
                },
            }
        ],
        "hypothesis_profiles": [
            {
                "hypothesis_id": "demo-hypothesis",
                "validation_signals": [
                    {
                        "ref": "demo.sig",
                        "role": "resource_signal",
                        "candidate_controls": ["OMP_PROC_BIND"],
                        "signal": {
                            "id": "sig",
                            "metric": "rss_mb",
                            "virtual_probe": "process.memory.rss",
                            "expected_direction_when_mitigated": "decrease",
                            "role": "resource_signal",
                        },
                    }
                ],
            }
        ],
    }
    baseline = {
        "pid": 1000,
        "launch_command": ["echo", "hi"],
        "wall_time": 5.0,
        "stdout": "baseline\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "profiling_text": "{}",
        "profiling_json": {"periodic_probe_samples": _sample_entries([10, 10, 10, 20, 20, 20, 20, 10, 10, 10])},
        "tool_calls_used": 0,
        "stop_reason": "completed",
        "stopped_intentionally": False,
    }

    async def _fake_profile(*args: object, **kwargs: object) -> dict[str, Any]:
        _ = args
        _ = kwargs
        return baseline

    with runner.isolated_filesystem():
        with (
            patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
            patch("syspilot.cli.SysPilotAgent", _FailingRecommendationAgent),
            patch("syspilot.cli._register_default_skills", _noop),
            patch("syspilot.cli._register_profiling_tools", _noop),
            patch("syspilot.cli._register_tuning_tools", _noop),
            patch("syspilot.cli._build_contract_context_bundle", lambda *args, **kwargs: structured_bundle),
            patch("syspilot.cli._run_live_profiling_pipeline", _fake_profile),
        ):
            result = runner.invoke(
                cli,
                [
                    "tune",
                    "--quick-mode",
                    "--ui-mode",
                    "plain",
                    "--command",
                    "echo hi",
                    "--contract",
                    "demo",
                    "--max-iterations",
                    "2",
                    "--timeout",
                    "5",
                ],
            )

        run_dir = Path("artifacts") / "quick_tune_runs" / "quick-test-run"
        report = json.loads((run_dir / "quick_report.json").read_text(encoding="utf-8"))
        assert result.exit_code == 0
        assert report["conclusion"] == "llm_provider_error"
        assert report["checked_window_ids"] == ["sig-w2"]
        assert report["short_results"] == []
        assert report["degraded_errors"][0]["phase"] == "recommendation"
        assert "empty_stream" in report["degraded_errors"][0]["message"]
        assert (run_dir / "degraded_errors.jsonl").exists()
