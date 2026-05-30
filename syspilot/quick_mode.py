from __future__ import annotations

import asyncio
import csv
import json
import math
import shlex
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.markup import escape as rich_escape

from syspilot.core.agent import SysPilotAgent
from syspilot.core.llm import LLMError
from syspilot.fixed_budget import FixedBudgetState
from syspilot.fixed_goal import fixed_goal_payload, fixed_goal_reached
from syspilot.prompt_budget import (
    hypothesis_profiles_yaml_for_prompt,
    profiling_json_for_prompt,
)
from syspilot.ui.output_sink import OutputSink, PlainOutputSink, TuneUIMode, make_output_sink


async def _agent_step_for_prompt(agent: Any, prompt: str) -> Any:
    stateless_step = getattr(agent, "step_stateless", None)
    if callable(stateless_step):
        typed_step = cast(Callable[..., Awaitable[Any]], stateless_step)
        return await typed_step(prompt, include_tools=False)
    return await agent.step(prompt)


@dataclass(frozen=True)
class QuickModeDefaults:
    sample_interval_sec: float = 5.0
    short_measurement_sec: float = 30.0
    full_run_ratio: float = 0.20
    max_full_runs_per_group: int = 3
    min_signal_samples: int = 4
    neutral_change_pct: float = 2.0
    significant_improvement_pct: float = 5.0
    max_windows_per_signal: int = 2
    max_signal_groups: int = 6
    max_total_short_runs: int = 12
    max_total_full_runs: int = 6

    def __post_init__(self) -> None:
        if self.short_measurement_sec < (3.0 * self.sample_interval_sec):
            raise ValueError("short_measurement_sec must be at least 3 * sample_interval_sec")

    @property
    def periodic_sample_interval_s(self) -> float:
        return self.sample_interval_sec

    @property
    def short_measurement_duration_s(self) -> float:
        return self.short_measurement_sec

    @property
    def promotion_fraction(self) -> float:
        return self.full_run_ratio

    @property
    def max_promoted_candidates(self) -> int:
        return self.max_full_runs_per_group

    @property
    def window_samples(self) -> int:
        return max(3, int(round(self.short_measurement_sec / self.sample_interval_sec)))


@dataclass
class QuickSignalSpec:
    signal_ref: str
    signal_id: str
    metric: str
    virtual_probe: str
    expected_direction_when_mitigated: str
    role: str
    hypothesis_name: str = ""
    candidate_controls: list[str] = field(default_factory=list)
    status: str = "checking"
    notes: str = ""


@dataclass
class SampledSignalPoint:
    time_offset_s: float
    value: float
    timestamp: str = ""
    unit: str | None = None
    probe_status: str = "observed"


@dataclass
class QuickBaselineWindow:
    window_id: str
    signal_name: str
    signal_ref: str
    hypothesis_name: str
    start_offset_s: float
    end_offset_s: float
    duration_s: float
    sample_count: int
    expected_direction: str
    severity: float
    abnormality_score: float
    candidate_controls: list[str]
    window_summary: dict[str, float]
    global_summary: dict[str, float]
    reason: str


@dataclass
class QuickRunMetadata:
    run_id: str
    config_id: str
    run_kind: str
    candidate_id: int | None
    candidate_label: str
    signal_ref: str
    signal_name: str
    window_id: str
    command: list[str]
    env_overrides: dict[str, str]
    started_at: str
    finished_at: str
    wall_time_s: float
    returncode: int
    stop_reason: str
    stopped_intentionally: bool
    artifact_path: str
    stdout_path: str
    stderr_path: str


@dataclass
class QuickShortResult:
    candidate_id: int
    candidate_label: str
    config_id: str
    signal_ref: str
    signal_name: str
    window_id: str
    command: list[str]
    env_overrides: dict[str, str]
    run_kind: str
    wall_time_s: float
    returncode: int
    stop_reason: str
    stopped_intentionally: bool
    target_signal_mean: float | None
    target_signal_improvement_pct: float | None
    status: str
    objective_metric: str = ""
    objective_mode: str = ""
    objective_value: float | None = None
    objective_improvement_pct: float | None = None
    objective_improved: bool | None = None
    objective_speedup_ratio: float | None = None
    objective_speedup_label: str = ""
    metadata: QuickRunMetadata | None = None
    artifact: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuickPromotionDecision:
    candidate_id: int
    candidate_label: str
    config_id: str
    signal_ref: str
    signal_name: str
    window_id: str
    promoted: bool
    rank: int | None
    reason: str


@dataclass
class QuickSignalUsefulness:
    signal_ref: str
    signal_name: str
    status: str
    notes: str


@dataclass
class QuickSignalWindowRef:
    window_id: str
    start_offset_s: float
    end_offset_s: float
    severity: float
    checked: bool


@dataclass
class QuickSignalWindowGroup:
    signal_ref: str
    signal_name: str
    windows: list[QuickSignalWindowRef]


@dataclass
class QuickBaselineSummary:
    baseline_run: QuickRunMetadata
    objective_metric: str
    objective_mode: str
    objective_value: float | None
    sample_row_count: int
    useful_signal_count: int
    abnormal_window_count: int


@dataclass
class QuickModeReport:
    schema_version: str
    mode: str
    generated_at: str
    run_id: str
    primary_objective_query: str
    primary_objective_metric: str
    primary_objective_mode: str
    baseline_summary: QuickBaselineSummary
    abnormal_windows: list[QuickBaselineWindow]
    signal_windows: list[QuickSignalWindowGroup]
    short_results: list[QuickShortResult]
    promotions: list[QuickPromotionDecision]
    full_results: list[QuickShortResult]
    signal_usefulness: list[QuickSignalUsefulness]
    final_selection_basis: str
    selected_candidate_id: int | None
    selected_candidate_label: str
    conclusion: str
    conclusion_reason: str
    checked_window_ids: list[str]
    artifact_dir: str
    degraded_errors: list[dict[str, str]] = field(default_factory=list)


RunExecutor = Callable[..., Awaitable[dict[str, Any]]]
ParseActions = Callable[..., dict[str, Any]]
BuildCandidates = Callable[..., list[dict[str, Any]]]
ExtractHypothesisProfiles = Callable[
    [dict[str, object] | None, list[str] | None], list[dict[str, Any]]
]
RecommendationActionDisplay = Callable[[list[dict[str, Any]]], str]
ExtractInlineEnvAssignments = Callable[[list[str]], tuple[list[str], dict[str, str]]]
ExtractMetricsFromOutput = Callable[[SysPilotAgent, str, str], Awaitable[dict[str, Any]]]
ResolvePrimaryObjectiveMetric = Callable[[dict[str, Any], str | None], str | None]
ExtractPrimaryObjectiveValue = Callable[[dict[str, Any], str | None], float | None]
InferPrimaryObjectiveMode = Callable[[str], str | None]
CompareObjectiveValues = Callable[[float, float, str], int]
ComputeObjectiveChangePct = Callable[[float, float, str], float]
ObjectiveDisplayName = Callable[[str | None], str]
FormatMetricValue = Callable[[float | None], str]
TuneEventEmitterProtocol = Any


def _fixed_budget_payload(
    fixed_budget_state: FixedBudgetState,
    *,
    stop_reason: str = "",
) -> dict[str, object]:
    if not fixed_budget_state.enabled:
        return {}
    payload = fixed_budget_state.to_payload()
    return {
        "fixed_budget": payload,
        "fixed_budget_total": payload["total"],
        "fixed_budget_used": payload["used"],
        "fixed_budget_remaining": payload["remaining"],
        "fixed_budget_stop_reason": stop_reason,
        "no_action_reprompts_used": payload["no_action_reprompts_used"],
    }


def _render_excluded_config_names(names: Sequence[str]) -> str:
    if not names:
        return "(none)"
    joined = ", ".join(names)
    return (
        f"Keep exactly these controls unchanged: {joined}. Only these exact listed "
        "environment variables, command-line parameters, or config controls are immutable; "
        "related controls with different names remain tunable when supported by evidence."
    )


def _render_tunable_whitelist_names(names: Sequence[str]) -> str:
    if not names:
        return (
            "A tunable whitelist file was provided, but it contained no controls. "
            "Return no tuning actions."
        )
    joined = ", ".join(names)
    return (
        f"Change only these exact tunable controls: {joined}. Environment variables, "
        "command-line parameters, command-prefix parameters, or config controls not listed here "
        "must remain unchanged even when evidence suggests they may help."
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _append_csv_row(path: Path, row: dict[str, Any], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in headers})


def _event_output_payload(stdout: str, stderr: str, max_chars: int = 8000) -> dict[str, Any]:
    stdout_text = stdout or ""
    stderr_text = stderr or ""
    return {
        "stdout": stdout_text[:max_chars],
        "stderr": stderr_text[:max_chars],
        "stdout_len": len(stdout_text),
        "stderr_len": len(stderr_text),
        "stdout_truncated": len(stdout_text) > max_chars,
        "stderr_truncated": len(stderr_text) > max_chars,
    }


def _emit_tune_event(
    event_emitter: TuneEventEmitterProtocol | None,
    event_name: str,
    payload: dict[str, object],
) -> None:
    if event_emitter is None:
        return
    try:
        _ = event_emitter.publish(event_name, payload)
    except Exception:
        return


def _record_degraded_error(
    degraded_errors: list[dict[str, str]],
    *,
    phase: str,
    message: str,
    window_id: str = "",
    signal_ref: str = "",
    error_type: str = "",
) -> None:
    degraded_errors.append(
        {
            "phase": phase,
            "window_id": window_id,
            "signal_ref": signal_ref,
            "error_type": error_type,
            "message": message,
        }
    )


def _is_llm_degraded_error(entry: dict[str, str]) -> bool:
    error_text = f"{entry.get('error_type', '')} {entry.get('message', '')}".lower()
    return any(
        marker in error_text for marker in ("llm", "provider", "openai", "stream", "timeout")
    )


def _render_spawn_manual_hints(manuals: Sequence[str]) -> str:
    if not manuals:
        return ""
    return "\n".join(f"- {manual}" for manual in manuals)


def _render_loaded_manual_context(loaded_manuals: dict[str, str]) -> str:
    if not loaded_manuals:
        return ""
    parts: list[str] = []
    for manual_name, manual_content in loaded_manuals.items():
        parts.append(f"### Manual: {manual_name}\n{manual_content.strip()}")
    return "\n\n".join(parts)


_SHORT_RUN_WRAPPER_FOOTER_PREFIXES = (
    "[kafka-runner] Timing summary:",
    "RUN_LABEL=",
    "RUN_DIR=",
    "RESULT_JSON=",
    "RESULT_STATUS=",
    "ELAPSED_SEC=",
    "TASK_COMPLETIONS_PER_SEC=",
    "OBSERVED_TOTAL_COMPLETES=",
    "EXPECTED_TOTAL_COMPLETIONS=",
    "PEAK_RSS_KB=",
)


def _sink_uses_plain_live_output_filter(sink: OutputSink) -> bool:
    if isinstance(sink, PlainOutputSink):
        return True
    is_tui_active = getattr(sink, "is_tui_active", None)
    if callable(is_tui_active):
        try:
            return not bool(is_tui_active())
        except Exception:
            return False
    return False


def _should_suppress_short_run_live_stdout_line(
    sink: OutputSink,
    run_kind: str,
    stream_name: str,
    text: str,
) -> bool:
    if run_kind != "short" or stream_name != "stdout":
        return False
    if not _sink_uses_plain_live_output_filter(sink):
        return False
    return any(text.startswith(prefix) for prefix in _SHORT_RUN_WRAPPER_FOOTER_PREFIXES)


def _plain_live_output_callback(
    sink: OutputSink,
    stream_name: str,
    line: str,
    *,
    run_kind: str,
) -> None:
    text = line.rstrip("\r\n")
    if not text:
        return
    if _should_suppress_short_run_live_stdout_line(sink, run_kind, stream_name, text):
        return
    escaped = rich_escape(text)
    if stream_name == "stderr":
        sink.print(f"  [dim]stderr:[/dim] {escaped}")
        return
    sink.print(f"  {escaped}")


def _normalize_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _signal_semantic_key(metric: str, virtual_probe: str, direction: str) -> tuple[str, str, str]:
    return (metric, virtual_probe, direction)


def _extract_signal_specs(
    structured_context_bundle: dict[str, object] | None,
    hypothesis_profiles: list[dict[str, Any]],
) -> list[QuickSignalSpec]:
    specs: list[QuickSignalSpec] = []
    semantic_index: dict[tuple[str, str, str], int] = {}

    def _add(
        ref: str,
        signal: dict[str, Any],
        role: str,
        hypothesis_name: str,
        candidate_controls: list[str],
    ) -> None:
        direction = str(signal.get("expected_direction_when_mitigated", "")).strip()
        virtual_probe = str(signal.get("virtual_probe", "")).strip()
        metric = str(signal.get("metric", "")).strip()
        signal_id = str(signal.get("id", ref.rsplit(".", 1)[-1])).strip()
        if not ref:
            return
        if direction not in {"increase", "decrease"}:
            return
        if not virtual_probe or not metric:
            return
        if role == "outcome":
            return
        semantic_key = _signal_semantic_key(metric, virtual_probe, direction)
        normalized_controls = _string_list(candidate_controls)
        existing_index = semantic_index.get(semantic_key)
        if existing_index is not None:
            existing = specs[existing_index]
            existing.candidate_controls = list(
                dict.fromkeys([*existing.candidate_controls, *normalized_controls])
            )
            if not existing.hypothesis_name and hypothesis_name:
                existing.hypothesis_name = hypothesis_name
            return
        specs.append(
            QuickSignalSpec(
                signal_ref=ref,
                signal_id=signal_id,
                metric=metric,
                virtual_probe=virtual_probe,
                expected_direction_when_mitigated=direction,
                role=role or "resource_signal",
                hypothesis_name=hypothesis_name,
                candidate_controls=normalized_controls,
            )
        )
        semantic_index[semantic_key] = len(specs) - 1

    if isinstance(structured_context_bundle, dict):
        raw_profile_first = structured_context_bundle.get("profile_first", [])
        if isinstance(raw_profile_first, list):
            for entry in raw_profile_first:
                if not isinstance(entry, dict):
                    continue
                signal = entry.get("signal")
                if not isinstance(signal, dict):
                    continue
                _add(
                    str(entry.get("signal_ref", "")).strip(),
                    signal,
                    str(entry.get("role", signal.get("role", "resource_signal"))).strip(),
                    str(entry.get("hypothesis_name", entry.get("hypothesis_id", ""))).strip(),
                    _string_list(entry.get("candidate_controls"))
                    or _string_list(signal.get("candidate_controls")),
                )

    for profile in hypothesis_profiles:
        raw_signals = profile.get("validation_signals", [])
        if not isinstance(raw_signals, list):
            continue
        hypothesis_name = str(
            profile.get("hypothesis_name", profile.get("hypothesis_id", ""))
        ).strip()
        for entry in raw_signals:
            if not isinstance(entry, dict):
                continue
            signal = entry.get("signal")
            if not isinstance(signal, dict):
                continue
            _add(
                str(entry.get("ref", "")).strip(),
                signal,
                str(entry.get("role", signal.get("role", "resource_signal"))).strip(),
                hypothesis_name,
                _string_list(entry.get("candidate_controls"))
                or _string_list(signal.get("candidate_controls")),
            )
    return specs


def _extract_probe_value(probe_result: dict[str, Any]) -> tuple[float | None, str | None, str]:
    status = "observed" if probe_result.get("observed", True) else "unavailable"
    unit = probe_result.get("unit") if isinstance(probe_result.get("unit"), str) else None
    value = _normalize_numeric(probe_result.get("value"))
    if value is not None:
        return value, unit, status
    for container_name in ("flat", "structured"):
        container = probe_result.get(container_name)
        if isinstance(container, dict):
            for key, candidate in container.items():
                value = _normalize_numeric(candidate)
                if value is not None:
                    inferred_unit = "bytes" if "bytes" in str(key).lower() else unit
                    return value, inferred_unit, status
    return None, unit, status


def _extract_signal_points(
    artifact: dict[str, Any], spec: QuickSignalSpec
) -> list[SampledSignalPoint]:
    profiling_json = artifact.get("profiling_json", {})
    if not isinstance(profiling_json, dict):
        return []
    samples = profiling_json.get("periodic_probe_samples", [])
    if not isinstance(samples, list):
        return []

    points: list[SampledSignalPoint] = []
    for entry in samples:
        if not isinstance(entry, dict):
            continue
        raw_time = _normalize_numeric(entry.get("time_offset_s"))
        results = entry.get("results", {})
        if raw_time is None or not isinstance(results, dict):
            continue
        probe_result = results.get(spec.virtual_probe, {})
        if not isinstance(probe_result, dict):
            continue
        value, unit, probe_status = _extract_probe_value(probe_result)
        if value is None:
            continue
        points.append(
            SampledSignalPoint(
                time_offset_s=raw_time,
                value=value,
                timestamp=str(entry.get("timestamp", "")),
                unit=unit,
                probe_status=probe_status,
            )
        )
    return points


def _signal_sample_diagnostics(
    artifact: dict[str, Any],
    spec: QuickSignalSpec,
) -> tuple[list[SampledSignalPoint], bool, str]:
    profiling_json = artifact.get("profiling_json", {})
    if not isinstance(profiling_json, dict):
        return [], False, "baseline profiling JSON was missing or malformed"
    samples = profiling_json.get("periodic_probe_samples", [])
    if not isinstance(samples, list) or not samples:
        return [], False, "baseline captured no periodic probe samples"

    points: list[SampledSignalPoint] = []
    probe_seen_count = 0
    unavailable_count = 0
    nonnumeric_count = 0
    missing_result_count = 0
    last_unavailable_detail = ""

    for entry in samples:
        if not isinstance(entry, dict):
            missing_result_count += 1
            continue
        raw_time = _normalize_numeric(entry.get("time_offset_s"))
        results = entry.get("results", {})
        if raw_time is None or not isinstance(results, dict):
            missing_result_count += 1
            continue
        probe_result = results.get(spec.virtual_probe)
        if not isinstance(probe_result, dict):
            continue
        probe_seen_count += 1
        value, unit, probe_status = _extract_probe_value(probe_result)
        if probe_status != "observed":
            unavailable_count += 1
            raw_detail = (
                probe_result.get("error")
                or probe_result.get("message")
                or probe_result.get("reason")
            )
            if raw_detail and not last_unavailable_detail:
                last_unavailable_detail = str(raw_detail)
        if value is None:
            nonnumeric_count += 1
            continue
        points.append(
            SampledSignalPoint(
                time_offset_s=raw_time,
                value=value,
                timestamp=str(entry.get("timestamp", "")),
                unit=unit,
                probe_status=probe_status,
            )
        )

    if points:
        return (
            points,
            True,
            f"captured {len(points)} numeric sample(s) for virtual probe {spec.virtual_probe}",
        )
    if probe_seen_count == 0:
        return (
            [],
            False,
            (
                f"periodic samples did not include virtual probe {spec.virtual_probe!r}; "
                "check the LPCL signal virtual_probe and toolbox/probes script mapping"
            ),
        )
    if unavailable_count:
        detail = f": {last_unavailable_detail}" if last_unavailable_detail else ""
        return (
            [],
            True,
            (
                f"virtual probe {spec.virtual_probe!r} reported unavailable in "
                f"{unavailable_count}/{probe_seen_count} sample(s){detail}"
            ),
        )
    if nonnumeric_count:
        return (
            [],
            True,
            (
                f"virtual probe {spec.virtual_probe!r} produced {nonnumeric_count} "
                "nonnumeric sample result(s)"
            ),
        )
    return (
        [],
        bool(missing_result_count),
        f"virtual probe {spec.virtual_probe!r} produced no usable sample rows",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _directional_delta_sum(values: Sequence[float], direction: str) -> float:
    total = 0.0
    for idx in range(1, len(values)):
        if direction == "increase":
            total += max(values[idx - 1] - values[idx], 0.0)
        else:
            total += max(values[idx] - values[idx - 1], 0.0)
    return total


def _select_non_overlapping_windows(
    candidates: list[QuickBaselineWindow],
    limit: int,
) -> list[QuickBaselineWindow]:
    selected: list[QuickBaselineWindow] = []
    for candidate in candidates:
        overlaps = any(
            not (
                candidate.end_offset_s < item.start_offset_s
                or candidate.start_offset_s > item.end_offset_s
            )
            for item in selected
        )
        if overlaps:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _interleave_windows_by_signal(windows: list[QuickBaselineWindow]) -> list[QuickBaselineWindow]:
    buckets: dict[str, list[QuickBaselineWindow]] = {}
    signal_order: list[str] = []
    for window in windows:
        if window.signal_ref not in buckets:
            buckets[window.signal_ref] = []
            signal_order.append(window.signal_ref)
        buckets[window.signal_ref].append(window)

    interleaved: list[QuickBaselineWindow] = []
    index = 0
    while True:
        emitted = False
        for signal_ref in signal_order:
            bucket = buckets[signal_ref]
            if index >= len(bucket):
                continue
            interleaved.append(bucket[index])
            emitted = True
        if not emitted:
            break
        index += 1
    return interleaved


def _rank_windows_for_signal(
    spec: QuickSignalSpec,
    points: list[SampledSignalPoint],
    window_samples: int,
    max_windows: int,
) -> list[QuickBaselineWindow]:
    if len(points) < window_samples or window_samples < 2:
        return []

    values = [point.value for point in points]
    global_min = min(values)
    global_max = max(values)
    global_mean = _mean(values)
    value_range = max(global_max - global_min, 1e-9)
    global_directional_delta_sum = _directional_delta_sum(
        values, spec.expected_direction_when_mitigated
    )
    candidates: list[QuickBaselineWindow] = []

    for idx in range(0, len(points) - window_samples + 1):
        window = points[idx : idx + window_samples]
        window_values = [point.value for point in window]
        window_mean = _mean(window_values)
        window_min = min(window_values)
        window_max = max(window_values)
        window_summary = {
            "mean": window_mean,
            "min": window_min,
            "max": window_max,
            "delta_from_global_mean": window_mean - global_mean,
        }
        global_summary = {
            "mean": global_mean,
            "min": global_min,
            "max": global_max,
            "range": value_range,
            "directional_delta_sum": global_directional_delta_sum,
        }
        directional_delta_sum = _directional_delta_sum(
            window_values,
            spec.expected_direction_when_mitigated,
        )
        window_summary["directional_delta_sum"] = directional_delta_sum
        if spec.expected_direction_when_mitigated == "decrease":
            abnormality = window_mean - global_mean
            severity = max(abnormality / value_range, 0.0) * 100.0
            reason = (
                f"window directional burst {directional_delta_sum:.4f} with mean "
                f"{window_mean:.4f} is above "
                f"global mean {global_mean:.4f}; burst windows outrank later plateaus when present"
            )
        else:
            abnormality = global_mean - window_mean
            severity = max(abnormality / value_range, 0.0) * 100.0
            reason = (
                f"window directional burst {directional_delta_sum:.4f} with mean "
                f"{window_mean:.4f} is below "
                f"global mean {global_mean:.4f}; burst windows outrank later plateaus when present"
            )
        candidates.append(
            QuickBaselineWindow(
                window_id=f"{spec.signal_id}-w{idx + 1}",
                signal_name=spec.metric,
                signal_ref=spec.signal_ref,
                hypothesis_name=spec.hypothesis_name,
                start_offset_s=window[0].time_offset_s,
                end_offset_s=window[-1].time_offset_s,
                duration_s=max(window[-1].time_offset_s - window[0].time_offset_s, 0.0),
                sample_count=len(window),
                expected_direction=spec.expected_direction_when_mitigated,
                severity=severity,
                abnormality_score=max(abnormality, 0.0),
                candidate_controls=list(spec.candidate_controls),
                window_summary=window_summary,
                global_summary=global_summary,
                reason=reason,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item.window_summary.get("directional_delta_sum", 0.0)),
            item.severity,
            item.abnormality_score,
            -item.start_offset_s,
        ),
        reverse=True,
    )
    return _select_non_overlapping_windows(ordered, max_windows)


def detect_baseline_window(
    signal_ref: str,
    points: list[SampledSignalPoint],
    direction: str,
    window_samples: int,
) -> QuickBaselineWindow | None:
    spec = QuickSignalSpec(
        signal_ref=signal_ref,
        signal_id=signal_ref.rsplit(".", 1)[-1],
        metric=signal_ref,
        virtual_probe=signal_ref,
        expected_direction_when_mitigated=direction,
        role="resource_signal",
    )
    ranked = _rank_windows_for_signal(spec, points, window_samples, 1)
    return ranked[0] if ranked else None


def rank_abnormal_windows(
    signal_specs: list[QuickSignalSpec],
    baseline_artifact: dict[str, Any],
    defaults: QuickModeDefaults,
) -> tuple[list[QuickBaselineWindow], list[QuickSignalUsefulness], bool]:
    ranked_windows: list[QuickBaselineWindow] = []
    usefulness: list[QuickSignalUsefulness] = []
    profiling_available = False

    for spec in signal_specs:
        points, signal_profile_available, diagnostic_note = _signal_sample_diagnostics(
            baseline_artifact,
            spec,
        )
        if signal_profile_available:
            profiling_available = True
        if len(points) < defaults.min_signal_samples:
            if points:
                diagnostic_note = (
                    f"only {len(points)} numeric sample(s) for virtual probe "
                    f"{spec.virtual_probe!r}; need at least {defaults.min_signal_samples} "
                    "(increase workload duration or lower quick sample interval)"
                )
            usefulness.append(
                QuickSignalUsefulness(
                    signal_ref=spec.signal_ref,
                    signal_name=spec.metric,
                    status="unavailable",
                    notes=diagnostic_note,
                )
            )
            continue
        values = [point.value for point in points]
        if max(values) - min(values) <= 1e-9:
            usefulness.append(
                QuickSignalUsefulness(
                    signal_ref=spec.signal_ref,
                    signal_name=spec.metric,
                    status="uninformative",
                    notes="baseline signal stayed flat",
                )
            )
            continue
        signal_windows = _rank_windows_for_signal(
            spec,
            points,
            defaults.window_samples,
            defaults.max_windows_per_signal,
        )
        if not signal_windows:
            usefulness.append(
                QuickSignalUsefulness(
                    signal_ref=spec.signal_ref,
                    signal_name=spec.metric,
                    status="unavailable",
                    notes="could not form a ranked baseline screening window",
                )
            )
            continue
        ranked_windows.extend(signal_windows)
        usefulness.append(
            QuickSignalUsefulness(
                signal_ref=spec.signal_ref,
                signal_name=spec.metric,
                status="checking",
                notes=f"ranked {len(signal_windows)} abnormal window(s) for quick screening",
            )
        )

    ranked_windows.sort(key=lambda item: (item.severity, item.abnormality_score), reverse=True)
    return _interleave_windows_by_signal(ranked_windows), usefulness, profiling_available


def _window_points_in_range(
    points: list[SampledSignalPoint],
    start_offset_s: float,
    end_offset_s: float,
) -> list[SampledSignalPoint]:
    return [
        point
        for point in points
        if point.time_offset_s + 1e-9 >= start_offset_s
        and point.time_offset_s - 1e-9 <= end_offset_s
    ]


def _signal_improvement_pct(
    baseline_value: float,
    candidate_value: float,
    direction: str,
) -> float:
    if baseline_value == 0:
        return 0.0
    if direction == "increase":
        return ((candidate_value - baseline_value) / abs(baseline_value)) * 100.0
    return ((baseline_value - candidate_value) / abs(baseline_value)) * 100.0


def _objective_speedup_ratio(
    baseline_value: float | None,
    candidate_value: float | None,
    objective_mode: str,
) -> float | None:
    if baseline_value is None or candidate_value is None:
        return None
    if baseline_value <= 0.0 or candidate_value <= 0.0:
        return None
    if objective_mode == "maximize":
        return candidate_value / baseline_value
    return baseline_value / candidate_value


def _objective_speedup_label(ratio: float | None) -> str:
    if ratio is None:
        return ""
    if math.isclose(ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return "same"
    return "better" if ratio > 1.0 else "slower"


def _format_objective_speedup(ratio: float | None) -> str:
    if ratio is None:
        return "ratio unavailable"
    return f"{ratio:.2f}x ({_objective_speedup_label(ratio)})"


def _materialize_result(
    *,
    candidate_id: int,
    candidate_label: str,
    config_id: str,
    signal_ref: str,
    signal_name: str,
    window_id: str,
    command: list[str],
    env_overrides: dict[str, str],
    run_kind: str,
    artifact: dict[str, Any],
    metadata: QuickRunMetadata,
    target_points: list[SampledSignalPoint],
    baseline_window: QuickBaselineWindow,
    defaults: QuickModeDefaults,
) -> QuickShortResult:
    returncode = int(artifact.get("returncode", 0))
    stop_reason = str(artifact.get("stop_reason", "completed"))
    stopped_intentionally = bool(artifact.get("stopped_intentionally", False))
    window_points = _window_points_in_range(
        target_points,
        baseline_window.start_offset_s,
        baseline_window.end_offset_s,
    )
    mean_value: float | None = None
    improvement_pct: float | None = None
    status = "screened"
    if returncode != 0 and not stopped_intentionally:
        status = "failed"
    elif len(window_points) >= max(2, defaults.window_samples // 2):
        mean_value = _mean([point.value for point in window_points])
        improvement_pct = _signal_improvement_pct(
            baseline_window.window_summary["mean"],
            mean_value,
            baseline_window.expected_direction,
        )
        if abs(improvement_pct) < defaults.neutral_change_pct:
            status = "neutral"
    else:
        status = "unavailable"

    return QuickShortResult(
        candidate_id=candidate_id,
        candidate_label=candidate_label,
        config_id=config_id,
        signal_ref=signal_ref,
        signal_name=signal_name,
        window_id=window_id,
        command=command,
        env_overrides=env_overrides,
        run_kind=run_kind,
        wall_time_s=float(artifact.get("wall_time", 0.0)),
        returncode=returncode,
        stop_reason=stop_reason,
        stopped_intentionally=stopped_intentionally,
        target_signal_mean=mean_value,
        target_signal_improvement_pct=improvement_pct,
        status=status,
        metadata=metadata,
        artifact=artifact,
    )


def _result_completed_naturally(result: QuickShortResult) -> bool:
    return (
        result.returncode == 0
        and not result.stopped_intentionally
        and result.stop_reason == "completed"
    )


def _result_is_auto_validated(result: QuickShortResult) -> bool:
    return _result_completed_naturally(result) and result.objective_value is not None


async def _resolve_full_run_objective(
    *,
    agent: SysPilotAgent,
    artifact: dict[str, Any],
    primary_objective_metric: str,
    primary_objective_mode: str,
    extract_metrics_from_output: ExtractMetricsFromOutput,
    extract_primary_objective_value: ExtractPrimaryObjectiveValue,
    compute_objective_change_pct: ComputeObjectiveChangePct,
    baseline_objective_value: float,
    context: str,
) -> tuple[dict[str, Any], float | None, float | None, bool | None]:
    metrics = await extract_metrics_from_output(
        agent,
        str(artifact.get("stdout", "")),
        context,
    )
    objective_value = extract_primary_objective_value(metrics, primary_objective_metric)
    if objective_value is None:
        return metrics, None, None, None
    objective_improvement_pct = compute_objective_change_pct(
        baseline_objective_value,
        objective_value,
        primary_objective_mode,
    )
    objective_improved = objective_improvement_pct > 0.0
    return metrics, objective_value, objective_improvement_pct, objective_improved


async def _populate_result_objective(
    *,
    result: QuickShortResult,
    agent: SysPilotAgent,
    artifact: dict[str, Any],
    primary_objective: str | None,
    baseline_objective_metric: str,
    baseline_objective_mode: str,
    extract_metrics_from_output: ExtractMetricsFromOutput,
    extract_primary_objective_value: ExtractPrimaryObjectiveValue,
    compute_objective_change_pct: ComputeObjectiveChangePct,
    baseline_objective_value: float | None,
    baseline_wall_time: float,
    context: str,
    degraded_errors: list[dict[str, str]] | None = None,
    error_phase: str = "objective_extraction",
) -> None:
    if primary_objective and baseline_objective_value is not None:
        result.objective_metric = baseline_objective_metric
        result.objective_mode = baseline_objective_mode
        try:
            (
                _metrics,
                objective_value,
                objective_improvement_pct,
                objective_improved,
            ) = await _resolve_full_run_objective(
                agent=agent,
                artifact=artifact,
                primary_objective_metric=baseline_objective_metric,
                primary_objective_mode=baseline_objective_mode,
                extract_metrics_from_output=extract_metrics_from_output,
                extract_primary_objective_value=extract_primary_objective_value,
                compute_objective_change_pct=compute_objective_change_pct,
                baseline_objective_value=baseline_objective_value,
                context=context,
            )
        except LLMError as exc:
            if degraded_errors is not None:
                _record_degraded_error(
                    degraded_errors,
                    phase=error_phase,
                    window_id=result.window_id,
                    signal_ref=result.signal_ref,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
            return
        result.objective_value = objective_value
        result.objective_improvement_pct = objective_improvement_pct
        result.objective_improved = objective_improved
        result.objective_speedup_ratio = _objective_speedup_ratio(
            baseline_objective_value,
            objective_value,
            baseline_objective_mode,
        )
        result.objective_speedup_label = _objective_speedup_label(result.objective_speedup_ratio)
        return

    result.objective_metric = "wall_time"
    result.objective_mode = "minimize"
    result.objective_value = result.wall_time_s
    result.objective_improvement_pct = compute_objective_change_pct(
        baseline_wall_time,
        result.wall_time_s,
        "minimize",
    )
    result.objective_improved = bool(result.objective_improvement_pct > 0.0)
    result.objective_speedup_ratio = _objective_speedup_ratio(
        baseline_wall_time,
        result.wall_time_s,
        "minimize",
    )
    result.objective_speedup_label = _objective_speedup_label(result.objective_speedup_ratio)


def select_promotions(
    short_results: list[QuickShortResult],
    promotion_fraction: float | None = None,
    max_promoted_candidates: int | None = None,
    *,
    full_run_ratio: float | None = None,
    max_full_runs_per_group: int | None = None,
) -> list[QuickPromotionDecision]:
    ratio = full_run_ratio if full_run_ratio is not None else promotion_fraction
    cap = (
        max_full_runs_per_group if max_full_runs_per_group is not None else max_promoted_candidates
    )
    ratio = 0.20 if ratio is None else ratio
    cap = 3 if cap is None else cap

    usable = [
        result
        for result in short_results
        if (
            result.target_signal_improvement_pct is not None
            and result.status in {"screened", "neutral"}
            and not _result_is_auto_validated(result)
        )
    ]
    usable.sort(
        key=lambda item: (
            float(item.target_signal_improvement_pct or float("-inf")),
            item.wall_time_s * -1.0,
        ),
        reverse=True,
    )

    promoted_count = 0
    if usable:
        promoted_count = min(
            cap,
            max(1, int(math.ceil(len(usable) * ratio))),
        )
    promoted_ids = {result.candidate_id for result in usable[:promoted_count]}

    decisions: list[QuickPromotionDecision] = []
    for result in short_results:
        if result.candidate_id in promoted_ids:
            rank = next(
                index + 1
                for index, item in enumerate(usable)
                if item.candidate_id == result.candidate_id
            )
            decisions.append(
                QuickPromotionDecision(
                    candidate_id=result.candidate_id,
                    candidate_label=result.candidate_label,
                    config_id=result.config_id,
                    signal_ref=result.signal_ref,
                    signal_name=result.signal_name,
                    window_id=result.window_id,
                    promoted=True,
                    rank=rank,
                    reason="top short-screen signal improvement",
                )
            )
        else:
            reason = "outside promotion cap"
            if _result_is_auto_validated(result):
                reason = "already completed full workload during short screen"
            elif result.status == "failed":
                reason = "short screen failed before producing usable validation data"
            elif result.target_signal_improvement_pct is None or result.status == "unavailable":
                reason = "no usable target-signal score"
            decisions.append(
                QuickPromotionDecision(
                    candidate_id=result.candidate_id,
                    candidate_label=result.candidate_label,
                    config_id=result.config_id,
                    signal_ref=result.signal_ref,
                    signal_name=result.signal_name,
                    window_id=result.window_id,
                    promoted=False,
                    rank=None,
                    reason=reason,
                )
            )
    return decisions


def classify_signal_usefulness(
    *,
    target_spec: QuickSignalSpec,
    short_results: list[QuickShortResult],
    full_results: list[QuickShortResult],
    defaults: QuickModeDefaults,
) -> QuickSignalUsefulness:
    signal_short = [item for item in short_results if item.signal_ref == target_spec.signal_ref]
    numeric_short = [
        item for item in signal_short if item.target_signal_improvement_pct is not None
    ]
    if not numeric_short:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="unavailable",
            notes="screening samples never produced a numeric target signal",
        )
    max_abs_short = max(
        abs(float(item.target_signal_improvement_pct or 0.0)) for item in numeric_short
    )
    if max_abs_short < defaults.neutral_change_pct:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="neutral",
            notes="short-run signal changes stayed near neutral",
        )

    signal_full = [item for item in full_results if item.signal_ref == target_spec.signal_ref]
    if not signal_full:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="unchecked",
            notes="signal screened candidates but did not reach full validation",
        )

    full_with_objective = [item for item in signal_full if item.objective_improved is not None]
    if not full_with_objective:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="unavailable",
            notes="full validation lacked end-to-end objective data",
        )

    best_short = max(
        numeric_short, key=lambda item: float(item.target_signal_improvement_pct or float("-inf"))
    )
    best_objective = max(
        full_with_objective,
        key=lambda item: float(item.objective_improvement_pct or float("-inf")),
    )
    best_objective_change = float(best_objective.objective_improvement_pct or 0.0)

    if best_objective_change <= -defaults.neutral_change_pct:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="rejected",
            notes="validated candidates regressed on the full-run optimization objective",
        )
    if abs(best_objective_change) < defaults.neutral_change_pct:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="neutral",
            notes=(
                "validated candidates changed the signal but left the full-run "
                "objective near neutral"
            ),
        )
    if best_objective.candidate_id == best_short.candidate_id:
        return QuickSignalUsefulness(
            signal_ref=target_spec.signal_ref,
            signal_name=target_spec.metric,
            status="predictive",
            notes="best short-screen candidate also won on the full-run objective",
        )
    return QuickSignalUsefulness(
        signal_ref=target_spec.signal_ref,
        signal_name=target_spec.metric,
        status="weak",
        notes="signal found a full-run winner, but the short-run ranking shifted during validation",
    )


def _sample_row_headers() -> list[str]:
    return [
        "run_id",
        "config_id",
        "run_kind",
        "elapsed_sec",
        "timestamp",
        "signal_name",
        "signal_value",
        "signal_unit",
        "probe_status",
    ]


def _flatten_sample_rows(
    *,
    run_id: str,
    config_id: str,
    run_kind: str,
    started_at: datetime,
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    profiling_json = artifact.get("profiling_json", {})
    if not isinstance(profiling_json, dict):
        return []
    samples = profiling_json.get("periodic_probe_samples", [])
    if not isinstance(samples, list):
        return []

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        elapsed_sec = _normalize_numeric(sample.get("time_offset_s"))
        results = sample.get("results", {})
        if elapsed_sec is None or not isinstance(results, dict):
            continue
        timestamp = (started_at + timedelta(seconds=elapsed_sec)).isoformat()
        for signal_name, result in results.items():
            if not isinstance(result, dict):
                continue
            signal_value, signal_unit, probe_status = _extract_probe_value(result)
            rows.append(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "run_kind": run_kind,
                    "elapsed_sec": elapsed_sec,
                    "timestamp": timestamp,
                    "signal_name": signal_name,
                    "signal_value": signal_value,
                    "signal_unit": signal_unit,
                    "probe_status": probe_status,
                }
            )
    return rows


def _coerce_artifact_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _persist_run_artifacts(
    *,
    artifact_dir: Path,
    run_id: str,
    config_id: str,
    run_kind: str,
    candidate_id: int | None,
    candidate_label: str,
    signal_ref: str,
    signal_name: str,
    window_id: str,
    command: list[str],
    env_overrides: dict[str, str],
    artifact: dict[str, Any],
) -> tuple[QuickRunMetadata, list[dict[str, Any]]]:
    finished_at = _coerce_artifact_datetime(artifact.get("finished_at")) or datetime.now(
        timezone.utc
    )
    run_started = _coerce_artifact_datetime(artifact.get("started_at")) or (
        finished_at - timedelta(seconds=max(float(artifact.get("wall_time", 0.0)), 0.0))
    )
    stdout_text = str(artifact.get("stdout", ""))
    stderr_text = str(artifact.get("stderr", ""))
    run_path = artifact_dir / "runs" / config_id
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_path = run_path / "run_artifact.json"
    stdout_path = run_path / "stdout.txt"
    stderr_path = run_path / "stderr.txt"
    _write_json(artifact_path, artifact)
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    metadata = QuickRunMetadata(
        run_id=run_id,
        config_id=config_id,
        run_kind=run_kind,
        candidate_id=candidate_id,
        candidate_label=candidate_label,
        signal_ref=signal_ref,
        signal_name=signal_name,
        window_id=window_id,
        command=list(command),
        env_overrides=dict(env_overrides),
        started_at=run_started.isoformat(),
        finished_at=finished_at.isoformat(),
        wall_time_s=float(artifact.get("wall_time", 0.0)),
        returncode=int(artifact.get("returncode", 0)),
        stop_reason=str(artifact.get("stop_reason", "completed")),
        stopped_intentionally=bool(artifact.get("stopped_intentionally", False)),
        artifact_path=str(artifact_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
    _write_json(run_path / "run_metadata.json", metadata)
    stage_file = artifact_dir / f"{run_kind}_run_metadata.jsonl"
    _append_jsonl(stage_file, metadata)
    sample_rows = _flatten_sample_rows(
        run_id=run_id,
        config_id=config_id,
        run_kind=run_kind,
        started_at=run_started,
        artifact=artifact,
    )
    for row in sample_rows:
        _append_jsonl(artifact_dir / "signal_samples.jsonl", row)
        _append_csv_row(artifact_dir / "signal_samples.csv", row, _sample_row_headers())
    return metadata, sample_rows


def _candidate_signature(plan: dict[str, Any]) -> str:
    return json.dumps(
        {
            "cmd": list(plan.get("cmd", [])),
            "env": dict(plan.get("env_overrides", {})),
        },
        sort_keys=True,
    )


def _candidate_plan_guard_reason(plan: dict[str, Any]) -> str | None:
    cmd = plan.get("cmd")
    if (
        not isinstance(cmd, list)
        or not cmd
        or not all(isinstance(token, str) and token for token in cmd)
    ):
        return "candidate plan command is empty or invalid"
    env_overrides = plan.get("env_overrides", {})
    if not isinstance(env_overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env_overrides.items()
    ):
        return "candidate plan environment overrides are invalid"
    return None


def _candidate_plan_label(plan: dict[str, Any], *, fallback: str) -> str:
    label = str(plan.get("label") or "").strip()
    return label or fallback


def _summarize_abnormal_window(window: QuickBaselineWindow) -> str:
    return (
        f"{window.signal_name} [{window.window_id}; {window.start_offset_s:.1f}s, "
        f"{window.end_offset_s:.1f}s] "
        f"severity={window.severity:.1f}% reason={window.reason}"
    )


def _ordered_unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _render_prompt_list(items: Sequence[str], *, empty: str = "(none)") -> str:
    unique_items = _ordered_unique(items)
    if not unique_items:
        return empty
    return "\n".join(f"- {item}" for item in unique_items)


def _recommendation_action_controls(actions: Sequence[dict[str, Any]]) -> list[str]:
    controls: list[str] = []
    for action in actions:
        config_type = str(action.get("config_type", "")).strip()
        parameter = str(action.get("parameter", "")).strip()
        if config_type == "env_var" and parameter:
            controls.append(parameter)
        elif config_type:
            controls.append(config_type)
    return _ordered_unique(controls)


def _successful_prior_config_summaries(
    short_results: Sequence[QuickShortResult],
    full_results: Sequence[QuickShortResult],
) -> list[str]:
    summaries: list[str] = []
    for result in full_results:
        if result.objective_value is None:
            continue
        summaries.append(
            f"{result.candidate_label} "
            f"(full validation {_format_objective_speedup(result.objective_speedup_ratio)})"
        )
    for result in short_results:
        if not _result_is_auto_validated(result):
            continue
        summaries.append(
            f"{result.candidate_label} "
            f"(completed short run {_format_objective_speedup(result.objective_speedup_ratio)})"
        )
    return _ordered_unique(summaries)


def _build_quick_recommendation_prompt(
    *,
    cmd_parts: Sequence[str],
    inline_env_overrides: dict[str, str],
    excluded_config_content: str,
    excluded_config_list: Sequence[str],
    tunable_whitelist_content: str,
    tunable_whitelist_list: Sequence[str] | None,
    window: QuickBaselineWindow,
    max_iterations: int,
    baseline_profiling_json: dict[str, Any],
    generated_hypothesis_content: str,
    loaded_manual_content: str,
    manual_hint_content: str,
    fixed_budget_remaining: int | None,
    successful_prior_configs: Sequence[str],
    skipped_rejected_candidates: Sequence[str],
    untouched_controls: Sequence[str],
    retry_reason: str,
) -> str:
    action_limit = max(1, min(max_iterations, 5))
    return f"""Analyze this workload for quick screening mode and propose candidate actions.

## Command
{" ".join(cmd_parts)}

## Current Environment Overrides
{json.dumps(inline_env_overrides, indent=2, sort_keys=True) if inline_env_overrides else "(none)"}

## Candidate Base State
- Every recommended action is applied relative to the Command and Current Environment Overrides
  shown above.
- Prior successful, skipped, or rejected candidates are history only; they are not active unless
  their changes appear in the Command or Current Environment Overrides.
- If you want to keep an environment or command-parameter value that is not in the current
  Command/Environment, include it explicitly in this response.

## Immutable Controls
{excluded_config_content}

## Tunable Whitelist
{tunable_whitelist_content}

## Ranked Abnormal Window
- signal: {window.signal_name}
- ref: {window.signal_ref}
- hypothesis: {window.hypothesis_name or "(unknown)"}
- expected direction: {window.expected_direction}
- severity: {window.severity:.1f}
- window: {window.start_offset_s:.2f}s -> {window.end_offset_s:.2f}s
- reason: {window.reason}
- candidate controls: {
            ", ".join(window.candidate_controls) if window.candidate_controls else "(not declared)"
        }

## Fixed-Budget Retry Context
- remaining fixed candidate budget: {
            fixed_budget_remaining if fixed_budget_remaining is not None else "(not enabled)"
        }
- retry reason: {retry_reason or "first attempt for this window"}
- successful prior configs:
{_render_prompt_list(successful_prior_configs, empty="(none yet)")}
- skipped or rejected candidates for this window:
{_render_prompt_list(skipped_rejected_candidates, empty="(none yet)")}
- untouched candidate controls for this window:
{_render_prompt_list(untouched_controls, empty="(unknown or fully explored)")}
- protected immutable controls:
{_render_prompt_list(excluded_config_list, empty="(none)")}
- allowed tunable controls:
{_render_prompt_list(tunable_whitelist_list or [], empty="(not restricted)")}

## Rules
- Short-run ranking must use only target-signal improvement inside the matching window.
- Do not optimize directly for wall time or the primary objective in the short screen.
- Keep only the exact immutable controls listed above unchanged; do not treat same-library
  or related controls as immutable unless their exact names are listed.
- If a tunable whitelist is active, recommend and change only exact controls listed in
  the Tunable Whitelist section.
- Keep workload-defining arguments fixed.
- If this window is being retried under a fixed budget, propose finer-grained variants of
  earlier ideas or use untouched candidate controls; outputting a candidate is mandatory while
  any allowed, new, safe candidate can be formed.
- In any mode, you may ignore whether direct evidence supports the decision, and you do not
  need to provide evidence to support a candidate.
- Do not reject or omit a candidate only because evidence is absent or insufficient.
- If no untouched control seems actionable, try different concrete values for previously tried
  parameters or environment variables until the allowed value space is exhausted or the fixed
  budget is exhausted.
- Do not repeat skipped or rejected candidates exactly.
- Treat each object in `candidates` as one complete concrete candidate delta from the Candidate
  Base State, not as a patch on your previous suggestion.
- To test mixed environment-variable and command-parameter changes together, put them in the same
  candidate's `changes` object as exact-name-to-value mappings.
- Keys starting with `-` are command-line parameters; all other keys are environment variables.
- Separate candidate objects are alternatives; they are not sequential edits to combine into one
  command.
- Each materialized candidate is one runnable command plus its environment overrides.
- Return JSON only.
- Provide up to {action_limit} candidate objects ordered best-first.

## Baseline Profiling JSON
{profiling_json_for_prompt(baseline_profiling_json)}

## Generated Contract Hypotheses
{generated_hypothesis_content if generated_hypothesis_content else "(No generated contract hypotheses)"}

## Loaded Manual Context
{loaded_manual_content if loaded_manual_content else "(No manual content loaded)"}

## Application Manual Hints
{manual_hint_content if manual_hint_content else "(No application manual hints provided)"}

## JSON Schema
{{
  "summary": "short decision summary",
  "manual_citation": "optional concrete rule from loaded manual(s), or empty",
  "evidence": ["optional concise evidence item; may be empty", "..."],
  "hypothesis_refs": ["optional hypothesis ids to prioritize next profiling round"],
  "candidates": [
    {{
      "reason": "why this complete candidate should be tried",
      "manual_citation": "optional manual/contract rule supporting this candidate",
      "changes": {{
        "--exact-command-parameter": "concrete value",
        "EXACT_ENV_NAME": "concrete value"
      }}
    }}
  ]
}}

Do not use placeholder keys in real responses. Do not use bundled or alternation syntax such as
`AAA|BBB` anywhere. Values must be concrete scalar target values, not arrays or objects."""


def _maybe_reprompt_same_window(
    fixed_budget_state: FixedBudgetState,
    same_window_no_action_reprompts: int,
) -> tuple[bool, bool]:
    if not fixed_budget_state.enabled:
        return False, False
    if not fixed_budget_state.can_consume():
        return False, False
    if fixed_budget_state.record_no_action_reprompt(same_window_no_action_reprompts):
        return True, False
    return False, True


def _build_signal_window_groups(
    ranked_windows: list[QuickBaselineWindow],
    checked_window_ids: list[str],
) -> list[QuickSignalWindowGroup]:
    checked_set = set(checked_window_ids)
    grouped: dict[str, QuickSignalWindowGroup] = {}
    signal_order: list[str] = []
    for window in ranked_windows:
        if window.signal_ref not in grouped:
            grouped[window.signal_ref] = QuickSignalWindowGroup(
                signal_ref=window.signal_ref,
                signal_name=window.signal_name,
                windows=[],
            )
            signal_order.append(window.signal_ref)
        grouped[window.signal_ref].windows.append(
            QuickSignalWindowRef(
                window_id=window.window_id,
                start_offset_s=window.start_offset_s,
                end_offset_s=window.end_offset_s,
                severity=window.severity,
                checked=window.window_id in checked_set,
            )
        )
    return [grouped[signal_ref] for signal_ref in signal_order]


def _format_signal_window_group(group: QuickSignalWindowGroup) -> str:
    windows = []
    for window in group.windows:
        status = "checked" if window.checked else "ranked"
        windows.append(
            f"{window.window_id} ({window.start_offset_s:.1f}s-"
            f"{window.end_offset_s:.1f}s, {status})"
        )
    return "; ".join(windows) if windows else "(none)"


def _summarize_unavailable_signals(signal_usefulness: list[QuickSignalUsefulness]) -> str:
    unavailable = [entry for entry in signal_usefulness if entry.status == "unavailable"]
    if not unavailable:
        return "no periodic probe samples were captured during the baseline run"
    summaries = [f"{entry.signal_name}: {entry.notes}" for entry in unavailable[:3]]
    if len(unavailable) > 3:
        summaries.append(f"{len(unavailable) - 3} more unavailable signal(s)")
    return "; ".join(summaries)


def _conclusion_from_results(
    *,
    profiling_available: bool,
    abnormal_windows: list[QuickBaselineWindow],
    checked_window_ids: list[str],
    short_results: list[QuickShortResult],
    full_results: list[QuickShortResult],
    selected_result: QuickShortResult | None,
    budget_exhausted: bool,
    fixed_budget_exhausted: bool,
    unfillable_no_action: bool,
    defaults: QuickModeDefaults,
    signal_usefulness: list[QuickSignalUsefulness] | None = None,
    degraded_errors: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    if not profiling_available:
        return "profiling_unavailable", _summarize_unavailable_signals(signal_usefulness or [])
    if not abnormal_windows:
        unavailable_summary = _summarize_unavailable_signals(signal_usefulness or [])
        if unavailable_summary != "no periodic probe samples were captured during the baseline run":
            return (
                "no_valid_signal_windows",
                "baseline analysis found no useful ranked abnormal signal windows; "
                f"{unavailable_summary}",
            )
        return (
            "no_valid_signal_windows",
            "baseline analysis found no useful ranked abnormal signal windows",
        )
    if (
        selected_result is not None
        and selected_result.objective_improvement_pct is not None
        and selected_result.objective_improvement_pct >= defaults.significant_improvement_pct
    ):
        return (
            "significant_improvement_found",
            "a validated candidate met the significant full-run improvement threshold",
        )
    if degraded_errors:
        llm_errors = [entry for entry in degraded_errors if _is_llm_degraded_error(entry)]
        if llm_errors:
            first = llm_errors[0]
            return (
                "llm_provider_error",
                "quick mode preserved partial evidence, but an LLM/provider failure "
                "blocked complete candidate search: "
                f"{first.get('phase', 'unknown phase')} "
                f"{first.get('error_type', '')}: {first.get('message', '')}",
            )
        first = degraded_errors[0]
        return (
            "quick_mode_degraded",
            "quick mode preserved partial evidence, but a nonfatal internal error "
            "blocked complete candidate search: "
            f"{first.get('phase', 'unknown phase')} "
            f"{first.get('error_type', '')}: {first.get('message', '')}",
        )
    if unfillable_no_action:
        return (
            "unfillable_no_action",
            "fixed-budget quick mode exhausted same-window re-prompts without "
            "finding a concrete executable candidate",
        )
    if fixed_budget_exhausted and len(checked_window_ids) < len(abnormal_windows):
        return (
            "budget_exhausted",
            "fixed candidate budget was exhausted before all ranked signal windows were checked",
        )
    if budget_exhausted and len(checked_window_ids) < len(abnormal_windows):
        return (
            "budget_exhausted",
            "quick-mode run budgets were exhausted before all ranked signal windows were checked",
        )
    if any(
        (item.target_signal_improvement_pct or 0.0) >= defaults.neutral_change_pct
        for item in short_results
    ):
        if not any(bool(item.objective_improved) for item in full_results):
            return (
                "signal_improved_but_no_full_speedup",
                "short-run proxy signals improved, but no validated candidate "
                "improved the full-run objective",
            )
    return "safe_rejection", "validated candidates did not deliver a useful end-to-end improvement"


async def _run_candidate(
    *,
    run_executor: RunExecutor,
    agent: SysPilotAgent,
    plan: dict[str, Any],
    timeout: int,
    workdir: str | None,
    memory_high: str | None,
    hypothesis_profiles: list[dict[str, Any]],
    sink: OutputSink,
    sample_interval_sec: float,
    stop_after_seconds: float | None,
    run_kind: str,
) -> dict[str, Any]:
    return await run_executor(
        agent,
        plan["cmd"],
        timeout,
        workdir=workdir,
        memory_high=memory_high,
        env_overrides=plan["env_overrides"],
        hypothesis_profiles=hypothesis_profiles,
        live_output_callback=lambda stream_name, line: _plain_live_output_callback(
            sink,
            stream_name,
            line,
            run_kind=run_kind,
        ),
        periodic_sample_interval_s=sample_interval_sec,
        stop_after_seconds=stop_after_seconds,
        intentional_stop_reason="screen_budget_reached" if run_kind == "short" else "completed",
    )


async def run_quick_tune_loop_async(
    *,
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    fixed_budget: int | None,
    fixed_goal: float | None,
    max_iterations: int,
    timeout: int,
    primary_objective: str | None,
    primary_objective_mode: str | None,
    workdir: str | None,
    memory_high: str | None,
    sink: OutputSink | None,
    event_emitter: TuneEventEmitterProtocol | None,
    structured_context_bundle: dict[str, object] | None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None,
    run_executor: RunExecutor,
    parse_spawn_actions: ParseActions,
    build_spawn_candidates: BuildCandidates,
    extract_hypothesis_profiles: ExtractHypothesisProfiles,
    recommendation_next_action_display: RecommendationActionDisplay,
    extract_inline_env_assignments: ExtractInlineEnvAssignments,
    extract_metrics_from_output: ExtractMetricsFromOutput,
    resolve_primary_objective_metric: ResolvePrimaryObjectiveMetric,
    extract_primary_objective_value: ExtractPrimaryObjectiveValue,
    infer_primary_objective_mode: InferPrimaryObjectiveMode,
    compare_objective_values: CompareObjectiveValues,
    compute_objective_change_pct: ComputeObjectiveChangePct,
    objective_display_name: ObjectiveDisplayName,
    format_metric_value: FormatMetricValue,
    defaults: QuickModeDefaults | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> None:
    tune_started_at = time.time()
    baseline_config_find_wall_time_s = 0.0
    result_find_wall_times: dict[str, float] = {}
    configure_prompt = getattr(agent, "configure_tuning_prompt", None)
    if callable(configure_prompt):
        agent_context = getattr(agent, "context", None)
        configure_prompt(
            contract_enabled=bool(structured_context_bundle),
            manual_enabled=bool(getattr(agent_context, "loaded_manuals", {}) or manual_hints),
            quick_mode=True,
            profiling_enabled=True,
        )

    def _elapsed_payload(config_find_wall_time_s: float) -> dict[str, float]:
        return {
            "total_wall_time_s": round(time.time() - tune_started_at, 6),
            "config_find_wall_time_s": round(config_find_wall_time_s, 6),
        }

    if sink is None:
        sink = make_output_sink(ui_mode=TuneUIMode.PLAIN, console=Console())
    resolved_defaults = defaults or QuickModeDefaults()
    fixed_budget_state = FixedBudgetState(total=fixed_budget)
    if fixed_budget_state.enabled:
        fixed_budget_state.max_no_action_reprompts = min(
            fixed_budget_state.max_no_action_reprompts,
            cast(int, fixed_budget_state.total),
        )

    cmd_parts = shlex.split(command)
    if not cmd_parts:
        sink.print("[red]Spawn command is empty after parsing[/red]")
        raise SystemExit(1)

    cmd_parts, inline_env_overrides = extract_inline_env_assignments(cmd_parts)
    excluded_config_list = list(excluded_config_names or [])
    tunable_whitelist_list = (
        None if tunable_whitelist_names is None else list(tunable_whitelist_names)
    )
    loaded_manual_names = sorted(getattr(agent.context, "loaded_manuals", {}).keys())
    manual_hint_list = list(manual_hints)
    manual_context_status = (
        f"Manual context: {len(loaded_manual_names)} loaded, "
        f"{len(manual_hint_list)} hints provided"
    )
    hypothesis_profiles = [
        item for item in (initial_hypothesis_profiles or []) if isinstance(item, dict)
    ]
    signal_specs = _extract_signal_specs(structured_context_bundle, hypothesis_profiles)
    if not signal_specs:
        sink.print(
            "[red]Quick mode needs contract signals with numeric directional virtual probes[/red]"
        )
        raise SystemExit(1)

    run_id = str(getattr(event_emitter, "run_id", "quick-run"))
    artifact_dir = Path("artifacts") / "quick_tune_runs" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": "quick",
        "command": command,
        "contracts": list(contracts),
        "loaded_manuals": loaded_manual_names,
        "manual_hints": manual_hint_list,
        "timeout_s": timeout,
        "max_iterations": max_iterations,
        "fixed_goal_threshold": fixed_goal,
        "primary_objective_query": primary_objective or "",
        "primary_objective_mode": primary_objective_mode or "",
        "workdir": workdir,
        "memory_high": memory_high,
        "excluded_config_names": excluded_config_list,
        "tunable_whitelist_names": tunable_whitelist_list or [],
        "defaults": asdict(resolved_defaults),
        **_fixed_budget_payload(fixed_budget_state),
    }
    _write_json(artifact_dir / "metadata.json", metadata)

    _emit_tune_event(
        event_emitter,
        "run.started",
        {
            "mode": "spawn",
            "quick_mode": True,
            "command": command,
            "workdir": workdir or "",
            "memory_high": memory_high or "",
            "max_iterations": max_iterations,
            "fixed_goal_threshold": fixed_goal,
            "timeout_s": timeout,
            "loaded_contracts": list(contracts),
            "loaded_manuals": loaded_manual_names,
            "manual_hints": manual_hint_list,
            "excluded_config_names": excluded_config_list,
            "tunable_whitelist_names": tunable_whitelist_list or [],
            "primary_objective_query": primary_objective or "",
            "primary_objective_mode": primary_objective_mode or "",
            "artifact_dir": str(artifact_dir),
            **_fixed_budget_payload(fixed_budget_state),
        },
    )

    sink.panel(
        f"[bold green]SysPilot Quick Tuning Mode[/bold green]\n"
        f"Command: {command}\n"
        f"Workdir: {workdir or '(current)'}\n"
        f"MemoryHigh: {memory_high or '(none)'}\n"
        f"Contracts: {len(list(contracts))} loaded\n"
        f"Tunable whitelist: "
        f"{', '.join(tunable_whitelist_list) if tunable_whitelist_list is not None and tunable_whitelist_list else ('(empty)' if tunable_whitelist_list is not None else '(not set)')}\n"
        f"{manual_context_status}\n"
        f"Primary objective: {primary_objective or '(wall time)'}"
        f"{f' ({primary_objective_mode})' if primary_objective_mode else ''}\n"
        f"Artifact dir: {artifact_dir}",
        title="Starting Quick Mode",
    )

    sink.print("\n[bold cyan]Quick Phase 1: Baseline Run with Periodic Signal Sampling[/bold cyan]")
    _emit_tune_event(
        event_emitter,
        "phase.started",
        {"mode": "spawn", "quick_mode": True, "phase": "spawn.quick_baseline"},
    )
    baseline_artifact = await run_executor(
        agent,
        cmd_parts,
        timeout,
        workdir=workdir,
        memory_high=memory_high,
        env_overrides=inline_env_overrides,
        hypothesis_profiles=hypothesis_profiles,
        live_output_callback=lambda stream_name, line: _plain_live_output_callback(
            sink,
            stream_name,
            line,
            run_kind="baseline",
        ),
        periodic_sample_interval_s=resolved_defaults.sample_interval_sec,
    )
    baseline_metadata, baseline_sample_rows = _persist_run_artifacts(
        artifact_dir=artifact_dir,
        run_id=run_id,
        config_id="baseline",
        run_kind="baseline",
        candidate_id=None,
        candidate_label="baseline",
        signal_ref="",
        signal_name="",
        window_id="",
        command=cmd_parts,
        env_overrides=inline_env_overrides,
        artifact=baseline_artifact,
    )
    _emit_tune_event(
        event_emitter,
        "baseline.finished",
        {
            "mode": "spawn",
            "quick_mode": True,
            "phase": "spawn.quick_baseline",
            "returncode": baseline_artifact.get("returncode", 0),
            "wall_time_s": baseline_artifact.get("wall_time", 0.0),
            **_event_output_payload(
                str(baseline_artifact.get("stdout", "")), str(baseline_artifact.get("stderr", ""))
            ),
        },
    )

    baseline_wall_time = float(baseline_artifact.get("wall_time", 0.0))
    baseline_objective_metric = "wall_time"
    baseline_objective_mode = "minimize"
    baseline_objective_value: float | None = baseline_wall_time
    if primary_objective:
        sink.print("  Parsing baseline output for primary objective...")
        baseline_metrics = await extract_metrics_from_output(
            agent,
            str(baseline_artifact.get("stdout", "")),
            f"baseline run without tuning actions; primary objective query: {primary_objective}",
        )
        resolved_metric = resolve_primary_objective_metric(baseline_metrics, primary_objective)
        if resolved_metric is None:
            fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
            sink.print(
                "[red]Baseline output did not resolve primary objective query "
                f"'{primary_objective}' to a numeric metric[/red]"
            )
            _emit_tune_event(
                event_emitter,
                "run.finished",
                {
                    "mode": "spawn",
                    "quick_mode": True,
                    "status": "failed",
                    "failure_reason": "missing_primary_objective",
                    **_elapsed_payload(baseline_config_find_wall_time_s),
                    **_fixed_budget_payload(
                        fixed_budget_state,
                        stop_reason=fixed_budget_stop_reason,
                    ),
                    **fixed_goal_payload(
                        threshold=fixed_goal,
                        value=None,
                        objective_mode="minimize",
                        stop_reason="hard_failure" if fixed_goal is not None else "",
                    ),
                },
            )
            raise SystemExit(1)
        baseline_objective_metric = resolved_metric
        baseline_objective_mode = (
            primary_objective_mode or infer_primary_objective_mode(resolved_metric) or ""
        )
        if not baseline_objective_mode:
            fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
            sink.print(
                "[red]Could not infer --primary-objective-mode after resolving "
                f"'{primary_objective}' to '{baseline_objective_metric}'. Please "
                "specify --primary-objective-mode.[/red]"
            )
            _emit_tune_event(
                event_emitter,
                "run.finished",
                {
                    "mode": "spawn",
                    "quick_mode": True,
                    "status": "failed",
                    "failure_reason": "missing_primary_objective_mode",
                    **_elapsed_payload(baseline_config_find_wall_time_s),
                    **_fixed_budget_payload(
                        fixed_budget_state,
                        stop_reason=fixed_budget_stop_reason,
                    ),
                    **fixed_goal_payload(
                        threshold=fixed_goal,
                        value=None,
                        objective_mode="minimize",
                        stop_reason="hard_failure" if fixed_goal is not None else "",
                    ),
                },
            )
            raise SystemExit(1)
        baseline_objective_value = extract_primary_objective_value(
            baseline_metrics, baseline_objective_metric
        )
        if baseline_objective_value is None:
            fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
            sink.print(
                "[red]Baseline output did not produce numeric primary objective "
                f"'{primary_objective}'[/red]"
            )
            _emit_tune_event(
                event_emitter,
                "run.finished",
                {
                    "mode": "spawn",
                    "quick_mode": True,
                    "status": "failed",
                    "failure_reason": "missing_primary_objective",
                    **_elapsed_payload(baseline_config_find_wall_time_s),
                    **_fixed_budget_payload(
                        fixed_budget_state,
                        stop_reason=fixed_budget_stop_reason,
                    ),
                    **fixed_goal_payload(
                        threshold=fixed_goal,
                        value=None,
                        objective_mode="minimize",
                        stop_reason="hard_failure" if fixed_goal is not None else "",
                    ),
                },
            )
            raise SystemExit(1)
        sink.print(
            "  Resolved primary objective metric: "
            f"{baseline_objective_metric} ({baseline_objective_mode})"
        )
        sink.print(
            f"  Baseline {objective_display_name(primary_objective)}: "
            f"{format_metric_value(baseline_objective_value)}"
        )
    baseline_config_find_wall_time_s = time.time() - tune_started_at

    ranked_windows, initial_usefulness, profiling_available = rank_abnormal_windows(
        signal_specs,
        baseline_artifact,
        resolved_defaults,
    )
    ranked_windows = ranked_windows[: resolved_defaults.max_signal_groups]
    _write_json(artifact_dir / "abnormal_windows.json", ranked_windows)
    for window in ranked_windows:
        _append_jsonl(artifact_dir / "abnormal_windows.jsonl", window)

    baseline_summary = QuickBaselineSummary(
        baseline_run=baseline_metadata,
        objective_metric=baseline_objective_metric,
        objective_mode=baseline_objective_mode,
        objective_value=baseline_objective_value,
        sample_row_count=len(baseline_sample_rows),
        useful_signal_count=len({window.signal_ref for window in ranked_windows}),
        abnormal_window_count=len(ranked_windows),
    )

    _emit_tune_event(
        event_emitter,
        "phase.finished",
        {
            "mode": "spawn",
            "quick_mode": True,
            "phase": "spawn.quick_baseline",
            "status": "success" if int(baseline_artifact.get("returncode", 0)) == 0 else "failed",
            "abnormal_window_count": len(ranked_windows),
        },
    )

    if not ranked_windows:
        conclusion, reason = _conclusion_from_results(
            profiling_available=profiling_available,
            abnormal_windows=[],
            checked_window_ids=[],
            short_results=[],
            full_results=[],
            selected_result=None,
            budget_exhausted=False,
            fixed_budget_exhausted=False,
            unfillable_no_action=False,
            defaults=resolved_defaults,
            signal_usefulness=initial_usefulness,
        )
        report = QuickModeReport(
            schema_version="syspilot.quick_mode/v3",
            mode="quick",
            generated_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            primary_objective_query=primary_objective or "",
            primary_objective_metric=baseline_objective_metric,
            primary_objective_mode=baseline_objective_mode,
            baseline_summary=baseline_summary,
            abnormal_windows=[],
            signal_windows=[],
            short_results=[],
            promotions=[],
            full_results=[],
            signal_usefulness=initial_usefulness,
            final_selection_basis=baseline_objective_metric,
            selected_candidate_id=None,
            selected_candidate_label="(none)",
            conclusion=conclusion,
            conclusion_reason=reason,
            checked_window_ids=[],
            artifact_dir=str(artifact_dir),
            degraded_errors=[],
        )
        report_payload = asdict(report)
        report_payload.update(_elapsed_payload(baseline_config_find_wall_time_s))
        report_payload.update(_fixed_budget_payload(fixed_budget_state))
        report_payload.update(
            fixed_goal_payload(
                threshold=fixed_goal,
                value=baseline_objective_value,
                objective_mode=baseline_objective_mode,
            )
        )
        _write_json(artifact_dir / "quick_report.json", report_payload)
        sink.print("[yellow]Quick mode found no useful ranked abnormal signal windows.[/yellow]")
        _emit_tune_event(
            event_emitter,
            "run.finished",
            {
                "mode": "spawn",
                "quick_mode": True,
                "status": conclusion,
                "artifact_dir": str(artifact_dir),
                **_elapsed_payload(baseline_config_find_wall_time_s),
                **_fixed_budget_payload(fixed_budget_state),
                **fixed_goal_payload(
                    threshold=fixed_goal,
                    value=baseline_objective_value,
                    objective_mode=baseline_objective_mode,
                ),
            },
        )
        return

    sink.print("\n[bold cyan]Quick Phase 2: Ranked Signal Screening[/bold cyan]")
    _emit_tune_event(
        event_emitter,
        "phase.started",
        {"mode": "spawn", "quick_mode": True, "phase": "spawn.quick_screen"},
    )

    short_results: list[QuickShortResult] = []
    full_results: list[QuickShortResult] = []
    promotions: list[QuickPromotionDecision] = []
    checked_window_ids: list[str] = []
    used_signatures: set[str] = set()
    signal_spec_map = {spec.signal_ref: spec for spec in signal_specs}
    short_budget_used = 0
    full_budget_used = 0
    budget_exhausted = False
    fixed_budget_exhausted = False
    max_iterations_exhausted = False
    max_iteration_trials_used = 0
    unfillable_no_action = False
    candidate_space_exhausted = False
    degraded_errors: list[dict[str, str]] = []

    fixed_goal_stop_reason = ""

    def _max_iteration_trials_remaining() -> int:
        if fixed_budget_state.enabled:
            return fixed_budget_state.remaining or 0
        return max(0, max_iterations - max_iteration_trials_used)

    def _select_best_validated_result(
        candidates: Sequence[QuickShortResult],
    ) -> QuickShortResult | None:
        if primary_objective:
            objective_candidates = [item for item in candidates if item.objective_value is not None]
            if not objective_candidates:
                return None
            selected = objective_candidates[0]
            for candidate in objective_candidates[1:]:
                assert selected.objective_value is not None
                assert candidate.objective_value is not None
                compare_result = compare_objective_values(
                    candidate.objective_value,
                    selected.objective_value,
                    baseline_objective_mode,
                )
                if compare_result > 0 or (
                    compare_result == 0 and candidate.wall_time_s < selected.wall_time_s
                ):
                    selected = candidate
            return selected
        if candidates:
            return min(candidates, key=lambda item: item.wall_time_s)
        return None

    def _fixed_goal_value(result: QuickShortResult | None) -> float | None:
        if result is None:
            return baseline_objective_value
        if primary_objective:
            return result.objective_value
        return result.wall_time_s

    def _fixed_goal_is_reached(result: QuickShortResult | None) -> bool:
        return fixed_goal_reached(
            _fixed_goal_value(result),
            fixed_goal,
            baseline_objective_mode,
        )

    for window_index, window in enumerate(ranked_windows, start=1):
        if fixed_goal is not None and _fixed_goal_is_reached(None):
            fixed_goal_stop_reason = "fixed_goal_reached"
            break
        if fixed_budget_state.enabled and not fixed_budget_state.can_consume():
            fixed_budget_exhausted = True
            break
        if _max_iteration_trials_remaining() == 0:
            max_iterations_exhausted = True
            if fixed_goal is not None:
                fixed_goal_stop_reason = "max_iterations_exhausted"
            break
        if (
            short_budget_used >= resolved_defaults.max_total_short_runs
            or full_budget_used >= resolved_defaults.max_total_full_runs
        ):
            budget_exhausted = True
            break
        checked_window_ids.append(window.window_id)
        target_spec = signal_spec_map[window.signal_ref]
        sink.print(
            f"  Screening window {window_index}/{len(ranked_windows)}: "
            f"{_summarize_abnormal_window(window)}"
        )
        _emit_tune_event(
            event_emitter,
            "progress.update",
            {
                "mode": "spawn",
                "quick_mode": True,
                "phase": "spawn.quick_screen",
                "window_id": window.window_id,
                "signal_ref": window.signal_ref,
                "signal_name": window.signal_name,
                "severity": window.severity,
            },
        )

        generated_hypothesis_content = hypothesis_profiles_yaml_for_prompt(hypothesis_profiles)
        manual_hint_content = _render_spawn_manual_hints(manual_hints)
        loaded_manual_content = _render_loaded_manual_context(agent.context.loaded_manuals)
        excluded_config_content = _render_excluded_config_names(excluded_config_list)
        tunable_whitelist_content = (
            _render_tunable_whitelist_names(tunable_whitelist_list)
            if tunable_whitelist_list is not None
            else "(not set; all supported non-immutable controls remain tunable)"
        )
        group_short_results: list[QuickShortResult] = []
        skipped_rejected_candidates: list[str] = []
        attempted_controls: set[str] = set()
        retry_reason = ""
        candidate_hypothesis_profiles = hypothesis_profiles
        screen_budget_s = window.end_offset_s
        provider_failed = False
        same_window_no_action_reprompts = 0

        while True:
            prompt = _build_quick_recommendation_prompt(
                cmd_parts=cmd_parts,
                inline_env_overrides=inline_env_overrides,
                excluded_config_content=excluded_config_content,
                excluded_config_list=excluded_config_list,
                tunable_whitelist_content=tunable_whitelist_content,
                tunable_whitelist_list=tunable_whitelist_list,
                window=window,
                max_iterations=_max_iteration_trials_remaining(),
                baseline_profiling_json=cast(
                    dict[str, Any], baseline_artifact.get("profiling_json", {})
                ),
                generated_hypothesis_content=generated_hypothesis_content,
                loaded_manual_content=loaded_manual_content,
                manual_hint_content=manual_hint_content,
                fixed_budget_remaining=fixed_budget_state.remaining,
                successful_prior_configs=_successful_prior_config_summaries(
                    short_results, full_results
                ),
                skipped_rejected_candidates=skipped_rejected_candidates,
                untouched_controls=[
                    control
                    for control in window.candidate_controls
                    if control not in attempted_controls and control not in excluded_config_list
                    and (
                        tunable_whitelist_list is None
                        or control in tunable_whitelist_list
                    )
                ],
                retry_reason=retry_reason,
            )
            try:
                response = await _agent_step_for_prompt(agent, prompt)
            except LLMError as exc:
                _record_degraded_error(
                    degraded_errors,
                    phase="recommendation",
                    window_id=window.window_id,
                    signal_ref=window.signal_ref,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
                _append_jsonl(artifact_dir / "degraded_errors.jsonl", degraded_errors[-1])
                sink.print(
                    "    [yellow]Window recommendation failed because the LLM provider returned "
                    "an error; preserving collected evidence and continuing.[/yellow]"
                )
                _emit_tune_event(
                    event_emitter,
                    "recommendation.failed",
                    {
                        "mode": "spawn",
                        "quick_mode": True,
                        "phase": "spawn.quick_screen",
                        "window_id": window.window_id,
                        "signal_ref": window.signal_ref,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                provider_failed = True
                break

            recommendation = parse_spawn_actions(
                response.content,
                baseline_command_tokens=cmd_parts,
                baseline_env_overrides=inline_env_overrides,
            )
            if not recommendation.get("valid", False):
                error_text = str(recommendation.get("error", "")).strip() or "unknown parse error"
                sink.print(
                    "    [yellow]Window recommendation was not actionable: "
                    f"{error_text}[/yellow]"
                )
                skipped_rejected_candidates.append(f"invalid recommendation: {error_text}")
                retry_reason = f"previous recommendation was not actionable: {error_text}"
                should_reprompt, exhausted_reprompts = _maybe_reprompt_same_window(
                    fixed_budget_state,
                    same_window_no_action_reprompts,
                )
                if should_reprompt:
                    same_window_no_action_reprompts += 1
                    sink.print(
                        "    [yellow]Re-prompting the same ranked window because fixed budget "
                        "remains "
                        "and no concrete candidate was produced.[/yellow]"
                    )
                    continue
                if exhausted_reprompts:
                    unfillable_no_action = True
                break

            candidate_specs = recommendation.get("candidates", [])
            actions = recommendation.get("actions", [])
            if not isinstance(candidate_specs, list) or not candidate_specs:
                sink.print(
                    "    [yellow]Window recommendation returned no concrete candidates; nothing "
                    "executable to screen.[/yellow]"
                )
                skipped_rejected_candidates.append("recommendation returned no concrete candidates")
                retry_reason = "previous recommendation returned no concrete candidates"
                should_reprompt, exhausted_reprompts = _maybe_reprompt_same_window(
                    fixed_budget_state,
                    same_window_no_action_reprompts,
                )
                if should_reprompt:
                    same_window_no_action_reprompts += 1
                    sink.print(
                        "    [yellow]Re-prompting the same ranked window because fixed budget "
                        "remains "
                        "and no concrete candidate was produced.[/yellow]"
                    )
                    continue
                if exhausted_reprompts:
                    unfillable_no_action = True
                break

            attempted_controls.update(_recommendation_action_controls(actions))
            _emit_tune_event(
                event_emitter,
                "recommendation.ready",
                {
                    "mode": "spawn",
                    "quick_mode": True,
                    "phase": "spawn.quick_screen",
                    "window_id": window.window_id,
                    "summary": recommendation.get("summary", ""),
                    "next_action": recommendation_next_action_display(actions),
                },
            )
            candidate_hypothesis_profiles = extract_hypothesis_profiles(
                structured_context_bundle,
                recommendation.get("hypothesis_refs", []),
            )
            if not candidate_hypothesis_profiles:
                candidate_hypothesis_profiles = hypothesis_profiles
            candidate_plan_limit = _max_iteration_trials_remaining()
            if fixed_budget_state.enabled:
                candidate_plan_limit = min(
                    candidate_plan_limit,
                    fixed_budget_state.remaining or 0,
                )
            candidate_plans = build_spawn_candidates(
                cmd_parts,
                candidate_specs,
                candidate_plan_limit,
                base_env_overrides=inline_env_overrides,
                excluded_config_names=excluded_config_list,
                tunable_whitelist_names=tunable_whitelist_list,
            )
            if not candidate_plans:
                sink.print(
                    "    [yellow]No new executable candidate plans remained for this "
                    "window[/yellow]"
                )
                skipped_rejected_candidates.append(
                    "recommendation materialized no executable candidate plans"
                )
                retry_reason = "previous recommendation produced no executable candidate plans"
                should_reprompt, exhausted_reprompts = _maybe_reprompt_same_window(
                    fixed_budget_state,
                    same_window_no_action_reprompts,
                )
                if should_reprompt:
                    same_window_no_action_reprompts += 1
                    sink.print(
                        "    [yellow]Re-prompting the same ranked window because fixed budget "
                        "remains "
                        "and no concrete candidate was produced.[/yellow]"
                    )
                    continue
                if exhausted_reprompts:
                    candidate_space_exhausted = True
                break

            if fixed_budget_state.enabled:
                remaining_budget = fixed_budget_state.remaining or 0
                if remaining_budget <= 0:
                    fixed_budget_exhausted = True
                    break
                if len(candidate_plans) > remaining_budget:
                    candidate_plans = candidate_plans[:remaining_budget]
                    fixed_budget_exhausted = True

            materialized_without_execution = 0
            for plan in candidate_plans:
                if short_budget_used >= resolved_defaults.max_total_short_runs:
                    budget_exhausted = True
                    break
                if fixed_budget_state.enabled and not fixed_budget_state.can_consume():
                    fixed_budget_exhausted = True
                    break
                if _max_iteration_trials_remaining() == 0:
                    max_iterations_exhausted = True
                    break
                max_iteration_trials_used += 1
                candidate_label = _candidate_plan_label(plan, fallback="(candidate)")
                if fixed_budget_state.enabled:
                    consumed = fixed_budget_state.consume(
                        kind="quick_short_candidate",
                        label=candidate_label,
                        reason=f"materialized short-screen candidate for {window.window_id}",
                    )
                    if not consumed:
                        fixed_budget_exhausted = True
                        break
                guard_reason = _candidate_plan_guard_reason(plan)
                if guard_reason is not None:
                    materialized_without_execution += 1
                    skipped_rejected_candidates.append(f"{candidate_label}: {guard_reason}")
                    sink.print(
                        f"    [yellow]Skipping materialized candidate: {guard_reason}[/yellow]"
                    )
                    continue
                signature = _candidate_signature(plan)
                if signature in used_signatures:
                    materialized_without_execution += 1
                    skipped_rejected_candidates.append(
                        f"{candidate_label}: duplicate candidate plan"
                    )
                    sink.print(
                        "    [yellow]Skipping duplicate materialized candidate: "
                        f"{candidate_label}[/yellow]"
                    )
                    continue
                short_budget_used += 1
                candidate_id = len(short_results) + 1
                config_id = f"short-{candidate_id}"
                used_signatures.add(signature)
                attempted_controls.update(plan.get("env_overrides", {}).keys())
                sink.print(f"    Quick screen {candidate_id}: {candidate_label}")
                artifact = await _run_candidate(
                    run_executor=run_executor,
                    agent=agent,
                    plan=plan,
                    timeout=timeout,
                    workdir=workdir,
                    memory_high=memory_high,
                    hypothesis_profiles=candidate_hypothesis_profiles,
                    sink=sink,
                    sample_interval_sec=resolved_defaults.sample_interval_sec,
                    stop_after_seconds=screen_budget_s,
                    run_kind="short",
                )
                run_metadata, _ = _persist_run_artifacts(
                    artifact_dir=artifact_dir,
                    run_id=run_id,
                    config_id=config_id,
                    run_kind="short",
                    candidate_id=candidate_id,
                    candidate_label=candidate_label,
                    signal_ref=window.signal_ref,
                    signal_name=window.signal_name,
                    window_id=window.window_id,
                    command=list(plan["cmd"]),
                    env_overrides=dict(plan["env_overrides"]),
                    artifact=artifact,
                )
                points = _extract_signal_points(artifact, target_spec)
                result = _materialize_result(
                    candidate_id=candidate_id,
                    candidate_label=candidate_label,
                    config_id=config_id,
                    signal_ref=window.signal_ref,
                    signal_name=window.signal_name,
                    window_id=window.window_id,
                    command=list(plan["cmd"]),
                    env_overrides=dict(plan["env_overrides"]),
                    run_kind="short",
                    artifact=artifact,
                    metadata=run_metadata,
                    target_points=points,
                    baseline_window=window,
                    defaults=resolved_defaults,
                )
                if _result_completed_naturally(result):
                    degraded_count_before = len(degraded_errors)
                    await _populate_result_objective(
                        result=result,
                        agent=agent,
                        artifact=artifact,
                        primary_objective=primary_objective,
                        baseline_objective_metric=baseline_objective_metric,
                        baseline_objective_mode=baseline_objective_mode,
                        extract_metrics_from_output=extract_metrics_from_output,
                        extract_primary_objective_value=extract_primary_objective_value,
                        compute_objective_change_pct=compute_objective_change_pct,
                        baseline_objective_value=baseline_objective_value,
                        baseline_wall_time=baseline_wall_time,
                        context=(
                            "candidate completed during quick screen with actions: "
                            f"{result.candidate_label}; primary objective query: "
                            f"{primary_objective}"
                        ),
                        degraded_errors=degraded_errors,
                        error_phase="short_objective_extraction",
                    )
                    if len(degraded_errors) > degraded_count_before:
                        _append_jsonl(artifact_dir / "degraded_errors.jsonl", degraded_errors[-1])
                short_results.append(result)
                result_find_wall_times[result.config_id] = time.time() - tune_started_at
                group_short_results.append(result)
                _emit_tune_event(
                    event_emitter,
                    "candidate.evaluated",
                    {
                        "mode": "spawn",
                        "quick_mode": True,
                        "phase": "spawn.quick_screen",
                        "candidate_id": result.candidate_id,
                        "label": result.candidate_label,
                        "window_id": result.window_id,
                        "signal_ref": result.signal_ref,
                        "signal_name": result.signal_name,
                        "verdict": result.status,
                        "target_signal_improvement_pct": result.target_signal_improvement_pct,
                        "stop_reason": result.stop_reason,
                        "stopped_intentionally": result.stopped_intentionally,
                        "primary_objective_metric": result.objective_metric,
                        "primary_objective_value": result.objective_value,
                        "primary_objective_improvement_pct": result.objective_improvement_pct,
                        "primary_objective_speedup_ratio": result.objective_speedup_ratio,
                    },
                )
                if result.target_signal_improvement_pct is not None:
                    sink.print(
                        f"      {result.signal_name}: "
                        f"{format_metric_value(result.target_signal_mean)} "
                        f"({result.target_signal_improvement_pct:+.1f}% vs baseline window)"
                    )
                else:
                    sink.print(
                        "      [yellow]Target signal unavailable during this short screen[/yellow]"
                    )
                if _result_is_auto_validated(result):
                    sink.print(
                        "      Completed before the quick-screen budget; using this run "
                        "as validated objective evidence."
                    )
                    sink.print(
                        "      Completed-run objective: "
                        f"{format_metric_value(result.objective_value)} "
                        f"({_format_objective_speedup(result.objective_speedup_ratio)})"
                    )
                if result.stop_reason == "screen_budget_reached":
                    sink.print(
                        "      [dim]Stopped intentionally at the quick-screen budget.[/dim]"
                    )

            if (
                group_short_results
                or budget_exhausted
                or fixed_budget_exhausted
                or max_iterations_exhausted
            ):
                break
            if materialized_without_execution > 0:
                retry_reason = (
                    "all materialized candidate plans were skipped before execution; propose "
                    "finer-grained variants or untouched controls"
                )
                should_reprompt, exhausted_reprompts = _maybe_reprompt_same_window(
                    fixed_budget_state,
                    same_window_no_action_reprompts,
                )
                if should_reprompt:
                    same_window_no_action_reprompts += 1
                    sink.print(
                        "    [yellow]All materialized candidates were skipped before execution; "
                        "re-prompting the same ranked window while fixed budget remains.[/yellow]"
                    )
                    continue
                if exhausted_reprompts:
                    candidate_space_exhausted = True
            break

        if provider_failed:
            continue
        if unfillable_no_action:
            break

        if (
            budget_exhausted or fixed_budget_exhausted or max_iterations_exhausted
        ) and not group_short_results:
            break

        group_promotions = select_promotions(
            group_short_results,
            full_run_ratio=resolved_defaults.full_run_ratio,
            max_full_runs_per_group=resolved_defaults.max_full_runs_per_group,
        )
        promotions.extend(group_promotions)
        for recorded_promotion in group_promotions:
            _append_jsonl(artifact_dir / "promotion_decisions.jsonl", recorded_promotion)

        for result in group_short_results:
            matched_promotion = next(
                (item for item in group_promotions if item.candidate_id == result.candidate_id),
                None,
            )
            if matched_promotion is None or not matched_promotion.promoted:
                continue
            if fixed_budget_state.enabled and not fixed_budget_state.can_consume():
                fixed_budget_exhausted = True
                break
            if _max_iteration_trials_remaining() == 0:
                max_iterations_exhausted = True
                break
            if full_budget_used >= resolved_defaults.max_total_full_runs:
                budget_exhausted = True
                break
            max_iteration_trials_used += 1
            if fixed_budget_state.enabled:
                consumed = fixed_budget_state.consume(
                    kind="quick_full_candidate",
                    label=result.candidate_label,
                    reason=f"promoted full validation for {result.window_id}",
                )
                if not consumed:
                    fixed_budget_exhausted = True
                    break
            full_budget_used += 1
            config_id = f"full-{result.candidate_id}"
            sink.print(f"    Full validation: {result.candidate_label}")
            artifact = await _run_candidate(
                run_executor=run_executor,
                agent=agent,
                plan={
                    "cmd": result.command,
                    "env_overrides": result.env_overrides,
                    "label": result.candidate_label,
                },
                timeout=timeout,
                workdir=workdir,
                memory_high=memory_high,
                hypothesis_profiles=candidate_hypothesis_profiles,
                sink=sink,
                sample_interval_sec=resolved_defaults.sample_interval_sec,
                stop_after_seconds=None,
                run_kind="full",
            )
            run_metadata, _ = _persist_run_artifacts(
                artifact_dir=artifact_dir,
                run_id=run_id,
                config_id=config_id,
                run_kind="full",
                candidate_id=result.candidate_id,
                candidate_label=result.candidate_label,
                signal_ref=result.signal_ref,
                signal_name=result.signal_name,
                window_id=result.window_id,
                command=result.command,
                env_overrides=result.env_overrides,
                artifact=artifact,
            )
            points = _extract_signal_points(artifact, target_spec)
            full_result = _materialize_result(
                candidate_id=result.candidate_id,
                candidate_label=result.candidate_label,
                config_id=config_id,
                signal_ref=result.signal_ref,
                signal_name=result.signal_name,
                window_id=result.window_id,
                command=result.command,
                env_overrides=result.env_overrides,
                run_kind="full",
                artifact=artifact,
                metadata=run_metadata,
                target_points=points,
                baseline_window=window,
                defaults=resolved_defaults,
            )
            degraded_count_before = len(degraded_errors)
            await _populate_result_objective(
                result=full_result,
                agent=agent,
                artifact=artifact,
                primary_objective=primary_objective,
                baseline_objective_metric=baseline_objective_metric,
                baseline_objective_mode=baseline_objective_mode,
                extract_metrics_from_output=extract_metrics_from_output,
                extract_primary_objective_value=extract_primary_objective_value,
                compute_objective_change_pct=compute_objective_change_pct,
                baseline_objective_value=baseline_objective_value,
                baseline_wall_time=baseline_wall_time,
                context=(
                    "candidate run with actions: "
                    f"{result.candidate_label}; primary objective query: {primary_objective}"
                ),
                degraded_errors=degraded_errors,
                error_phase="full_objective_extraction",
            )
            if len(degraded_errors) > degraded_count_before:
                _append_jsonl(artifact_dir / "degraded_errors.jsonl", degraded_errors[-1])
            full_results.append(full_result)
            result_find_wall_times[full_result.config_id] = time.time() - tune_started_at
            if full_result.target_signal_improvement_pct is not None:
                sink.print(
                    f"      Full-window {full_result.signal_name}: "
                    f"{format_metric_value(full_result.target_signal_mean)} "
                    f"({full_result.target_signal_improvement_pct:+.1f}% vs baseline window)"
                )
            sink.print(
                "      Full-run objective: "
                f"{format_metric_value(full_result.objective_value)} "
                f"({_format_objective_speedup(full_result.objective_speedup_ratio)})"
            )
            _emit_tune_event(
                event_emitter,
                "candidate.evaluated",
                {
                    "mode": "spawn",
                    "quick_mode": True,
                    "phase": "spawn.quick_full_validation",
                    "candidate_id": full_result.candidate_id,
                    "label": full_result.candidate_label,
                    "window_id": full_result.window_id,
                    "signal_ref": full_result.signal_ref,
                    "signal_name": full_result.signal_name,
                    "verdict": full_result.status,
                    "target_signal_improvement_pct": full_result.target_signal_improvement_pct,
                    "primary_objective_metric": full_result.objective_metric,
                    "primary_objective_value": full_result.objective_value,
                    "primary_objective_improvement_pct": full_result.objective_improvement_pct,
                    "primary_objective_speedup_ratio": full_result.objective_speedup_ratio,
                    "primary_objective_speedup_label": full_result.objective_speedup_label,
                    "wall_time_s": full_result.wall_time_s,
                },
            )

        validated_results_for_goal_by_candidate: dict[int, QuickShortResult] = {
            item.candidate_id: item for item in short_results if _result_is_auto_validated(item)
        }
        for item in full_results:
            validated_results_for_goal_by_candidate[item.candidate_id] = item
        goal_candidate = _select_best_validated_result(
            list(validated_results_for_goal_by_candidate.values())
        )
        if fixed_goal is not None and _fixed_goal_is_reached(goal_candidate):
            fixed_goal_stop_reason = "fixed_goal_reached"
            sink.print(
                "    Stopped: fixed goal reached "
                f"({format_metric_value(_fixed_goal_value(goal_candidate))} target, "
                f"threshold {format_metric_value(fixed_goal)})"
            )
            break

    _emit_tune_event(
        event_emitter,
        "phase.finished",
        {"mode": "spawn", "quick_mode": True, "phase": "spawn.quick_screen", "status": "completed"},
    )

    validated_results_by_candidate: dict[int, QuickShortResult] = {
        item.candidate_id: item for item in short_results if _result_is_auto_validated(item)
    }
    for item in full_results:
        validated_results_by_candidate[item.candidate_id] = item
    validated_results = list(validated_results_by_candidate.values())

    grouped_usefulness: dict[str, QuickSignalUsefulness] = {
        entry.signal_ref: entry for entry in initial_usefulness
    }
    for spec in signal_specs:
        checked_for_signal = [
            window_id
            for window_id in checked_window_ids
            if window_id.startswith(f"{spec.signal_id}-")
        ]
        if not checked_for_signal:
            continue
        grouped_usefulness[spec.signal_ref] = classify_signal_usefulness(
            target_spec=spec,
            short_results=short_results,
            full_results=validated_results,
            defaults=resolved_defaults,
        )
    signal_usefulness = list(grouped_usefulness.values())

    final_selection_basis = baseline_objective_metric if primary_objective else "wall_time"
    selected_result = _select_best_validated_result(validated_results)
    config_find_wall_time_s = baseline_config_find_wall_time_s
    if selected_result is not None:
        config_find_wall_time_s = result_find_wall_times.get(
            selected_result.config_id,
            baseline_config_find_wall_time_s,
        )
    if (
        fixed_goal is not None
        and fixed_goal_stop_reason == ""
        and _fixed_goal_is_reached(selected_result)
    ):
        fixed_goal_stop_reason = "fixed_goal_reached"

    conclusion, conclusion_reason = _conclusion_from_results(
        profiling_available=profiling_available,
        abnormal_windows=ranked_windows,
        checked_window_ids=checked_window_ids,
        short_results=short_results,
        full_results=validated_results,
        selected_result=selected_result,
        budget_exhausted=budget_exhausted,
        fixed_budget_exhausted=fixed_budget_exhausted,
        unfillable_no_action=unfillable_no_action,
        defaults=resolved_defaults,
        signal_usefulness=signal_usefulness,
        degraded_errors=degraded_errors,
    )
    signal_window_groups = _build_signal_window_groups(ranked_windows, checked_window_ids)

    report = QuickModeReport(
        schema_version="syspilot.quick_mode/v3",
        mode="quick",
        generated_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        primary_objective_query=primary_objective or "",
        primary_objective_metric=baseline_objective_metric,
        primary_objective_mode=baseline_objective_mode,
        baseline_summary=baseline_summary,
        abnormal_windows=ranked_windows,
        signal_windows=signal_window_groups,
        short_results=short_results,
        promotions=promotions,
        full_results=full_results,
        signal_usefulness=signal_usefulness,
        final_selection_basis=final_selection_basis,
        selected_candidate_id=selected_result.candidate_id if selected_result else None,
        selected_candidate_label=selected_result.candidate_label if selected_result else "(none)",
        conclusion=conclusion,
        conclusion_reason=conclusion_reason,
        checked_window_ids=checked_window_ids,
        artifact_dir=str(artifact_dir),
        degraded_errors=degraded_errors,
    )
    fixed_budget_stop_reason = ""
    if fixed_budget_state.enabled:
        if unfillable_no_action:
            fixed_budget_stop_reason = "unfillable_no_action"
        elif fixed_budget_exhausted:
            fixed_budget_stop_reason = "fixed_budget_exhausted"
        elif not degraded_errors and (
            candidate_space_exhausted or len(checked_window_ids) == len(ranked_windows)
        ):
            fixed_budget_stop_reason = "candidate_space_exhausted"

    max_iterations_stop_reason = (
        "max_iterations_exhausted" if max_iterations_exhausted else ""
    )

    if fixed_goal is not None and fixed_goal_stop_reason == "" and not _fixed_goal_is_reached(
        selected_result
    ):
        fixed_goal_stop_reason = "max_iterations_exhausted" if max_iterations_exhausted else ""

    report_payload = asdict(report)
    report_payload.update(_elapsed_payload(config_find_wall_time_s))
    report_payload.update(
        {
            "max_iteration_trials_used": max_iteration_trials_used,
            "max_iterations_stop_reason": max_iterations_stop_reason,
        }
    )
    report_payload.update(
        _fixed_budget_payload(
            fixed_budget_state,
            stop_reason=fixed_budget_stop_reason,
        )
    )
    report_payload.update(
        fixed_goal_payload(
            threshold=fixed_goal,
            value=_fixed_goal_value(selected_result),
            objective_mode=baseline_objective_mode,
            stop_reason=fixed_goal_stop_reason,
        )
    )
    _write_json(artifact_dir / "quick_report.json", report_payload)

    sink.table(
        title="Quick Mode Signal/Window Map",
        columns=[("Signal", "green"), ("Ranked Windows", "white")],
        rows=[
            [group.signal_name, _format_signal_window_group(group)]
            for group in signal_window_groups
        ],
    )
    sink.table(
        title="Quick Mode Abnormal Windows",
        columns=[
            ("Signal", "green"),
            ("Window ID", "white"),
            ("Window", "white"),
            ("Severity", "yellow"),
            ("Reason", "cyan"),
        ],
        rows=[
            [
                window.signal_name,
                window.window_id,
                f"{window.start_offset_s:.1f}s-{window.end_offset_s:.1f}s",
                f"{window.severity:.1f}%",
                window.reason,
            ]
            for window in ranked_windows
        ],
    )
    sink.table(
        title="Quick Mode Short Screens",
        columns=[
            ("Signal", "green"),
            ("Window ID", "white"),
            ("Candidate", "white"),
            ("Screen Time", "magenta"),
            ("Signal Change", "yellow"),
            ("Status", "cyan"),
        ],
        rows=[
            [
                result.signal_name,
                result.window_id,
                result.candidate_label,
                f"{result.wall_time_s:.2f}s",
                f"{result.target_signal_improvement_pct:+.1f}%"
                if result.target_signal_improvement_pct is not None
                else "-",
                result.status,
            ]
            for result in short_results
        ],
    )
    sink.table(
        title="Quick Mode Promotions",
        columns=[
            ("Signal", "green"),
            ("Window ID", "white"),
            ("Candidate", "white"),
            ("Promoted", "yellow"),
            ("Reason", "cyan"),
        ],
        rows=[
            [
                decision.signal_name,
                decision.window_id,
                decision.candidate_label,
                "yes" if decision.promoted else "no",
                decision.reason,
            ]
            for decision in promotions
        ],
    )
    sink.table(
        title="Quick Mode Full Validation",
        columns=[
            ("Signal", "green"),
            ("Window ID", "white"),
            ("Candidate", "white"),
            ("Objective", "magenta"),
            ("Objective Ratio", "yellow"),
        ],
        rows=[
            [
                result.signal_name,
                result.window_id,
                result.candidate_label,
                format_metric_value(result.objective_value),
                _format_objective_speedup(result.objective_speedup_ratio),
            ]
            for result in full_results
        ],
    )
    sink.table(
        title="Quick Mode Signal Usefulness",
        columns=[("Signal", "green"), ("Status", "yellow"), ("Notes", "cyan")],
        rows=[[entry.signal_name, entry.status, entry.notes] for entry in signal_usefulness],
    )
    sink.print(f"  Quick-mode report: {artifact_dir / 'quick_report.json'}")
    if selected_result is not None:
        sink.print(
            "\n[bold green]✓ Quick mode selected "
            f"{selected_result.candidate_label} by {final_selection_basis} "
            f"({_format_objective_speedup(selected_result.objective_speedup_ratio)})[/bold green]"
        )
    else:
        sink.print("\n[yellow]Quick mode finished without a validated winning candidate.[/yellow]")
    if degraded_errors:
        sink.print(
            f"  [yellow]Degraded errors preserved in report: {len(degraded_errors)} "
            f"(see {artifact_dir / 'degraded_errors.jsonl'})[/yellow]"
        )
    if fixed_budget_stop_reason == "fixed_budget_exhausted":
        sink.print(
            "  Stopped: fixed budget exhausted after "
            f"{fixed_budget_state.used}/{fixed_budget_state.total} candidate attempts"
        )
    elif fixed_budget_stop_reason == "candidate_space_exhausted":
        sink.print("  Stopped: candidate space exhausted before budget")
    elif fixed_budget_stop_reason == "unfillable_no_action":
        sink.print(
            "  Stopped: no valid action after "
            f"{fixed_budget_state.no_action_reprompts_used} re-prompts"
        )
    if fixed_goal_stop_reason == "fixed_goal_reached":
        sink.print(
            "  Stopped: fixed goal reached "
            f"({format_metric_value(_fixed_goal_value(selected_result))} target, "
            f"threshold {format_metric_value(fixed_goal)})"
        )
    elif fixed_goal_stop_reason == "max_iterations_exhausted":
        sink.print(
            "  Stopped: max iterations exhausted before fixed goal "
            f"({format_metric_value(_fixed_goal_value(selected_result))} target, "
            f"threshold {format_metric_value(fixed_goal)})"
        )
    elif max_iterations_stop_reason == "max_iterations_exhausted":
        sink.print(
            "  Stopped: max candidate trial limit exhausted after "
            f"{max_iteration_trials_used}/{max_iterations} candidate attempts"
        )
    sink.print(
        f"  Total SysPilot wall time: "
        f"{_elapsed_payload(config_find_wall_time_s)['total_wall_time_s']:.2f}s"
    )
    sink.print(f"  Config find wall time: {config_find_wall_time_s:.2f}s")
    sink.print(f"  Conclusion: {conclusion} — {conclusion_reason}")

    _emit_tune_event(
        event_emitter,
        "result.comparison",
        {
            "mode": "spawn",
            "quick_mode": True,
            "selected_candidate_id": report.selected_candidate_id,
            "selected_candidate_label": report.selected_candidate_label,
            "primary_objective_metric": baseline_objective_metric,
            "primary_objective_mode": baseline_objective_mode,
            "baseline_primary_objective": baseline_objective_value,
            "selected_primary_objective": selected_result.objective_value
            if selected_result
            else None,
            "primary_objective_improvement_pct": selected_result.objective_improvement_pct
            if selected_result
            else None,
            "primary_objective_speedup_ratio": selected_result.objective_speedup_ratio
            if selected_result
            else None,
            "primary_objective_speedup_label": selected_result.objective_speedup_label
            if selected_result
            else "",
            "final_selection_basis": final_selection_basis,
            "conclusion": conclusion,
            "artifact_dir": str(artifact_dir),
            "degraded_error_count": len(degraded_errors),
            "stop_reason": fixed_goal_stop_reason
            or fixed_budget_stop_reason
            or max_iterations_stop_reason,
            "max_iteration_trials_used": max_iteration_trials_used,
            "max_iterations_stop_reason": max_iterations_stop_reason,
            **_elapsed_payload(config_find_wall_time_s),
            **fixed_goal_payload(
                threshold=fixed_goal,
                value=_fixed_goal_value(selected_result),
                objective_mode=baseline_objective_mode,
                stop_reason=fixed_goal_stop_reason,
            ),
            **_fixed_budget_payload(
                fixed_budget_state,
                stop_reason=fixed_budget_stop_reason,
            ),
        },
    )
    _emit_tune_event(
        event_emitter,
        "run.finished",
        {
            "mode": "spawn",
            "quick_mode": True,
            "status": conclusion,
            "artifact_dir": str(artifact_dir),
            "selected_candidate_id": report.selected_candidate_id,
            "final_selection_basis": final_selection_basis,
            "conclusion": conclusion,
            "degraded_error_count": len(degraded_errors),
            "stop_reason": fixed_goal_stop_reason
            or fixed_budget_stop_reason
            or max_iterations_stop_reason,
            "max_iteration_trials_used": max_iteration_trials_used,
            "max_iterations_stop_reason": max_iterations_stop_reason,
            **_elapsed_payload(config_find_wall_time_s),
            **fixed_goal_payload(
                threshold=fixed_goal,
                value=_fixed_goal_value(selected_result),
                objective_mode=baseline_objective_mode,
                stop_reason=fixed_goal_stop_reason,
            ),
            **_fixed_budget_payload(
                fixed_budget_state,
                stop_reason=fixed_budget_stop_reason,
            ),
        },
    )


def run_quick_tune_loop(
    *,
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    fixed_budget: int | None,
    fixed_goal: float | None,
    max_iterations: int,
    timeout: int,
    primary_objective: str | None,
    primary_objective_mode: str | None,
    workdir: Path | None,
    memory_high: str | None,
    sink: OutputSink | None,
    event_emitter: TuneEventEmitterProtocol | None,
    structured_context_bundle: dict[str, object] | None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None,
    run_executor: RunExecutor,
    parse_spawn_actions: ParseActions,
    build_spawn_candidates: BuildCandidates,
    extract_hypothesis_profiles: ExtractHypothesisProfiles,
    recommendation_next_action_display: RecommendationActionDisplay,
    extract_inline_env_assignments: ExtractInlineEnvAssignments,
    extract_metrics_from_output: ExtractMetricsFromOutput,
    resolve_primary_objective_metric: ResolvePrimaryObjectiveMetric,
    extract_primary_objective_value: ExtractPrimaryObjectiveValue,
    infer_primary_objective_mode: InferPrimaryObjectiveMode,
    compare_objective_values: CompareObjectiveValues,
    compute_objective_change_pct: ComputeObjectiveChangePct,
    objective_display_name: ObjectiveDisplayName,
    format_metric_value: FormatMetricValue,
    defaults: QuickModeDefaults | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> None:
    asyncio.run(
        run_quick_tune_loop_async(
            agent=agent,
            command=command,
            contracts=contracts,
            manual_hints=manual_hints,
            fixed_budget=fixed_budget,
            fixed_goal=fixed_goal,
            max_iterations=max_iterations,
            timeout=timeout,
            primary_objective=primary_objective,
            primary_objective_mode=primary_objective_mode,
            workdir=str(workdir) if workdir else None,
            memory_high=memory_high,
            sink=sink,
            event_emitter=event_emitter,
            structured_context_bundle=structured_context_bundle,
            initial_hypothesis_profiles=initial_hypothesis_profiles,
            run_executor=run_executor,
            parse_spawn_actions=parse_spawn_actions,
            build_spawn_candidates=build_spawn_candidates,
            extract_hypothesis_profiles=extract_hypothesis_profiles,
            recommendation_next_action_display=recommendation_next_action_display,
            extract_inline_env_assignments=extract_inline_env_assignments,
            extract_metrics_from_output=extract_metrics_from_output,
            resolve_primary_objective_metric=resolve_primary_objective_metric,
            extract_primary_objective_value=extract_primary_objective_value,
            infer_primary_objective_mode=infer_primary_objective_mode,
            compare_objective_values=compare_objective_values,
            compute_objective_change_pct=compute_objective_change_pct,
            objective_display_name=objective_display_name,
            format_metric_value=format_metric_value,
            defaults=defaults,
            excluded_config_names=excluded_config_names,
            tunable_whitelist_names=tunable_whitelist_names,
        )
    )
