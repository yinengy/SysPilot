from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import load_json_object
from .direct_commands import build_direct_command_from_row, rewrite_syspilot_command_value


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json_object(path)


def metric(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("throughput")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def ratio(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return new / old


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def best_successful_row(rows: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
    best_row: dict[str, Any] | None = None
    best_metric: float | None = None
    for row in rows:
        row_metric = metric(row)
        if row is None or not row.get("success") or row_metric is None:
            continue
        if best_metric is None or row_metric > best_metric:
            best_row = row
            best_metric = row_metric
    return best_row


def row_repro_command(workspace_root: Path, row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return build_direct_command_from_row(workspace_root.name, row)


def rewrite_tune_command_with_row(
    benchmark_id: str,
    tune_command: str | None,
    row: Mapping[str, Any] | None,
) -> str:
    command_text = "" if tune_command is None else tune_command.strip()
    if not command_text or row is None:
        return command_text
    inner_command = build_direct_command_from_row(benchmark_id, row)
    return rewrite_syspilot_command_value(command_text, inner_command)


def load_workspace_quick_mode_report(workspace_root: Path) -> dict[str, Any] | None:
    return load_json_if_exists(workspace_root / "quick_mode_report.json")


def load_workspace_direct_tune_report(workspace_root: Path) -> dict[str, Any] | None:
    return load_json_if_exists(workspace_root / "direct_tune_report.json")


def _append_case_section(lines: list[str], *, heading: str, row: dict[str, Any] | None, workspace_root: Path) -> None:
    case_id = None if row is None else str(row.get("case_id", "") or "")
    command = row_repro_command(workspace_root, row)
    lines.extend(
        [
            heading,
            "",
            f"- Case: `{case_id or 'n/a'}`",
            f"- Mode: `{(None if row is None else row.get('mode')) or 'n/a'}`",
            f"- Config summary: `{(None if row is None else row.get('config_summary')) or 'n/a'}`",
            f"- Metric: `{format_metric(metric(row))}`",
            f"- Result artifact: `{(None if row is None else row.get('result_path')) or 'n/a'}`",
            "- Re-run command from repo root:",
            "",
        ]
    )
    if command is not None:
        lines.extend(["```bash", command, "```", ""])
    else:
        lines.extend(["No stable direct command can be derived from the current archived benchmark row.", ""])


def _append_quick_mode_section(
    lines: list[str],
    *,
    quick_mode_report: dict[str, Any],
    best_metric: float | None,
) -> None:
    metric_label = str(quick_mode_report.get("primary_objective_metric") or "objective")
    comparable_to_archived_best = metric_label == "throughput"
    baseline_objective = quick_mode_report.get("baseline_objective_value")
    selected_candidate_label = str(quick_mode_report.get("selected_candidate_label") or "(none)")
    selected_full_run_objective = quick_mode_report.get("selected_full_run_objective")
    selected_full_run_speedup_ratio = quick_mode_report.get("selected_full_run_speedup_ratio")
    run_id = str(quick_mode_report.get("run_id") or "unknown")
    conclusion = str(quick_mode_report.get("conclusion") or "unknown")
    conclusion_reason = str(quick_mode_report.get("conclusion_reason") or "")
    tune_command = str(quick_mode_report.get("tune_command") or "")
    artifact_dir = str(quick_mode_report.get("artifact_dir") or "")
    quick_best_value = selected_full_run_objective
    if not isinstance(quick_best_value, (int, float)):
        quick_best_value = baseline_objective if isinstance(baseline_objective, (int, float)) else None

    lines.extend(["#### Quick-mode baseline rerun", "", "- Tune command from repo root:", ""])
    if tune_command:
        lines.extend(["```bash", tune_command, "```", ""])
    else:
        lines.extend(["No persistent quick-mode tune command was recorded.", ""])
    lines.append(f"- Quick baseline objective: `{format_metric(baseline_objective if isinstance(baseline_objective, (int, float)) else None)}` {metric_label}.")
    lines.append(f"- Selected quick-mode candidate: `{selected_candidate_label}`.")
    if isinstance(selected_full_run_objective, (int, float)) and isinstance(selected_full_run_speedup_ratio, (int, float)):
        lines.append(
            f"- Best validated quick-mode full run: `{format_metric(float(selected_full_run_objective))}` {metric_label} (`{format_ratio(float(selected_full_run_speedup_ratio))}` vs the quick baseline for run `{run_id}`)."
        )
    elif isinstance(selected_full_run_objective, (int, float)):
        lines.append(
            f"- Best validated quick-mode full run: `{format_metric(float(selected_full_run_objective))}` {metric_label} (run `{run_id}`)."
        )
    else:
        lines.append(f"- Best validated quick-mode full run: `n/a` {metric_label} (run `{run_id}`).")
    reason_suffix = f" — {conclusion_reason}" if conclusion_reason else ""
    lines.append(f"- Final quick-mode conclusion: `{conclusion}`{reason_suffix}.")
    if comparable_to_archived_best and isinstance(quick_best_value, (int, float)) and best_metric is not None:
        lines.append(
            f"- Comparison to packaged best archived result: quick-mode best `{format_metric(float(quick_best_value))}` {metric_label} vs archived best `{format_metric(best_metric)}`."
        )
    elif not comparable_to_archived_best:
        lines.append(
            "- Comparison to packaged best archived result: omitted because the quick-mode objective metric differs from the archived throughput metric."
        )
    if artifact_dir:
        lines.append(f"- Artifact directory: `{artifact_dir}`")
    lines.append("")


def _append_direct_tune_section(
    lines: list[str],
    *,
    direct_tune_report: dict[str, Any],
    best_metric: float | None,
) -> None:
    metric_label = str(direct_tune_report.get("primary_objective_metric") or "objective")
    comparable_to_archived_best = metric_label == "throughput"
    baseline_objective = direct_tune_report.get("baseline_objective_value")
    tuned_objective = direct_tune_report.get("selected_full_run_objective")
    tuned_ratio = direct_tune_report.get("selected_full_run_speedup_ratio")
    tune_command = str(direct_tune_report.get("tune_command") or "")
    selected_tuned_command = str(direct_tune_report.get("selected_tuned_command") or "")
    conclusion = str(direct_tune_report.get("conclusion") or "unknown")
    conclusion_reason = str(direct_tune_report.get("conclusion_reason") or "")
    artifact_path = str(direct_tune_report.get("artifact_path") or "")

    lines.extend(["#### Direct SysPilot fixed-baseline rerun", "", "- Tune command from repo root:", ""])
    if tune_command:
        lines.extend(["```bash", tune_command, "```", ""])
    else:
        lines.extend(["No persistent direct-tune command was recorded.", ""])
    lines.append(f"- Direct baseline objective: `{format_metric(baseline_objective if isinstance(baseline_objective, (int, float)) else None)}` {metric_label}.")
    if isinstance(tuned_objective, (int, float)) and isinstance(tuned_ratio, (int, float)):
        lines.append(
            f"- Best validated direct full run: `{format_metric(float(tuned_objective))}` {metric_label} (`{format_ratio(float(tuned_ratio))}` vs the direct baseline)."
        )
    elif isinstance(tuned_objective, (int, float)):
        lines.append(f"- Best validated direct full run: `{format_metric(float(tuned_objective))}` {metric_label}.")
    else:
        lines.append(f"- Best validated direct full run: `n/a` {metric_label}.")
    lines.append("- Selected tuned command:")
    lines.append("")
    if selected_tuned_command:
        lines.extend(["```bash", selected_tuned_command, "```", ""])
    else:
        lines.extend(["No validated tuned command was recorded.", ""])
    reason_suffix = f" — {conclusion_reason}" if conclusion_reason else ""
    lines.append(f"- Final direct-tune conclusion: `{conclusion}`{reason_suffix}.")
    if comparable_to_archived_best and isinstance(tuned_objective, (int, float)) and best_metric is not None:
        lines.append(
            f"- Comparison to packaged best archived result: direct-tune best `{format_metric(float(tuned_objective))}` {metric_label} vs archived best `{format_metric(best_metric)}`."
        )
    elif not comparable_to_archived_best:
        lines.append(
            "- Comparison to packaged best archived result: omitted because the direct-tune objective metric differs from the archived throughput metric."
        )
    if artifact_path:
        lines.append(f"- Report artifact: `{artifact_path}`")
    lines.append("")


def write_benchmark_summary_md(
    path: Path,
    *,
    title: str,
    resource: str,
    contending_libraries: list[str],
    winners: dict[str, dict[str, Any]],
    result_type: str,
    notes: str,
    workspace_root: Path,
    quick_mode_report: dict[str, Any] | None,
    direct_tune_report: dict[str, Any] | None = None,
) -> None:
    baseline = winners.get("baseline")
    app_only = winners.get("application_only")
    library = winners.get("library_aware")
    oracle = winners.get("oracle")
    best = best_successful_row(winners.values())
    best_metric = metric(best)
    lines = [
        f"# {title}",
        "",
        f"- Status: `{result_type}`",
        f"- Resource: {resource}",
        f"- Libraries: {', '.join(contending_libraries)}",
        f"- Performance snapshot: baseline `{format_metric(metric(baseline))}`, application-only `{format_metric(metric(app_only))}`, library-aware `{format_metric(metric(library))}`, oracle `{format_metric(metric(oracle))}`, best archived `{format_metric(best_metric)}`.",
        f"- Best archived speedup vs baseline: `{format_ratio(ratio(best_metric, metric(baseline)))}`.",
        f"- Notes: {notes}",
        "",
    ]
    _append_case_section(lines, heading="#### Baseline", row=baseline, workspace_root=workspace_root)
    _append_case_section(lines, heading="#### Best found", row=best, workspace_root=workspace_root)
    if quick_mode_report is not None:
        quick_mode_with_direct_command = dict(quick_mode_report)
        quick_mode_with_direct_command["tune_command"] = rewrite_tune_command_with_row(
            workspace_root.name,
            quick_mode_report.get("tune_command") if isinstance(quick_mode_report, Mapping) else None,
            baseline,
        )
        _append_quick_mode_section(lines, quick_mode_report=quick_mode_with_direct_command, best_metric=best_metric)
    if direct_tune_report is not None:
        _append_direct_tune_section(lines, direct_tune_report=direct_tune_report, best_metric=best_metric)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
