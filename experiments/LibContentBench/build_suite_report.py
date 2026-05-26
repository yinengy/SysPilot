#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.LibContentBench.libcontentbench.manifest import BenchmarkEntry, load_manifest
from experiments.LibContentBench.libcontentbench.reporting import (
    format_metric,
    format_ratio,
    load_json_if_exists,
    load_workspace_direct_tune_report,
    load_workspace_quick_mode_report,
    ratio,
    repo_relative_case_command,
    row_repro_command,
)


SUITE_ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = SUITE_ROOT / "suite_manifest.json"


def metric_from_row(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("throughput")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def winners_by_mode(comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = comparison.get("winners_by_mode", {})
    if not isinstance(payload, dict):
        return {}
    winners: dict[str, dict[str, Any]] = {}
    for mode, row in payload.items():
        if isinstance(mode, str) and isinstance(row, dict):
            winners[mode] = row
    return winners


def fallback_row_for_mode(comparison: dict[str, Any], mode: str) -> dict[str, Any] | None:
    cases = comparison.get("cases", [])
    if not isinstance(cases, list):
        return None
    for row in cases:
        if isinstance(row, dict) and str(row.get("mode", "")) == mode:
            return row
    return None


def row_for_mode(comparison: dict[str, Any], mode: str) -> dict[str, Any] | None:
    return winners_by_mode(comparison).get(mode) or fallback_row_for_mode(comparison, mode)


def best_successful_row(comparison: dict[str, Any]) -> dict[str, Any] | None:
    candidates = list(winners_by_mode(comparison).values())
    if not candidates:
        raw_cases = comparison.get("cases", [])
        if isinstance(raw_cases, list):
            candidates = [row for row in raw_cases if isinstance(row, dict)]
    successful = [row for row in candidates if row.get("success") and metric_from_row(row) is not None]
    if not successful:
        return None
    return max(successful, key=lambda row: metric_from_row(row) or float("-inf"))


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in load_manifest(MANIFEST_PATH):
        workspace = SUITE_ROOT / entry.relative_path
        comparison = load_json_if_exists(workspace / "comparison_report.json") or {}
        acceptance = load_json_if_exists(workspace / "acceptance_report.json") or {}
        quick_mode = load_workspace_quick_mode_report(workspace)
        direct_tune = load_workspace_direct_tune_report(workspace)
        baseline_row = row_for_mode(comparison, "baseline")
        best_row = best_successful_row(comparison)
        baseline_metric = acceptance.get("bad_baseline_metric")
        best_metric = metric_from_row(best_row)
        baseline_case_id = None if baseline_row is None else str(baseline_row.get("case_id", "") or "")
        best_case_id = None if best_row is None else str(best_row.get("case_id", "") or "")
        rows.append(
            {
                "benchmark": entry.benchmark_id,
                "workspace": entry.relative_path,
                "resource": comparison.get("resource", "unknown"),
                "contending_libraries": ", ".join(comparison.get("contending_libraries", [])),
                "bad_baseline_metric": baseline_metric,
                "application_only_metric": acceptance.get("application_only_metric"),
                "library_aware_metric": acceptance.get("library_aware_metric"),
                "oracle_best_metric": acceptance.get("oracle_best_metric"),
                "speedup_vs_baseline": acceptance.get("speedup_vs_baseline"),
                "speedup_vs_application_only": acceptance.get("speedup_vs_application_only"),
                "result_type": acceptance.get("result_type", "blocked" if entry.optional else "no_improvement"),
                "notes": acceptance.get("notes", "No summarized results available yet."),
                "quick_mode": quick_mode,
                "direct_tune": direct_tune,
                "baseline_case_id": baseline_case_id,
                "baseline_mode": None if baseline_row is None else baseline_row.get("mode"),
                "baseline_config_summary": None if baseline_row is None else baseline_row.get("config_summary"),
                "baseline_result_path": None if baseline_row is None else baseline_row.get("result_path"),
                "baseline_command": row_repro_command(workspace, baseline_row) or repo_relative_case_command(workspace, case_id=baseline_case_id, python_path=entry.python_path, matrix_name=entry.main_matrix),
                "best_case_id": best_case_id,
                "best_mode": None if best_row is None else best_row.get("mode"),
                "best_metric": best_metric,
                "best_config_summary": None if best_row is None else best_row.get("config_summary"),
                "best_result_path": None if best_row is None else best_row.get("result_path"),
                "best_command": row_repro_command(workspace, best_row) or repo_relative_case_command(workspace, case_id=best_case_id, python_path=entry.python_path, matrix_name=entry.main_matrix),
                "best_speedup_vs_baseline": ratio(best_metric, baseline_metric if isinstance(baseline_metric, (int, float)) else None),
            }
        )
    return rows


def append_quick_mode_section(lines: list[str], row: dict[str, Any]) -> None:
    quick_mode = row.get("quick_mode")
    if not isinstance(quick_mode, dict):
        return
    metric_label = str(quick_mode.get("primary_objective_metric") or "objective")
    comparable_to_archived_best = metric_label == "throughput"
    baseline_objective = quick_mode.get("baseline_objective_value")
    selected_label = str(quick_mode.get("selected_candidate_label") or "(none)")
    selected_full = quick_mode.get("selected_full_run_objective")
    selected_ratio = quick_mode.get("selected_full_run_speedup_ratio")
    conclusion = str(quick_mode.get("conclusion") or "unknown")
    conclusion_reason = str(quick_mode.get("conclusion_reason") or "")
    run_id = str(quick_mode.get("run_id") or "unknown")
    artifact_dir = str(quick_mode.get("artifact_dir") or "")
    tune_command = str(quick_mode.get("tune_command") or "")
    quick_best = selected_full if isinstance(selected_full, (int, float)) else baseline_objective if isinstance(baseline_objective, (int, float)) else None
    lines.extend(["#### Quick-mode baseline rerun", "", "- Tune command from repo root:", ""])
    if tune_command:
        lines.extend(["```bash", tune_command, "```", ""])
    else:
        lines.extend(["No persistent quick-mode tune command was recorded.", ""])
    lines.append(f"- Quick baseline objective: `{format_metric(baseline_objective if isinstance(baseline_objective, (int, float)) else None)}` {metric_label}.")
    lines.append(f"- Selected quick-mode candidate: `{selected_label}`.")
    if isinstance(selected_full, (int, float)) and isinstance(selected_ratio, (int, float)):
        lines.append(
            f"- Best validated quick-mode full run: `{format_metric(float(selected_full))}` {metric_label} (`{format_ratio(float(selected_ratio))}` vs the quick baseline for run `{run_id}`)."
        )
    elif isinstance(selected_full, (int, float)):
        lines.append(f"- Best validated quick-mode full run: `{format_metric(float(selected_full))}` {metric_label} (run `{run_id}`).")
    else:
        lines.append(f"- Best validated quick-mode full run: `n/a` {metric_label} (run `{run_id}`).")
    reason_suffix = f" — {conclusion_reason}" if conclusion_reason else ""
    lines.append(f"- Final quick-mode conclusion: `{conclusion}`{reason_suffix}.")
    if comparable_to_archived_best and isinstance(quick_best, (int, float)) and isinstance(row.get("best_metric"), (int, float)):
        lines.append(
            f"- Comparison to packaged best archived result: quick-mode best `{format_metric(float(quick_best))}` {metric_label} vs archived best `{format_metric(float(row['best_metric']))}`."
        )
    elif not comparable_to_archived_best:
        lines.append(
            "- Comparison to packaged best archived result: omitted because the quick-mode objective metric differs from the archived throughput metric."
        )
    if artifact_dir:
        lines.append(f"- Artifact directory: `{artifact_dir}`")
    lines.append("")


def append_direct_tune_section(lines: list[str], row: dict[str, Any]) -> None:
    direct_tune = row.get("direct_tune")
    if not isinstance(direct_tune, dict):
        return
    metric_label = str(direct_tune.get("primary_objective_metric") or "objective")
    comparable_to_archived_best = metric_label == "throughput"
    baseline_objective = direct_tune.get("baseline_objective_value")
    tuned_objective = direct_tune.get("selected_full_run_objective")
    tuned_ratio = direct_tune.get("selected_full_run_speedup_ratio")
    conclusion = str(direct_tune.get("conclusion") or "unknown")
    conclusion_reason = str(direct_tune.get("conclusion_reason") or "")
    tune_command = str(direct_tune.get("tune_command") or "")
    selected_tuned_command = str(direct_tune.get("selected_tuned_command") or "")
    artifact_path = str(direct_tune.get("artifact_path") or "")
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
    if comparable_to_archived_best and isinstance(tuned_objective, (int, float)) and isinstance(row.get("best_metric"), (int, float)):
        lines.append(
            f"- Comparison to packaged best archived result: direct-tune best `{format_metric(float(tuned_objective))}` {metric_label} vs archived best `{format_metric(float(row['best_metric']))}`."
        )
    elif not comparable_to_archived_best:
        lines.append(
            "- Comparison to packaged best archived result: omitted because the direct-tune objective metric differs from the archived throughput metric."
        )
    if artifact_path:
        lines.append(f"- Report artifact: `{artifact_path}`")
    lines.append("")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "resource",
        "contending_libraries",
        "bad_baseline_metric",
        "application_only_metric",
        "library_aware_metric",
        "oracle_best_metric",
        "speedup_vs_baseline",
        "speedup_vs_application_only",
        "result_type",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LibContentBench Summary",
        "",
        "This report aggregates benchmark-local comparison artifacts and highlights the baseline versus best archived run for each benchmark.",
        "",
        "| Benchmark | Resource | Result | Baseline | Best found | Best vs baseline |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['benchmark']} | {row['resource']} | {row['result_type']} | "
            f"{format_metric(row['bad_baseline_metric'])} | {format_metric(row.get('best_metric'))} | {format_ratio(row.get('best_speedup_vs_baseline'))} |"
        )
    lines.append("")
    lines.append("## Benchmark details")
    lines.append("")
    for row in rows:
        lines.extend(
            [
                f"### {row['benchmark']}",
                "",
                f"- Status: `{row['result_type']}`",
                f"- Resource: {row['resource']}",
                f"- Performance snapshot: baseline `{format_metric(row['bad_baseline_metric'])}`, application-only `{format_metric(row['application_only_metric'])}`, library-aware `{format_metric(row['library_aware_metric'])}`, oracle `{format_metric(row['oracle_best_metric'])}`, best archived `{format_metric(row.get('best_metric'))}`.",
                f"- Best archived speedup vs baseline: `{format_ratio(row.get('best_speedup_vs_baseline'))}`.",
                f"- Notes: {row['notes']}",
                "",
                "#### Baseline",
                "",
                f"- Case: `{row.get('baseline_case_id') or 'n/a'}`",
                f"- Mode: `{row.get('baseline_mode') or 'n/a'}`",
                f"- Config summary: `{row.get('baseline_config_summary') or 'n/a'}`",
                f"- Metric: `{format_metric(row['bad_baseline_metric'])}`",
                f"- Result artifact: `{row.get('baseline_result_path') or 'n/a'}`",
                "- Re-run command from repo root:",
                "",
            ]
        )
        if row.get("baseline_command"):
            lines.extend(["```bash", str(row["baseline_command"]), "```", ""])
        else:
            lines.extend(["No stable baseline command can be derived from the current matrix metadata.", ""])
        lines.extend(
            [
                "#### Best found",
                "",
                f"- Case: `{row.get('best_case_id') or 'n/a'}`",
                f"- Mode: `{row.get('best_mode') or 'n/a'}`",
                f"- Config summary: `{row.get('best_config_summary') or 'n/a'}`",
                f"- Metric: `{format_metric(row.get('best_metric'))}`",
                f"- Result artifact: `{row.get('best_result_path') or 'n/a'}`",
                "- Re-run command from repo root:",
                "",
            ]
        )
        if row.get("best_command"):
            lines.extend(["```bash", str(row["best_command"]), "```", ""])
        else:
            lines.extend(["No stable best command can be derived from the current matrix metadata.", ""])
        append_quick_mode_section(lines, row)
        append_direct_tune_section(lines, row)
    lines.append(
        "Library-level information helps because it exposes hidden knobs from individual libraries that application-only tuning cannot see."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    result_table = SUITE_ROOT / "result_table.csv"
    suite_report = SUITE_ROOT / "suite_report.json"
    summary_md = SUITE_ROOT / "summary.md"
    write_csv(result_table, rows)
    suite_report.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(summary_md, rows)
    print(json.dumps({"result_table": str(result_table), "suite_report": str(suite_report), "summary": str(summary_md)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
