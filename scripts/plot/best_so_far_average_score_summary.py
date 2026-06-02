from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_CSV = SCRIPT_DIR / "best_so_far_performance_over_iterations_raw.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "best_so_far_average_score_summary.csv"
MAX_PLOT_ITERATION = 30
AVERAGE_BENCHMARK = "__average__"
THRESHOLDS = [0.7, 0.8, 0.9]
ITERATIONS = [10, 20, 30]

TUNER_LABELS = {
    "syspilot_contract": "CONCORD (Lib Contract-Guided)",
    "syspilot_manual": "CONCORD (App Manual-Guided)",
    "smac3": "SMAC3",
    "opentuner": "OpenTuner",
}
TUNER_ORDER = ["syspilot_contract", "syspilot_manual", "smac3", "opentuner"]
BENCHMARK_ORDER = [
    "nestedcpu-blas",
    "arrow-dask-duckdb-memspill",
    "hf-datasets-tokenizers-dataloader",
    "torchdata-fsspec-s3-ddp",
    "dali-nvjpeg-pytorch-cudnn",
]


@dataclass(frozen=True)
class PlotSeries:
    tuner: str
    x: list[int]
    y: list[float]


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def normalized(value: float, baseline_floor: float, best_ceiling: float) -> float:
    denominator = best_ceiling - baseline_floor
    if denominator <= 0:
        return 0.0
    return (value - baseline_floor) / denominator


def latest_value_at(series: PlotSeries, iteration: int) -> float | None:
    if iteration == 0:
        return 0.0
    latest: float | None = None
    for x_value, y_value in zip(series.x, series.y):
        if x_value > iteration:
            break
        latest = y_value
    return latest


def add_average_series(
    benchmarks: list[str], benchmark_rows: dict[str, list[PlotSeries]], max_iteration: int
) -> None:
    tuners = sorted(
        {series.tuner for benchmark in benchmarks for series in benchmark_rows[benchmark]},
        key=lambda tuner: TUNER_ORDER.index(tuner) if tuner in TUNER_ORDER else 99,
    )
    average_rows: list[PlotSeries] = []
    for tuner in tuners:
        x_values: list[int] = []
        y_values: list[float] = []
        for iteration in range(max_iteration + 1):
            values: list[float] = []
            for benchmark in benchmarks:
                series = next(
                    (candidate for candidate in benchmark_rows[benchmark] if candidate.tuner == tuner),
                    None,
                )
                if series is None:
                    continue
                value = latest_value_at(series, iteration)
                if value is not None:
                    values.append(value)
            if values:
                x_values.append(iteration)
                y_values.append(sum(values) / len(values))
        if x_values:
            average_rows.append(PlotSeries(tuner=tuner, x=x_values, y=y_values))
    if average_rows:
        benchmark_rows[AVERAGE_BENCHMARK] = average_rows
        benchmarks.append(AVERAGE_BENCHMARK)


def collect_plot_data(raw_csv: Path, max_iteration: int) -> dict[str, list[PlotSeries]]:
    with raw_csv.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = [dict(row) for row in csv.DictReader(handle)]

    baselines: dict[str, list[float]] = defaultdict(list)
    final_bests: dict[str, list[float]] = defaultdict(list)
    for row in raw_rows:
        benchmark = row["benchmark"]
        baseline = parse_float(row.get("baseline_metric", ""))
        best = parse_float(row.get("final_best_metric", ""))
        if baseline is not None:
            baselines[benchmark].append(baseline)
        if best is not None:
            final_bests[benchmark].append(best)

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped_rows[(row["benchmark"], row["tuner"])].append(row)

    benchmark_rows: dict[str, list[PlotSeries]] = defaultdict(list)
    for (benchmark, tuner), rows in grouped_rows.items():
        if not baselines[benchmark] or not final_bests[benchmark]:
            continue
        baseline_floor = min(baselines[benchmark])
        best_ceiling = max(final_bests[benchmark])
        raw_series: list[tuple[int, float]] = []
        for row in rows:
            trial_text = row.get("trial", "").strip()
            current_best = parse_float(row.get("current_best_metric", ""))
            if current_best is None or not trial_text:
                continue
            try:
                trial = int(float(trial_text))
            except ValueError:
                continue
            raw_series.append((trial, current_best))
        raw_series.sort(key=lambda item: item[0])
        if not raw_series:
            continue
        benchmark_rows[benchmark].append(
            PlotSeries(
                tuner=tuner,
                x=[0] + [trial for trial, _ in raw_series if trial != 0],
                y=[0.0]
                + [
                    normalized(value, baseline_floor, best_ceiling)
                    for trial, value in raw_series
                    if trial != 0
                ],
            )
        )

    benchmarks = [benchmark for benchmark in BENCHMARK_ORDER if benchmark in benchmark_rows]
    benchmarks.extend(sorted(set(benchmark_rows) - set(benchmarks)))
    add_average_series(benchmarks, benchmark_rows, max_iteration)
    return benchmark_rows


def threshold_iteration(x_values: list[int], y_values: list[float], threshold: float) -> int | None:
    for x_value, y_value in zip(x_values, y_values):
        if y_value >= threshold:
            return x_value
    return None


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_iteration(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def build_summary_rows(raw_csv: Path, max_iteration: int) -> list[dict[str, str]]:
    benchmark_rows = collect_plot_data(raw_csv, max_iteration)
    average_series = {series.tuner: series for series in benchmark_rows.get(AVERAGE_BENCHMARK, [])}
    rows: list[dict[str, str]] = []
    for tuner in TUNER_ORDER:
        series = average_series.get(tuner)
        if series is None:
            continue
        row = {
            "tuner": tuner,
            "label": TUNER_LABELS.get(tuner, tuner.replace("_", " ").title()),
        }
        for threshold in THRESHOLDS:
            row[f"iterations_to_{threshold:.1f}"] = format_iteration(
                threshold_iteration(series.x, series.y, threshold)
            )
        for iteration in ITERATIONS:
            row[f"normalized_avg_score_at_{iteration}"] = format_float(latest_value_at(series, iteration))
        rows.append(row)
    return rows


def write_summary_csv(raw_csv: Path, output: Path, max_iteration: int) -> None:
    rows = build_summary_rows(raw_csv, max_iteration)
    if not rows:
        raise SystemExit(f"No average plot series found in {raw_csv}")
    fieldnames = [
        "tuner",
        "label",
        "iterations_to_0.7",
        "iterations_to_0.8",
        "iterations_to_0.9",
        "normalized_avg_score_at_10",
        "normalized_avg_score_at_20",
        "normalized_avg_score_at_30",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export average normalized best-so-far score milestones used by the plot."
    )
    _ = parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--max-iteration", type=int, default=MAX_PLOT_ITERATION)
    args = parser.parse_args()
    raw_csv = cast(Path, args.raw_csv).resolve()
    output = cast(Path, args.output).resolve()
    max_iteration = cast(int, args.max_iteration)
    write_summary_csv(raw_csv, output, max_iteration)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
