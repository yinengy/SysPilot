from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
import csv
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "best_so_far_performance_over_iterations.pdf"
DEFAULT_RAW_CSV = SCRIPT_DIR / "best_so_far_performance_over_iterations_raw.csv"
MAX_PLOT_ITERATION = 30
AVERAGE_BENCHMARK = "__average__"

TUNER_LABELS = {
    "syspilot_contract": "CONCORD (Lib Contract-Guided)",
    "syspilot_manual": "CONCORD (App Manual-Guided)",
    "smac3": "SMAC3",
    "opentuner": "OpenTuner",
}

TUNER_ORDER = ["syspilot_contract", "syspilot_manual", "smac3", "opentuner"]

BENCHMARK_LABELS = {
    "nestedcpu-blas": "Joblib/BLAS\nNested CPU Contention",
    "arrow-dask-duckdb-memspill": "Dask/DuckDB\nColumnar Memory-Spill",
    "hf-datasets-tokenizers-dataloader": "HF/PyTorch\nText Data Pipeline",
    "torchdata-fsspec-s3-ddp": "TorchData/S3-DDP\nObject-Store Pipeline",
    "dali-nvjpeg-pytorch-cudnn": "DALI/PyTorch\nGPU Decode–Inference",
}

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


def benchmark_label(benchmark: str) -> str:
    if benchmark == AVERAGE_BENCHMARK:
        return "Average\n(all benchmarks)"
    return BENCHMARK_LABELS.get(benchmark, "\n".join(textwrap.wrap(benchmark, width=16)))


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


def collect_plot_data(raw_csv: Path, max_iteration: int) -> tuple[list[str], dict[str, list[PlotSeries]]]:
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
    return benchmarks, benchmark_rows


def plot(raw_csv: Path, output_path: Path, max_iteration: int) -> None:
    benchmarks, benchmark_rows = collect_plot_data(raw_csv, max_iteration)
    if not benchmarks:
        raise SystemExit(f"No plot data found in {raw_csv}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ncols = 3
    nrows = math.ceil(len(benchmarks) / ncols)
    fig_obj, axes_obj = plt.subplots(
        nrows,
        ncols,
        figsize=(8.8, 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig = fig_obj
    axes_grid = cast(list[list[Axes]], axes_obj.tolist())
    axes_list = [ax for row in axes_grid for ax in row]

    colors = {
        "syspilot_contract": "#1f77b4",
        "syspilot_manual": "#ff7f0e",
        "smac3": "#2ca02c",
        "opentuner": "#9467bd",
    }
    for ax, benchmark in zip(axes_list, benchmarks):
        rows = sorted(
            benchmark_rows[benchmark],
            key=lambda row: TUNER_ORDER.index(row.tuner) if row.tuner in TUNER_ORDER else 99,
        )
        for row in rows:
            tuner = row.tuner
            plotted_points = [
                (x_value, y_value)
                for x_value, y_value in zip(row.x, row.y)
                if x_value <= max_iteration
            ]
            if not plotted_points:
                continue
            _ = ax.plot(
                [x_value for x_value, _ in plotted_points],
                [y_value for _, y_value in plotted_points],
                label=TUNER_LABELS.get(tuner, tuner.replace("_", " ").title()),
                color=colors.get(tuner),
                linestyle="-",
                linewidth=1.7,
                marker="o",
                markersize=2.4,
                markeredgewidth=0,
            )
        _ = ax.set_xlim(0, max_iteration)
        _ = ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelbottom=True, pad=1)
        _ = ax.set_xlabel("iteration", labelpad=0)
        _ = ax.text(
            0.5,
            -0.23,
            benchmark_label(benchmark),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.grid(axis="both", linestyle="-", linewidth=0.35, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_list[len(benchmarks) :]:
        ax.set_visible(False)

    for row_index in range(nrows):
        _ = axes_list[row_index * ncols].set_ylabel("Normalized Score")

    handles, labels = axes_list[0].get_legend_handles_labels()
    _ = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(labels), 4),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99), w_pad=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("raw_csv", nargs="?", type=Path, default=DEFAULT_RAW_CSV)
    _ = parser.add_argument("--max-iteration", type=int, default=MAX_PLOT_ITERATION)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw_csv = cast(Path, args.raw_csv).resolve()
    output = cast(Path, args.output).resolve()
    max_iteration: int = cast(int, args.max_iteration)
    plot(raw_csv, output, max_iteration)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
