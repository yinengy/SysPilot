from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "vlcs_autotuners_budget60_20260602T224228Z"
DEFAULT_PROCESSED_CSV = SCRIPT_DIR / "vlcs_best_so_far_processed.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "vlcs_best_so_far_performance_over_iterations.pdf"

TUNER_LABELS = {
    "contract": "CONCORD (Lib Contract-Guided)",
    "smac3": "SMAC3",
    "opentuner": "OpenTuner",
}

TUNER_ORDER = ["contract", "smac3", "opentuner"]

COLORS = {
    "contract": "#1f77b4",
    "smac3": "#2ca02c",
    "opentuner": "#9467bd",
}

CARRIED_FORWARD_TUNERS = {"contract"}

PROCESSED_FIELDS = [
    "benchmark",
    "tuner",
    "budget",
    "trial",
    "metric",
    "current_best_metric",
    "speedup_vs_baseline",
    "baseline_metric",
    "final_best_metric",
    "state",
    "source_csv",
    "config",
]


@dataclass(frozen=True)
class SeriesPoint:
    trial: int
    speedup: float


@dataclass(frozen=True)
class PlotSeries:
    tuner: str
    points: list[SeriesPoint]


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


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def tuner_sort_key(tuner: str) -> int:
    if tuner in TUNER_ORDER:
        return TUNER_ORDER.index(tuner)
    return len(TUNER_ORDER)


def read_summary(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise SystemExit(f"No summary rows found in {summary_csv}")
    return rows


def processed_row(
    *,
    tuner: str,
    budget: str,
    trial: int,
    metric: str,
    current_best: float,
    baseline: float,
    final_best: float,
    state: str,
    source_csv: str,
    config: str,
) -> dict[str, str]:
    return {
        "benchmark": "vlcs",
        "tuner": tuner,
        "budget": budget,
        "trial": str(trial),
        "metric": metric,
        "current_best_metric": f"{current_best:.15g}",
        "speedup_vs_baseline": f"{baseline / current_best:.15g}",
        "baseline_metric": f"{baseline:.15g}",
        "final_best_metric": f"{final_best:.15g}",
        "state": state,
        "source_csv": source_csv,
        "config": config,
    }


def build_processed_csv(run_dir: Path, output_csv: Path) -> None:
    summary_rows = read_summary(run_dir / "summary.csv")
    processed_rows: list[dict[str, str]] = []
    comparison_budget = max(
        (budget for row in summary_rows if (budget := parse_int(row.get("budget", ""))) is not None),
        default=0,
    )
    baseline_metric = max(
        (
            baseline
            for row in summary_rows
            if (baseline := parse_float(row.get("baseline_metric", ""))) is not None
        ),
        default=None,
    )
    if baseline_metric is None:
        raise SystemExit(f"Missing baseline metrics in {run_dir / 'summary.csv'}")

    for summary_row in sorted(summary_rows, key=lambda row: tuner_sort_key(row["tuner"])):
        tuner = summary_row["tuner"]
        budget = summary_row.get("budget", "")
        source_rel = summary_row["best_series_csv"]
        source_csv = run_dir / source_rel
        final_best = parse_float(summary_row.get("best_metric", ""))
        if final_best is None:
            raise SystemExit(f"Missing best metric for {tuner} in {run_dir / 'summary.csv'}")
        baseline = baseline_metric

        with source_csv.open(newline="", encoding="utf-8-sig") as handle:
            source_rows = [dict(row) for row in csv.DictReader(handle)]

        max_trial = 0
        for source_row in source_rows:
            trial = parse_int(source_row.get("trial", ""))
            current_best = parse_float(source_row.get("current_best_metric", ""))
            if trial is None or current_best is None:
                continue
            max_trial = max(max_trial, trial)
            current_best = min(current_best, baseline)
            processed_rows.append(
                processed_row(
                    tuner=tuner,
                    budget=budget,
                    trial=trial,
                    metric=source_row.get("metric", ""),
                    current_best=current_best,
                    baseline=baseline,
                    final_best=final_best,
                    state=source_row.get("state", ""),
                    source_csv=source_rel,
                    config=source_row.get("config", ""),
                )
            )

        if tuner in CARRIED_FORWARD_TUNERS and comparison_budget > max_trial:
            for trial in range(max_trial + 1, comparison_budget + 1):
                processed_rows.append(
                    processed_row(
                        tuner=tuner,
                        budget=budget,
                        trial=trial,
                        metric="",
                        current_best=final_best,
                        baseline=baseline,
                        final_best=final_best,
                        state="CARRIED_FORWARD",
                        source_csv=source_rel,
                        config=f"carried forward best from trial {max_trial}",
                    )
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_FIELDS)
        writer.writeheader()
        writer.writerows(processed_rows)


def collect_plot_data(processed_csv: Path) -> list[PlotSeries]:
    with processed_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    grouped: dict[str, list[SeriesPoint]] = {}
    for row in rows:
        tuner = row["tuner"]
        trial = parse_int(row.get("trial", ""))
        speedup = parse_float(row.get("speedup_vs_baseline", ""))
        if trial is None or speedup is None:
            continue
        grouped.setdefault(tuner, []).append(SeriesPoint(trial=trial, speedup=speedup))

    plot_series: list[PlotSeries] = []
    for tuner, points in grouped.items():
        points.sort(key=lambda point: point.trial)
        plot_series.append(PlotSeries(tuner=tuner, points=[SeriesPoint(0, 1.0)] + points))
    plot_series.sort(key=lambda series: tuner_sort_key(series.tuner))
    if not plot_series:
        raise SystemExit(f"No plot data found in {processed_csv}")
    return plot_series


def plot(processed_csv: Path, output_path: Path, max_iteration: int | None) -> None:
    series_rows = collect_plot_data(processed_csv)
    max_trial = max(point.trial for series in series_rows for point in series.points)
    plot_max_iteration = max_iteration if max_iteration is not None else max_trial
    max_speedup = max(point.speedup for series in series_rows for point in series.points)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig_obj, ax_obj = plt.subplots(figsize=(6.2, 3.6))
    fig = fig_obj
    ax = ax_obj

    for series in series_rows:
        plotted_points = [point for point in series.points if point.trial <= plot_max_iteration]
        if not plotted_points:
            continue
        _ = ax.plot(
            [point.trial for point in plotted_points],
            [point.speedup for point in plotted_points],
            label=TUNER_LABELS.get(series.tuner, series.tuner.replace("_", " ").title()),
            color=COLORS.get(series.tuner),
            linestyle="-",
            linewidth=1.9,
            marker="o",
            markersize=2.8,
            markeredgewidth=0,
        )

    _ = ax.set_xlabel("candidate trial")
    _ = ax.set_ylabel("speedup over even partition baseline")
    _ = ax.set_xlim(0, plot_max_iteration)
    _ = ax.set_ylim(1.0, max(1.25, max_speedup + 0.03))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2fx"))
    ax.grid(axis="both", linestyle="-", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _ = ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    _ = parser.add_argument("--processed-csv", type=Path, default=DEFAULT_PROCESSED_CSV)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--max-iteration", type=int, default=None)
    _ = parser.add_argument(
        "--use-existing-processed-csv",
        action="store_true",
        help="Skip regenerating the processed CSV and plot the existing processed CSV.",
    )
    args = parser.parse_args()

    run_dir = cast(Path, args.run_dir).resolve()
    processed_csv = cast(Path, args.processed_csv).resolve()
    output = cast(Path, args.output).resolve()
    max_iteration = cast(int | None, args.max_iteration)

    if not cast(bool, args.use_existing_processed_csv):
        build_processed_csv(run_dir, processed_csv)
    plot(processed_csv, output, max_iteration)
    print(processed_csv)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
