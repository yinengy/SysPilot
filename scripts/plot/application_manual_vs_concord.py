import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "application_manual_vs_concord.csv"
pdf_path = script_dir / "application_manual_vs_concord_budget30_paper.pdf"

METHOD_LABELS = {
    "manual": "Application-level Manual",
    "contract": "CONCORD",
}

BENCHMARK_LABELS = {
    "nestedcpu-blas": "NestedCPU\nBLAS",
    "arrow-dask-duckdb-memspill": "Arrow-Dask\nDuckDB",
    "hf-datasets-tokenizers-dataloader": "HF Datasets\nTokenizers",
    "torchdata-fsspec-s3-ddp": "TorchData\nS3-DDP",
    "dali-nvjpeg-pytorch-cudnn": "DALI/NVJPEG\nPyTorch",
}

raw_rows = []
with csv_path.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = set(reader.fieldnames or [])
    score_column = "normalized_score" if "normalized_score" in fieldnames else "speedup"
    for row in reader:
        raw_rows.append({
            "benchmark": row["benchmark"].strip(),
            "mode": row["mode"].strip().lower(),
            "score": float(row[score_column]),
        })

if score_column == "normalized_score":
    rows = raw_rows
else:
    best_scores = {}
    for row in raw_rows:
        benchmark = row["benchmark"]
        best_scores[benchmark] = max(best_scores.get(benchmark, 0.0), row["score"])

    rows = []
    for row in raw_rows:
        best_score = best_scores[row["benchmark"]]
        rows.append({
            "benchmark": row["benchmark"],
            "mode": row["mode"],
            "score": row["score"] / best_score if best_score else 0.0,
        })

benchmarks = []
modes = []
scores = {}
for row in rows:
    b = row["benchmark"]
    m = row["mode"]
    if b not in scores:
        benchmarks.append(b)
        scores[b] = {}
    if m not in modes:
        modes.append(m)
    scores[b][m] = row["score"]

if "manual" in modes and "contract" in modes:
    benchmarks = sorted(
        benchmarks,
        key=lambda b: scores[b].get("contract", 0) - scores[b].get("manual", 0),
        reverse=True,
    )

avg_scores = {}
for m in modes:
    vals = [scores[b][m] for b in benchmarks if m in scores[b]]
    avg_scores[m] = sum(vals) / len(vals)

benchmarks_with_avg = benchmarks + ["__avg__"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

n_groups = len(benchmarks_with_avg)
n_modes = len(modes)
x = list(range(n_groups))
group_width = 0.70
bar_width = group_width / max(n_modes, 1)

fig, ax = plt.subplots(figsize=(7.4, 3.05))

for i, mode in enumerate(modes):
    xpos = [xi - group_width / 2 + bar_width / 2 + i * bar_width for xi in x]
    values = []
    for b in benchmarks_with_avg:
        if b == "__avg__":
            values.append(avg_scores[mode])
        else:
            values.append(scores[b].get(mode, 0.0))

    bars = ax.bar(
        xpos,
        values,
        width=bar_width,
        label=METHOD_LABELS.get(mode, mode.replace("_", " ").title()),
        edgecolor="black",
        linewidth=0.35,
    )
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

xticklabels = [
    "Avg." if b == "__avg__" else BENCHMARK_LABELS.get(b, "\n".join(textwrap.wrap(b, width=12)))
    for b in benchmarks_with_avg
]

ax.set_ylabel("Normalized score")
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.set_ylim(0, 1.08)

ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

ax.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.35)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(len(benchmarks) - 0.5, linestyle="--", linewidth=0.7, alpha=0.45)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=min(n_modes, 4),
    frameon=False,
    handlelength=1.5,
    columnspacing=1.6,
)

plt.tight_layout(pad=0.35)
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()
