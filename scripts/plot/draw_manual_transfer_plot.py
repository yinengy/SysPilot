from __future__ import annotations

# pyright: reportAny=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

import argparse
import csv
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "manual_transfer.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "manual_transfer.pdf"
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 14
CELL_LABEL_FONT_SIZE = 14
COLORBAR_LABEL_FONT_SIZE = 14
SOFT_BLUES = LinearSegmentedColormap.from_list(
    "soft_blues",
    plt.colormaps["Blues"](np.linspace(0.08, 0.82, 256)),
)

NAME_MAP = {
    "hf-datasets-tokenizers-dataloader": "HF/PyTorch\nText Data Pipeline",
    "torchdata-fsspec-s3-ddp": "TorchData/S3-DDP\nObject-Store Pipeline",
    "dali-nvjpeg-pytorch-cudnn": "DALI/PyTorch GPU\nDecode–Inference",
}


def load_matrix(input_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise SystemExit(f"Input CSV is empty: {input_path}")
    columns = rows[0][1:]
    index = [row[0] for row in rows[1:]]
    values = np.array([[float(value) for value in row[1:]] for row in rows[1:]], dtype=float)
    if values.shape != (len(index), len(columns)):
        raise SystemExit(f"Input CSV has inconsistent row lengths: {input_path}")
    return index, columns, values


def draw_manual_transfer_plot(input_path: Path, output_path: Path) -> None:
    index, columns, values = load_matrix(input_path)
    display_index = [NAME_MAP.get(name, name) for name in index]
    display_columns = [NAME_MAP.get(name, name) for name in columns]
    row_baseline = np.diag(values)
    if np.any(row_baseline == 0):
        raise SystemExit("Diagonal baseline values must be non-zero for row normalization")

    ratio = values / row_baseline[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    image = ax.imshow(
        values,
        cmap=SOFT_BLUES,
        vmin=float(values.min()),
        vmax=float(values.max()),
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Value (lower is better)", fontsize=COLORBAR_LABEL_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)

    ax.set_xticks(np.arange(values.shape[1]), labels=display_columns)
    ax.set_yticks(np.arange(values.shape[0]), labels=display_index)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor", fontsize=TICK_LABEL_FONT_SIZE)
    plt.setp(ax.get_yticklabels(), fontsize=TICK_LABEL_FONT_SIZE)

    ax.set_xlabel("Target application", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Source manual", fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1, color="black")
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = (float(values.max()) + float(values.min())) / 2
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            if row_index == column_index:
                label = f"{int(values[row_index, column_index])}\n(1.0× row base)\nbaseline"
            else:
                label = f"{int(values[row_index, column_index])}\n({ratio[row_index, column_index]:.1f}× row base)"

            _ = ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] > threshold else "black",
                fontweight="bold" if row_index == column_index else "normal",
                fontsize=CELL_LABEL_FONT_SIZE,
            )

    for index in range(min(values.shape)):
        ax.add_patch(
            Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, linewidth=2, edgecolor="black")
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the manual-transfer matrix plot.")
    _ = parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = cast(Path, args.input).resolve()
    output_path = cast(Path, args.output).resolve()
    draw_manual_transfer_plot(input_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
