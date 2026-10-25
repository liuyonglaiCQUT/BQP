"""Render the algorithm-level BQP confidence intervals for IEEE/TKDE.

This script only changes presentation.  It reads the existing algorithm-level
summary produced by ``run_bqp_architecture_extension.py`` and does not rerun
training, evaluation, or resampling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


Z_975 = 1.959963984540054
DATASETS = ("FB15k237", "WN18RR")
DISPLAY_DATASET = {"FB15k237": "FB15k-237", "WN18RR": "WN18RR"}
DISPLAY_MODEL = {
    "complex": "ComplEx",
    "distmult": "DistMult",
    "cp": "CP",
    "rotate": "RotatE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results_bqp/architecture_extension/algorithm_level_comparisons.csv"),
        help="Algorithm-level comparison CSV produced by the BQP experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_bqp/architecture_extension/figures/fig_algorithm_level_comparisons_tkde_v2"),
        help="Output path without an extension; both PDF and PNG are written.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [dataset for dataset in DATASETS if not any(row["dataset"] == dataset for row in rows)]
    if missing:
        raise ValueError(f"Missing datasets in {path}: {', '.join(missing)}")
    return rows


def label(row: dict[str, str]) -> str:
    return f"{DISPLAY_MODEL[row['model_a']]} − {DISPLAY_MODEL[row['model_b']]}"


def add_intervals(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    *,
    se_column: str,
    offset: float,
    color: str,
    legend_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.asarray([float(row["delta_mrr"]) for row in rows])
    standard_errors = np.asarray([float(row[se_column]) for row in rows])
    half_widths = Z_975 * standard_errors
    positions = np.arange(len(rows), dtype=float) + offset
    axis.errorbar(
        estimates,
        positions,
        xerr=half_widths,
        fmt="o",
        markersize=4.0,
        markeredgewidth=0,
        color=color,
        ecolor=color,
        elinewidth=1.15,
        capsize=2.3,
        capthick=1.15,
        label=legend_label,
        zorder=3,
    )
    return estimates - half_widths, estimates + half_widths


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # 7.16 in is the nominal IEEE two-column text width.  Drawing at the
    # publication size prevents LaTeX from shrinking the labels after export.
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.62), sharey=False)
    colors = {"query": "#D55E00", "cluster": "#009E73"}

    for axis_index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        group = [row for row in rows if row["dataset"] == dataset]

        lower_q, upper_q = add_intervals(
            axis,
            group,
            se_column="query_independent_two_way_se",
            offset=-0.105,
            color=colors["query"],
            legend_label="Query-level",
        )
        lower_c, upper_c = add_intervals(
            axis,
            group,
            se_column="triple_cluster_two_way_se",
            offset=0.105,
            color=colors["cluster"],
            legend_label="Source-triple",
        )

        all_lower = np.concatenate([lower_q, lower_c, np.asarray([0.0])])
        all_upper = np.concatenate([upper_q, upper_c, np.asarray([0.0])])
        span = float(all_upper.max() - all_lower.min())
        x_padding = max(0.0012, 0.045 * span)
        axis.set_xlim(float(all_lower.min() - x_padding), float(all_upper.max() + x_padding))
        axis.set_ylim(-0.42, len(group) - 0.58)
        axis.axvline(
            0.0,
            color="#555555",
            linewidth=0.85,
            linestyle=(0, (4.0, 3.0)),
            zorder=1,
        )
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.65, zorder=0)
        axis.set_yticks(np.arange(len(group)))
        axis.set_yticklabels([label(row) for row in group])
        axis.set_title(DISPLAY_DATASET[dataset], pad=3.0)
        axis.set_xlabel(r"Mean $\Delta$MRR (95\% CI)", labelpad=2.5)
        axis.tick_params(axis="y", pad=2.0, length=2.6)
        axis.tick_params(axis="x", pad=1.8, length=2.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if axis_index == 0:
            axis.legend(
                loc="upper right",
                frameon=True,
                framealpha=0.96,
                borderpad=0.32,
                labelspacing=0.28,
                handlelength=1.55,
                handletextpad=0.45,
            )

    # Leave enough room for the right panel's comparison labels.  A larger
    # inter-panel gap is preferable to letting those labels visually merge
    # with the left panel at the final two-column publication size.
    figure.subplots_adjust(left=0.148, right=0.995, bottom=0.205, top=0.905, wspace=0.72)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.018,
        facecolor="white",
    )
    figure.savefig(
        args.output.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.018,
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
