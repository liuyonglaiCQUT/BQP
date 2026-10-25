"""Redraw the single-model coverage figure from the saved experiment CSV.

This script does not rerun the coverage simulation.  It aggregates the five
checkpoint-specific coverage estimates already stored in
``single_model_coverage.csv`` and produces a compact, publication-sized plot.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results_bqp" / "coverage_experiment" / "single_model_coverage.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results_bqp"
    / "coverage_experiment"
    / "figures"
    / "single_model_coverage_compact"
)

DATASETS = ("FB15k237", "WN18RR", "NELL995")
DATASET_TITLES = {
    "FB15k237": "FB15k-237",
    "WN18RR": "WN18RR",
    "NELL995": "NELL-995 (derived)",
}
MODELS_BY_DATASET = {
    "FB15k237": ("complex", "distmult", "cp", "rotate"),
    "WN18RR": ("complex", "distmult", "cp", "rotate"),
    "NELL995": ("complex", "distmult"),
}
MODEL_LABELS = {
    "complex": "ComplEx",
    "distmult": "DistMult",
    "cp": "CP",
    "rotate": "RotatE",
}
MODEL_COLORS = {
    "complex": "#1B9E77",
    "distmult": "#D95F02",
    "cp": "#7570B3",
    "rotate": "#E7298A",
}
EXPECTED_SEEDS = {7, 11, 19, 23, 31}
SAMPLE_SIZES = (250, 500, 1000, 2000)


def load_summary(path: Path) -> dict[tuple[str, str, int], tuple[float, float]]:
    grouped: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["family"] != "single_model" or row["condition"] != "observed":
                continue
            key = (row["dataset"], row["model"], int(row["sample_size"]))
            grouped[key].append(
                (
                    int(row["seed"]),
                    float(row["query_independent_coverage"]),
                    float(row["triple_cluster_coverage"]),
                )
            )

    summary: dict[tuple[str, str, int], tuple[float, float]] = {}
    for dataset in DATASETS:
        for model in MODELS_BY_DATASET[dataset]:
            for sample_size in SAMPLE_SIZES:
                key = (dataset, model, sample_size)
                values = grouped.get(key, [])
                seeds = {seed for seed, _, _ in values}
                if seeds != EXPECTED_SEEDS:
                    raise ValueError(
                        f"{key} has seeds {sorted(seeds)}, expected {sorted(EXPECTED_SEEDS)}"
                    )
                summary[key] = (
                    fmean(iid for _, iid, _ in values),
                    fmean(cluster for _, _, cluster in values),
                )
    return summary


def draw(summary: dict[tuple[str, str, int], tuple[float, float]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # IEEE two-column text width is approximately 7.16 inches.  Drawing at
    # final publication size prevents fonts from being shrunk after inclusion.
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.72),
        sharex=True,
        sharey=True,
    )

    for axis, dataset in zip(axes, DATASETS):
        for model in MODELS_BY_DATASET[dataset]:
            iid = [summary[(dataset, model, n)][0] for n in SAMPLE_SIZES]
            cluster = [summary[(dataset, model, n)][1] for n in SAMPLE_SIZES]
            color = MODEL_COLORS[model]
            axis.plot(
                SAMPLE_SIZES,
                iid,
                linestyle="--",
                marker="o",
                markersize=3.7,
                linewidth=1.35,
                color=color,
                alpha=0.85,
            )
            axis.plot(
                SAMPLE_SIZES,
                cluster,
                linestyle="-",
                marker="o",
                markersize=3.7,
                linewidth=1.55,
                color=color,
            )

        axis.axhline(0.95, color="#222222", linewidth=1.0, linestyle=":")
        axis.set_title(DATASET_TITLES[dataset], pad=4.0)
        axis.set_xlim(190, 2060)
        axis.set_ylim(0.835, 0.965)
        axis.set_xticks(SAMPLE_SIZES)
        axis.set_yticks((0.84, 0.87, 0.90, 0.93, 0.95, 0.96))
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.35)
        axis.tick_params(length=2.5, width=0.7, pad=2.0)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#333333")

    figure.supylabel("Empirical coverage of nominal 95% CI", x=0.012, fontsize=8.7)
    figure.supxlabel("Sampled source triples", y=0.155, fontsize=8.7)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            markersize=3.7,
            linewidth=1.55,
            label=MODEL_LABELS[model],
        )
        for model in ("complex", "distmult", "cp", "rotate")
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            linewidth=1.35,
            label="Query-level",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            linewidth=1.55,
            label="Source-triple",
        ),
    ]
    figure.legend(
        handles=model_handles + method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.45,
    )

    figure.subplots_adjust(
        left=0.088,
        right=0.995,
        top=0.91,
        bottom=0.285,
        wspace=0.10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = load_summary(args.input)
    draw(summary, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
