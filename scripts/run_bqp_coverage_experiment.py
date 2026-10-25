"""Empirical-population coverage experiment for BQP confidence intervals."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import stable_seed

DATASETS = ("FB15k237", "WN18RR", "NELL995")
MODELS_BY_DATASET = {
    "FB15k237": ("complex", "distmult", "cp", "rotate"),
    "WN18RR": ("complex", "distmult", "cp", "rotate"),
    "NELL995": ("complex", "distmult"),
}
SEEDS = (7, 11, 19, 23, 31)
SAMPLE_SIZES = (250, 500, 1000, 2000)
Z975 = 1.959963984540054


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pair(root: Path, dataset: str, model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
    with np.load(path, allow_pickle=False) as archive:
        rr = archive["reciprocal_ranks"].astype(np.float64)
        source = archive["source_triple_ids"]
        direction = archive["directions"]
    count = len(rr) // 2
    expected = np.arange(count, dtype=source.dtype)
    if not np.array_equal(source[:count], expected) or not np.array_equal(source[count:], expected):
        raise AssertionError(f"source pairing failed: {path}")
    if not np.all(direction[:count] == 0) or not np.all(direction[count:] == 1):
        raise AssertionError(f"direction ordering failed: {path}")
    return rr[:count], rr[count:]


def wilson(successes: int, trials: int, z: float = Z975) -> list[float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [center - half, center + half]


def simulate_matrix(
    tail: np.ndarray,
    head: np.ndarray,
    sample_size: int,
    repeats: int,
    rng: np.random.Generator,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Return coverage diagnostics for aligned rows shaped [series, triples]."""
    if tail.ndim != 2 or head.shape != tail.shape:
        raise ValueError("tail/head must have shape [series, triples]")
    series, population_size = tail.shape
    targets = ((tail + head) / 2.0).mean(axis=1)
    iid_hits = np.zeros(series, dtype=np.int64)
    cluster_hits = np.zeros(series, dtype=np.int64)
    iid_width_sum = np.zeros(series, dtype=np.float64)
    cluster_width_sum = np.zeros(series, dtype=np.float64)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        current = stop - start
        indices = rng.integers(0, population_size, size=(current, sample_size), dtype=np.int32)
        for row in range(series):
            sampled_tail = tail[row, indices]
            sampled_head = head[row, indices]
            tail_mean = sampled_tail.mean(axis=1)
            head_mean = sampled_head.mean(axis=1)
            estimates = (tail_mean + head_mean) / 2.0
            tail_var = sampled_tail.var(axis=1, ddof=1)
            head_var = sampled_head.var(axis=1, ddof=1)
            iid_se = np.sqrt(np.maximum(tail_var + head_var, 0.0) / (4.0 * sample_size))
            pair_var = ((sampled_tail + sampled_head) / 2.0).var(axis=1, ddof=1)
            cluster_se = np.sqrt(np.maximum(pair_var, 0.0) / sample_size)
            iid_half = Z975 * iid_se
            cluster_half = Z975 * cluster_se
            iid_hits[row] += np.count_nonzero(np.abs(estimates - targets[row]) <= iid_half)
            cluster_hits[row] += np.count_nonzero(np.abs(estimates - targets[row]) <= cluster_half)
            iid_width_sum[row] += float(np.sum(2.0 * iid_half))
            cluster_width_sum[row] += float(np.sum(2.0 * cluster_half))
    rows = []
    for row in range(series):
        iid_coverage = float(iid_hits[row] / repeats)
        cluster_coverage = float(cluster_hits[row] / repeats)
        rows.append(
            {
                "target": float(targets[row]),
                "population_size": population_size,
                "sample_size": sample_size,
                "repeats": repeats,
                "query_independent_hits": int(iid_hits[row]),
                "triple_cluster_hits": int(cluster_hits[row]),
                "query_independent_coverage": iid_coverage,
                "triple_cluster_coverage": cluster_coverage,
                "coverage_gap_cluster_minus_independent": cluster_coverage - iid_coverage,
                "query_independent_wilson95": wilson(int(iid_hits[row]), repeats),
                "triple_cluster_wilson95": wilson(int(cluster_hits[row]), repeats),
                "query_independent_mean_ci_width": float(iid_width_sum[row] / repeats),
                "triple_cluster_mean_ci_width": float(cluster_width_sum[row] / repeats),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, group in sorted(groups.items()):
        iid = np.asarray([row["query_independent_coverage"] for row in group])
        cluster = np.asarray([row["triple_cluster_coverage"] for row in group])
        item = {key: value for key, value in zip(keys, identity)}
        item.update(
            {
                "runs": len(group),
                "query_independent_coverage_mean": float(iid.mean()),
                "query_independent_coverage_min": float(iid.min()),
                "query_independent_coverage_max": float(iid.max()),
                "triple_cluster_coverage_mean": float(cluster.mean()),
                "triple_cluster_coverage_min": float(cluster.min()),
                "triple_cluster_coverage_max": float(cluster.max()),
                "coverage_gap_mean": float((cluster - iid).mean()),
            }
        )
        result.append(item)
    return result


def plot_single(summary: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, len(DATASETS), figsize=(5.5 * len(DATASETS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"complex": "#1b9e77", "distmult": "#d95f02", "cp": "#7570b3", "rotate": "#e7298a"}
    for axis, dataset in zip(axes, DATASETS):
        for model in MODELS_BY_DATASET[dataset]:
            group = sorted(
                [row for row in summary if row["dataset"] == dataset and row["model"] == model],
                key=lambda row: row["sample_size"],
            )
            x = [row["sample_size"] for row in group]
            axis.plot(x, [row["query_independent_coverage_mean"] for row in group], "--o", color=colors[model], alpha=0.8, label=f"{model} iid")
            axis.plot(x, [row["triple_cluster_coverage_mean"] for row in group], "-o", color=colors[model], label=f"{model} cluster")
        axis.axhline(0.95, color="black", linewidth=1, linestyle=":")
        axis.set_title(dataset)
        axis.set_xlabel("sampled source triples")
        axis.set_ylim(0.78, 0.98)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("empirical coverage of nominal 95% CI")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_control(summary: list[dict[str, Any]], output: Path) -> None:
    labels, iid, cluster = [], [], []
    for dataset in DATASETS:
        for condition in ("observed", "depaired_null"):
            row = next(item for item in summary if item["dataset"] == dataset and item["condition"] == condition)
            labels.append(f"{dataset}\n{condition}")
            iid.append(row["query_independent_coverage_mean"])
            cluster.append(row["triple_cluster_coverage_mean"])
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width / 2, iid, width, label="query independent", color="#d95f02")
    axis.bar(x + width / 2, cluster, width, label="source-triple cluster", color="#1b9e77")
    axis.axhline(0.95, color="black", linewidth=1, linestyle=":")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.78, 0.98)
    axis.set_ylabel("empirical coverage of nominal 95% CI")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def make_report(result: dict[str, Any], output: Path) -> None:
    single_summary = result["single_model_summary"]
    comparison_summary = result["comparison_summary"]
    control_summary = result["control_summary"]

    def average(rows: list[dict[str, Any]], field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    headline: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        single_rows = [
            row for row in single_summary if row["dataset"] == dataset and row["sample_size"] == 1000
        ]
        comparison_rows = [row for row in comparison_summary if row["dataset"] == dataset]
        observed = next(
            row for row in control_summary if row["dataset"] == dataset and row["condition"] == "observed"
        )
        depaired = next(
            row
            for row in control_summary
            if row["dataset"] == dataset and row["condition"] == "depaired_null"
        )
        headline[dataset] = {
            "single_iid": average(single_rows, "query_independent_coverage_mean"),
            "single_cluster": average(single_rows, "triple_cluster_coverage_mean"),
            "comparison_iid": average(comparison_rows, "query_independent_coverage_mean"),
            "comparison_cluster": average(comparison_rows, "triple_cluster_coverage_mean"),
            "observed_gap": float(observed["coverage_gap_mean"]),
            "depaired_gap": float(depaired["coverage_gap_mean"]),
        }

    fb = headline["FB15k237"]
    wn = headline["WN18RR"]
    nell = headline["NELL995"]
    lines = [
        "# BQP 置信区间覆盖率实验结果",
        "",
        "## 结论概要",
        "",
        f"- WN18RR 单模型在 n=1000 时，查询独立覆盖率为 {100*wn['single_iid']:.1f}%，"
        f"源三元组簇覆盖率为 {100*wn['single_cluster']:.1f}%；同种子模型差值分别为 "
        f"{100*wn['comparison_iid']:.1f}% 与 {100*wn['comparison_cluster']:.1f}%。",
        f"- WN18RR 真实配对条件下两种方法相差 {100*wn['observed_gap']:.1f} 个百分点；"
        f"去配对后只差 {100*wn['depaired_gap']:.1f} 个百分点，支持同源配对是误校准来源。",
        f"- FB15k-237 的总体效应较小：n=1000 时单模型覆盖率为 "
        f"{100*fb['single_iid']:.1f}% 与 {100*fb['single_cluster']:.1f}%，"
        f"模型差值为 {100*fb['comparison_iid']:.1f}% 与 {100*fb['comparison_cluster']:.1f}%。",
        f"- NELL995 的关系分层派生 split（200种关系）在 n=1000 时，单模型覆盖率为 "
        f"{100*nell['single_iid']:.1f}% 与 {100*nell['single_cluster']:.1f}%，"
        f"模型差值为 {100*nell['comparison_iid']:.1f}% 与 {100*nell['comparison_cluster']:.1f}%；"
        "该结果只针对这个派生测试集的经验分布。",
        "- FB15k-237 的 CP 单模型呈轻微反向，说明稳健主张应是“忽略配对造成方差误校准”，"
        "而非“查询独立法在所有模型上都低估标准误”。",
        "",
        "下表汇报跨种子均值以便阅读；每个运行基于3,000次重复的Wilson 95%区间保存在JSON与CSV中。",
        "",
        "## 单模型覆盖率（五个种子平均）",
        "",
        "| 数据集 | 模型 | n | 查询独立覆盖率 | 三元组簇覆盖率 | 改善 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in single_summary:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['sample_size']} | "
            f"{100*row['query_independent_coverage_mean']:.1f}% | "
            f"{100*row['triple_cluster_coverage_mean']:.1f}% | {100*row['coverage_gap_mean']:.1f} pp |"
        )
    lines += [
        "",
        "![单模型覆盖率曲线](../results_bqp/coverage_experiment/figures/single_model_coverage.png)",
        "",
        "## 同种子模型差值覆盖率（n=1000）",
        "",
        "| 数据集 | 架构对 | 运行数 | 查询独立覆盖率 | 三元组簇覆盖率 | 改善 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison_summary:
        lines.append(
            f"| {row['dataset']} | {row['model_a']}−{row['model_b']} | {row['runs']} | "
            f"{100*row['query_independent_coverage_mean']:.1f}% | "
            f"{100*row['triple_cluster_coverage_mean']:.1f}% | {100*row['coverage_gap_mean']:.1f} pp |"
        )
    lines += [
        "",
        "## 去配对空对照（n=1000）",
        "",
        "| 数据集 | 条件 | 运行数 | 查询独立覆盖率 | 三元组簇覆盖率 | 差值 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in control_summary:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['runs']} | "
            f"{100*row['query_independent_coverage_mean']:.1f}% | "
            f"{100*row['triple_cluster_coverage_mean']:.1f}% | {100*row['coverage_gap_mean']:.1f} pp |"
        )
    lines += [
        "",
        "![去配对空对照](../results_bqp/coverage_experiment/figures/depairing_control.png)",
        "",
        "覆盖率针对完整测试集经验分布，不应外推为现实知识图谱事实IID抽样下的最终覆盖保证。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "results_bqp" / "test_query_ranks")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_bqp" / "coverage_experiment")
    parser.add_argument("--repeats", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    cache = {
        (dataset, model, seed): load_pair(input_root, dataset, model, seed)
        for dataset in DATASETS for model in MODELS_BY_DATASET[dataset] for seed in SEEDS
    }

    single_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        models = MODELS_BY_DATASET[dataset]
        identities = [(model, seed) for model in models for seed in SEEDS]
        tail = np.stack([cache[(dataset, model, seed)][0] for model, seed in identities])
        head = np.stack([cache[(dataset, model, seed)][1] for model, seed in identities])
        for sample_size in SAMPLE_SIZES:
            rows = simulate_matrix(
                tail, head, sample_size, args.repeats,
                np.random.default_rng(stable_seed(20261018, "coverage", "single", dataset, sample_size)),
                args.batch_size,
            )
            for row, (model, seed) in zip(rows, identities):
                row.update({"family": "single_model", "condition": "observed", "dataset": dataset, "model": model, "seed": seed})
                single_rows.append(row)

        pair_identities = []
        pair_tail, pair_head = [], []
        for seed in SEEDS:
            for model_a, model_b in itertools.combinations(models, 2):
                pair_identities.append((model_a, model_b, seed))
                pair_tail.append(cache[(dataset, model_a, seed)][0] - cache[(dataset, model_b, seed)][0])
                pair_head.append(cache[(dataset, model_a, seed)][1] - cache[(dataset, model_b, seed)][1])
        rows = simulate_matrix(
            np.stack(pair_tail), np.stack(pair_head), 1000, args.repeats,
            np.random.default_rng(stable_seed(20261018, "coverage", "comparison", dataset)),
            args.batch_size,
        )
        for row, (model_a, model_b, seed) in zip(rows, pair_identities):
            row.update({"family": "model_comparison", "condition": "observed", "dataset": dataset, "model_a": model_a, "model_b": model_b, "seed": seed})
            comparison_rows.append(row)

        observed_rows = simulate_matrix(
            tail, head, 1000, args.repeats,
            np.random.default_rng(stable_seed(20261018, "coverage", "control-sample", dataset)),
            args.batch_size,
        )
        depaired_head = np.empty_like(head)
        for index, (model, seed) in enumerate(identities):
            rng = np.random.default_rng(stable_seed(20261018, "coverage", "depair", dataset, model, seed))
            depaired_head[index] = head[index, rng.permutation(head.shape[1])]
        null_rows = simulate_matrix(
            tail, depaired_head, 1000, args.repeats,
            np.random.default_rng(stable_seed(20261018, "coverage", "control-sample", dataset)),
            args.batch_size,
        )
        for condition, rows in (("observed", observed_rows), ("depaired_null", null_rows)):
            for row, (model, seed) in zip(rows, identities):
                row.update({"family": "control", "condition": condition, "dataset": dataset, "model": model, "seed": seed})
                control_rows.append(row)

    single_summary = aggregate(single_rows, ("dataset", "model", "sample_size"))
    comparison_summary = aggregate(comparison_rows, ("dataset", "model_a", "model_b"))
    control_summary = aggregate(control_rows, ("dataset", "condition"))
    result = {
        "format": "bqp-coverage-experiment-v1",
        "protocol": "docs/coverage_experiment_protocol.md",
        "models_by_dataset": {dataset: list(models) for dataset, models in MODELS_BY_DATASET.items()},
        "repeats": args.repeats,
        "sample_sizes": list(SAMPLE_SIZES),
        "single_model_rows": single_rows,
        "single_model_summary": single_summary,
        "comparison_rows": comparison_rows,
        "comparison_summary": comparison_summary,
        "control_rows": control_rows,
        "control_summary": control_summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "coverage_results.json", result)
    write_csv(output_root / "single_model_coverage.csv", single_rows)
    write_csv(output_root / "comparison_coverage.csv", comparison_rows)
    write_csv(output_root / "depairing_control.csv", control_rows)
    plot_single(single_summary, output_root / "figures" / "single_model_coverage")
    plot_control(control_summary, output_root / "figures" / "depairing_control")
    make_report(result, ROOT / "docs" / "coverage_experiment_results.md")
    print(json.dumps({
        "single_rows": len(single_rows),
        "comparison_rows": len(comparison_rows),
        "control_rows": len(control_rows),
        "output": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
