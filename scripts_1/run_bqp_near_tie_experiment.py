"""Exhaustively compare frozen BQP checkpoints and audit significance flips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_pair_mean,
    paired_statistics,
    stable_seed,
    summarize_bootstrap,
    two_sided_normal_p,
)


DATASETS = ("FB15k237", "WN18RR")
MODELS = ("complex", "distmult")
SEEDS = (7, 11, 19, 23, 31)
RUNS = tuple((model, seed) for model in MODELS for seed in SEEDS)
Z_975 = 1.959963984540054


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(root: Path, dataset: str, model: str, seed: int) -> dict[str, np.ndarray]:
    path = root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
    with np.load(path, allow_pickle=False) as archive:
        directions = archive["directions"]
        source_ids = archive["source_triple_ids"]
        reciprocal = archive["reciprocal_ranks"].astype(np.float64)
        relations = archive["base_relations"]
        heads = archive["original_heads"]
        tails = archive["original_tails"]
    count = len(reciprocal) // 2
    expected = np.arange(count, dtype=source_ids.dtype)
    if len(reciprocal) != 2 * count:
        raise AssertionError(f"Odd query count: {path}")
    if not np.array_equal(source_ids[:count], expected) or not np.array_equal(
        source_ids[count:], expected
    ):
        raise AssertionError(f"Broken source pairing: {path}")
    if not np.all(directions[:count] == 0) or not np.all(directions[count:] == 1):
        raise AssertionError(f"Broken direction ordering: {path}")
    return {
        "tail": reciprocal[:count],
        "head": reciprocal[count:],
        "source_ids": source_ids[:count],
        "relations": relations[:count],
        "original_heads": heads[:count],
        "original_tails": tails[:count],
    }


def ordered_pair(first: tuple[str, int], second: tuple[str, int]) -> tuple[tuple[str, int], tuple[str, int]]:
    if first[0] != second[0]:
        return (first, second) if first[0] == "complex" else (second, first)
    return (first, second) if first[1] < second[1] else (second, first)


def pair_category(first: tuple[str, int], second: tuple[str, int]) -> str:
    if first[0] == second[0]:
        return "same_model_cross_seed"
    if first[1] == second[1]:
        return "cross_model_same_seed"
    return "cross_model_cross_seed"


def excludes_zero(interval: list[float] | np.ndarray) -> bool:
    return bool(interval[0] > 0.0 or interval[1] < 0.0)


def normal_ci(estimate: float, standard_error: float) -> list[float]:
    return [estimate - Z_975 * standard_error, estimate + Z_975 * standard_error]


def centered_bootstrap_p(distribution: np.ndarray, estimate: float) -> float:
    """Two-sided bootstrap p-value from the empirical, null-centered error distribution."""
    errors = np.asarray(distribution, dtype=np.float64) - estimate
    extreme = np.count_nonzero(np.abs(errors) >= abs(estimate))
    return float((1 + extreme) / (len(errors) + 1))


def bootstrap_all_runs(
    tail: np.ndarray,
    head: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap all runs with common indices so every model difference stays paired."""
    if tail.ndim != 2 or head.shape != tail.shape:
        raise ValueError("tail/head matrices must have aligned shape [runs, triples]")
    run_count, count = tail.shape
    independent = np.empty((run_count, repeats), dtype=np.float64)
    cluster = np.empty((run_count, repeats), dtype=np.float64)
    pair_mean = (tail + head) / 2.0
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        size = stop - start
        tail_indices = rng.integers(0, count, size=(size, count), dtype=np.int32)
        head_indices = rng.integers(0, count, size=(size, count), dtype=np.int32)
        cluster_indices = rng.integers(0, count, size=(size, count), dtype=np.int32)
        for run_index in range(run_count):
            independent[run_index, start:stop] = (
                tail[run_index, tail_indices].mean(axis=1)
                + head[run_index, head_indices].mean(axis=1)
            ) / 2.0
            cluster[run_index, start:stop] = pair_mean[run_index, cluster_indices].mean(axis=1)
    return independent, cluster


def summarize_pair(
    dataset: str,
    first: tuple[str, int],
    second: tuple[str, int],
    tail_delta: np.ndarray,
    head_delta: np.ndarray,
    independent_distribution: np.ndarray,
    cluster_distribution: np.ndarray,
    near_tie_threshold: float,
) -> dict[str, Any]:
    analytic = paired_statistics(tail_delta, head_delta)
    bootstrap = summarize_bootstrap(independent_distribution, cluster_distribution)
    estimate = float(analytic["estimate"])
    independent_normal_ci = normal_ci(estimate, float(analytic["query_independent_se"]))
    cluster_normal_ci = normal_ci(estimate, float(analytic["triple_cluster_se"]))
    independent_bootstrap_sig = excludes_zero(bootstrap["query_independent_ci95"])
    cluster_bootstrap_sig = excludes_zero(bootstrap["triple_cluster_ci95"])
    independent_normal_sig = excludes_zero(independent_normal_ci)
    cluster_normal_sig = excludes_zero(cluster_normal_ci)
    independent_bootstrap_p = centered_bootstrap_p(independent_distribution, estimate)
    cluster_bootstrap_p = centered_bootstrap_p(cluster_distribution, estimate)
    return {
        "dataset": dataset,
        "comparison_id": (
            f"{dataset}__{first[0]}_seed_{first[1]}__minus__{second[0]}_seed_{second[1]}"
        ),
        "category": pair_category(first, second),
        "model_a": first[0],
        "seed_a": first[1],
        "model_b": second[0],
        "seed_b": second[1],
        "near_tie": abs(estimate) <= near_tie_threshold,
        "analytic": {
            **analytic,
            "query_independent_normal_ci95": independent_normal_ci,
            "triple_cluster_normal_ci95": cluster_normal_ci,
        },
        "bootstrap": bootstrap,
        "inference": {
            "query_independent_normal_p": two_sided_normal_p(
                estimate, float(analytic["query_independent_se"])
            ),
            "triple_cluster_normal_p": two_sided_normal_p(
                estimate, float(analytic["triple_cluster_se"])
            ),
            "query_independent_normal_significant": independent_normal_sig,
            "triple_cluster_normal_significant": cluster_normal_sig,
            "normal_decision_flip": independent_normal_sig != cluster_normal_sig,
            "query_independent_bootstrap_significant": independent_bootstrap_sig,
            "triple_cluster_bootstrap_significant": cluster_bootstrap_sig,
            "bootstrap_flip_candidate": independent_bootstrap_sig != cluster_bootstrap_sig,
            "query_independent_centered_bootstrap_p": independent_bootstrap_p,
            "triple_cluster_centered_bootstrap_p": cluster_bootstrap_p,
            "centered_bootstrap_p_decision_flip": bool(
                (independent_bootstrap_p < 0.05) != (cluster_bootstrap_p < 0.05)
            ),
        },
    }


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    analytic = result["analytic"]
    bootstrap = result["bootstrap"]
    inference = result["inference"]
    confirmation = result.get("confirmation")
    return {
        "dataset": result["dataset"],
        "comparison_id": result["comparison_id"],
        "category": result["category"],
        "model_a": result["model_a"],
        "seed_a": result["seed_a"],
        "model_b": result["model_b"],
        "seed_b": result["seed_b"],
        "near_tie": result["near_tie"],
        "delta_mrr": analytic["estimate"],
        "tail_delta_mrr": analytic["tail_mean"],
        "head_delta_mrr": analytic["head_mean"],
        "direction_delta_correlation": analytic["head_tail_correlation"],
        "query_independent_se": analytic["query_independent_se"],
        "triple_cluster_se": analytic["triple_cluster_se"],
        "se_ratio": analytic["se_ratio_cluster_over_independent"],
        "se_underestimation_fraction": analytic["independent_se_underestimation_fraction"],
        "independent_bootstrap_ci_low": bootstrap["query_independent_ci95"][0],
        "independent_bootstrap_ci_high": bootstrap["query_independent_ci95"][1],
        "cluster_bootstrap_ci_low": bootstrap["triple_cluster_ci95"][0],
        "cluster_bootstrap_ci_high": bootstrap["triple_cluster_ci95"][1],
        "independent_bootstrap_significant": inference[
            "query_independent_bootstrap_significant"
        ],
        "cluster_bootstrap_significant": inference["triple_cluster_bootstrap_significant"],
        "bootstrap_flip_candidate": inference["bootstrap_flip_candidate"],
        "centered_bootstrap_p_decision_flip": inference[
            "centered_bootstrap_p_decision_flip"
        ],
        "confirmed_bootstrap_ci_flip": inference["confirmed_bootstrap_ci_flip"],
        "confirmed_centered_bootstrap_p_flip": inference[
            "confirmed_centered_bootstrap_p_flip"
        ],
        "normal_decision_flip": inference["normal_decision_flip"],
        "independent_normal_p": inference["query_independent_normal_p"],
        "cluster_normal_p": inference["triple_cluster_normal_p"],
        "independent_normal_q": inference["query_independent_normal_q"],
        "cluster_normal_q": inference["triple_cluster_normal_q"],
        "independent_bh_significant": inference["query_independent_bh_significant"],
        "cluster_bh_significant": inference["triple_cluster_bh_significant"],
        "bh_decision_flip": inference["bh_decision_flip"],
        "independent_centered_bootstrap_p": inference[
            "query_independent_centered_bootstrap_p"
        ],
        "cluster_centered_bootstrap_p": inference[
            "triple_cluster_centered_bootstrap_p"
        ],
        "independent_centered_bootstrap_q": inference[
            "query_independent_centered_bootstrap_q"
        ],
        "cluster_centered_bootstrap_q": inference[
            "triple_cluster_centered_bootstrap_q"
        ],
        "centered_bootstrap_bh_decision_flip": inference[
            "centered_bootstrap_bh_decision_flip"
        ],
        "confirmation_repeats": confirmation["repeats"] if confirmation else 0,
        "confirmation_independent_ci_low": (
            confirmation["query_independent_ci95"][0] if confirmation else ""
        ),
        "confirmation_independent_ci_high": (
            confirmation["query_independent_ci95"][1] if confirmation else ""
        ),
        "confirmation_cluster_ci_low": (
            confirmation["triple_cluster_ci95"][0] if confirmation else ""
        ),
        "confirmation_cluster_ci_high": (
            confirmation["triple_cluster_ci95"][1] if confirmation else ""
        ),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "comparisons": len(group),
            "near_ties": sum(item["near_tie"] for item in group),
            "bootstrap_flip_candidates": sum(
                item["inference"]["bootstrap_flip_candidate"] for item in group
            ),
            "confirmed_bootstrap_flips": sum(
                item["inference"]["confirmed_bootstrap_ci_flip"] for item in group
            ),
            "centered_bootstrap_p_flips": sum(
                item["inference"]["centered_bootstrap_p_decision_flip"] for item in group
            ),
            "confirmed_centered_bootstrap_p_flips": sum(
                item["inference"]["confirmed_centered_bootstrap_p_flip"] for item in group
            ),
            "normal_decision_flips": sum(
                item["inference"]["normal_decision_flip"] for item in group
            ),
            "bh_decision_flips": sum(item["inference"]["bh_decision_flip"] for item in group),
            "centered_bootstrap_bh_decision_flips": sum(
                item["inference"]["centered_bootstrap_bh_decision_flip"] for item in group
            ),
            "mean_se_ratio": float(
                np.mean([item["analytic"]["se_ratio_cluster_over_independent"] for item in group])
            ),
            "max_se_ratio": float(
                np.max([item["analytic"]["se_ratio_cluster_over_independent"] for item in group])
            ),
        }

    output: dict[str, Any] = {"all": counts(results)}
    output["near_ties"] = counts([item for item in results if item["near_tie"]])
    output["by_dataset"] = {}
    for dataset in DATASETS:
        dataset_group = [item for item in results if item["dataset"] == dataset]
        output["by_dataset"][dataset] = counts(dataset_group)
    output["by_category"] = {}
    for category in sorted({item["category"] for item in results}):
        category_group = [item for item in results if item["category"] == category]
        output["by_category"][category] = counts(category_group)
    return output


def plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"FB15k237": "#4477AA", "WN18RR": "#CC6677"}
    markers = {
        "same_model_cross_seed": "o",
        "cross_model_same_seed": "s",
        "cross_model_cross_seed": "^",
    }
    limit = max(
        3.0,
        max(abs(row["delta_mrr"]) / row["query_independent_se"] for row in rows) * 1.04,
    )
    category_labels = {
        "same_model_cross_seed": "same model / cross seed",
        "cross_model_same_seed": "cross model / same seed",
        "cross_model_cross_seed": "cross model / cross seed",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), constrained_layout=False)
    for axis_index, axis in enumerate(axes):
        for dataset in DATASETS:
            for category in markers:
                group = [
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["category"] == category
                ]
                if not group:
                    continue
                x = [abs(row["delta_mrr"]) / row["query_independent_se"] for row in group]
                y = [abs(row["delta_mrr"]) / row["triple_cluster_se"] for row in group]
                axis.scatter(
                    x,
                    y,
                    s=34,
                    alpha=0.75,
                    color=colors[dataset],
                    marker=markers[category],
                    label=(
                        f"{dataset} · {category_labels[category]}"
                        if axis_index == 0
                        else None
                    ),
                )
        axis.plot([0, limit], [0, limit], color="0.65", linewidth=1, linestyle=":")
        axis.axvline(Z_975, color="0.25", linewidth=1, linestyle="--")
        axis.axhline(Z_975, color="0.25", linewidth=1, linestyle="--")
        axis.fill_betweenx([0, Z_975], Z_975, limit, color="#EECC66", alpha=0.25)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("|ΔMRR| / query-independent SE")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_ylabel("|ΔMRR| / source-triple-cluster SE")
    axes[0].set_title("(a) All 90 comparisons")
    axes[1].set_xlim(1.6, 2.5)
    axes[1].set_ylim(1.6, 2.5)
    axes[1].set_title("(b) Decision-boundary zoom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=8,
        ncol=3,
    )
    fig.suptitle("Significance boundary under two resampling units")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.84, wspace=0.28)
    fig.savefig(output_dir / "all_pair_significance_map.png", dpi=220)
    fig.savefig(output_dir / "all_pair_significance_map.pdf")
    plt.close(fig)

    candidates = [
        row
        for row in rows
        if row["bootstrap_flip_candidate"]
        or row["normal_decision_flip"]
        or row["centered_bootstrap_p_decision_flip"]
    ]
    if not candidates:
        candidates = sorted(
            [row for row in rows if row["near_tie"]],
            key=lambda row: abs(abs(row["delta_mrr"]) / row["triple_cluster_se"] - Z_975),
        )[:12]
    candidates = sorted(candidates, key=lambda row: (row["dataset"], row["delta_mrr"]))
    fig, axis = plt.subplots(
        figsize=(8.0, max(3.5, 0.42 * len(candidates) + 1.4)), constrained_layout=True
    )
    positions = np.arange(len(candidates))
    for position, row in zip(positions, candidates):
        estimate = row["delta_mrr"]
        if row["confirmation_repeats"]:
            independent = np.asarray(
                [
                    row["confirmation_independent_ci_low"],
                    row["confirmation_independent_ci_high"],
                ]
            )
            cluster = np.asarray(
                [
                    row["confirmation_cluster_ci_low"],
                    row["confirmation_cluster_ci_high"],
                ]
            )
        else:
            independent = np.asarray(
                [row["independent_bootstrap_ci_low"], row["independent_bootstrap_ci_high"]]
            )
            cluster = np.asarray(
                [row["cluster_bootstrap_ci_low"], row["cluster_bootstrap_ci_high"]]
            )
        axis.errorbar(
            estimate,
            position + 0.10,
            xerr=np.asarray([[estimate - independent[0]], [independent[1] - estimate]]),
            fmt="o",
            color="#4477AA",
            capsize=2,
            label="Query independent" if position == 0 else None,
        )
        axis.errorbar(
            estimate,
            position - 0.10,
            xerr=np.asarray([[estimate - cluster[0]], [cluster[1] - estimate]]),
            fmt="o",
            color="#CC6677",
            capsize=2,
            label="Source-triple cluster" if position == 0 else None,
        )
    axis.axvline(0.0, color="0.3", linestyle="--", linewidth=1)
    labels = [
        f"{row['dataset']} · {row['model_a']} s{row['seed_a']} − {row['model_b']} s{row['seed_b']}"
        for row in candidates
    ]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("ΔMRR (95% bootstrap CI)")
    axis.set_title("Boundary and flip comparisons")
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output_dir / "boundary_comparison_ci.png", dpi=220)
    fig.savefig(output_dir / "boundary_comparison_ci.pdf")
    plt.close(fig)


def format_comparison(item: dict[str, Any]) -> str:
    return (
        f"{item['dataset']} / {item['model_a']} seed {item['seed_a']} − "
        f"{item['model_b']} seed {item['seed_b']}"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    aggregate_result = result["aggregate"]
    rows = [flatten(item) for item in result["comparisons"]]
    confirmed = [
        item
        for item in result["comparisons"]
        if item["inference"]["confirmed_bootstrap_ci_flip"]
        or item["inference"]["confirmed_centered_bootstrap_p_flip"]
    ]
    candidates = [
        item
        for item in result["comparisons"]
        if item["inference"]["bootstrap_flip_candidate"]
        or item["inference"]["normal_decision_flip"]
        or item["inference"]["centered_bootstrap_p_decision_flip"]
    ]
    lines = [
        "# BQP 近性能模型全集比较结果",
        "",
        "本结果严格枚举每个数据集内 10 个冻结 checkpoint 的全部 45 个无序模型对。",
        f"初筛使用 {result['protocol']['bootstrap_repeats']:,} 次 bootstrap；翻转候选使用 "
        f"{result['protocol']['confirmation_repeats']:,} 次独立 bootstrap 确认。",
        "",
        "## 总体结果",
        "",
        "| 范围 | 比较数 | 近性能数 | 10k percentile CI翻转 | 中心化p翻转 | 高精度确认CI翻转 | 高精度确认p翻转 | 正态区间翻转 | bootstrap BH翻转 | 正态BH翻转 | 平均SE比率 | 最大SE比率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    scopes = [("全部", aggregate_result["all"]), ("abs(ΔMRR)≤0.005", aggregate_result["near_ties"])]
    scopes.extend((dataset, aggregate_result["by_dataset"][dataset]) for dataset in DATASETS)
    for label, stats in scopes:
        lines.append(
            f"| {label} | {stats['comparisons']} | {stats['near_ties']} | "
            f"{stats['bootstrap_flip_candidates']} | {stats['centered_bootstrap_p_flips']} | "
            f"{stats['confirmed_bootstrap_flips']} | "
            f"{stats['confirmed_centered_bootstrap_p_flips']} | {stats['normal_decision_flips']} | "
            f"{stats['centered_bootstrap_bh_decision_flips']} | {stats['bh_decision_flips']} | "
            f"{stats['mean_se_ratio']:.3f} | {stats['max_se_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 按比较类型",
            "",
            "| 类型 | 比较数 | 近性能数 | 10k percentile CI翻转 | 中心化p翻转 | 高精度确认CI翻转 | 高精度确认p翻转 | bootstrap BH翻转 | 正态BH翻转 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    category_labels = {
        "same_model_cross_seed": "同模型、不同种子",
        "cross_model_same_seed": "跨模型、相同种子",
        "cross_model_cross_seed": "跨模型、不同种子",
    }
    for category, stats in aggregate_result["by_category"].items():
        lines.append(
            f"| {category_labels[category]} | {stats['comparisons']} | {stats['near_ties']} | "
            f"{stats['bootstrap_flip_candidates']} | {stats['centered_bootstrap_p_flips']} | "
            f"{stats['confirmed_bootstrap_flips']} | "
            f"{stats['confirmed_centered_bootstrap_p_flips']} | "
            f"{stats['centered_bootstrap_bh_decision_flips']} | {stats['bh_decision_flips']} |"
        )
    lines.extend(["", "## 翻转案例", ""])
    if confirmed:
        lines.append(
            "以下案例在 10,000 次初筛中出现中心化 p 值和解析正态判断翻转，并在高重复数确认中同时出现 percentile CI 与中心化 p 值翻转："
        )
        lines.append("")
        lines.append(
            "| 比较 | ΔMRR | SE 比率 | 查询独立95% CI | 三元组簇95% CI | 查询独立p | 簇p | 确认类型 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in confirmed:
            row = flatten(item)
            confirmation = item["confirmation"]
            lines.append(
                f"| {format_comparison(item)} | {row['delta_mrr']:.6f} | {row['se_ratio']:.3f} | "
                f"[{confirmation['query_independent_ci95'][0]:.6f}, {confirmation['query_independent_ci95'][1]:.6f}] | "
                f"[{confirmation['triple_cluster_ci95'][0]:.6f}, {confirmation['triple_cluster_ci95'][1]:.6f}] | "
                f"{confirmation['query_independent_centered_p']:.5f} | "
                f"{confirmation['triple_cluster_centered_p']:.5f} | "
                f"{'CI' if item['inference']['confirmed_bootstrap_ci_flip'] else ''}"
                f"{'p' if item['inference']['confirmed_centered_bootstrap_p_flip'] else ''} |"
            )
    else:
        lines.append("没有边界候选在高重复数确认 bootstrap 后保持判断翻转。")
    lines.extend(
        [
            "",
            f"初筛共发现 {len(candidates)} 个需要高重复数审计的边界候选；主张只使用确认后仍成立的案例。",
            "",
            "![全部模型对的显著性边界](../results_bqp/near_tie_experiment/figures/all_pair_significance_map.png)",
            "",
            "图中黄色区域表示查询独立正态检验显著、三元组簇正态检验不显著。",
            "",
            "![边界与翻转比较的区间](../results_bqp/near_tie_experiment/figures/boundary_comparison_ci.png)",
            "",
            "## 解释边界",
            "",
            "同模型不同种子的翻转说明两个真实训练实例的微小性能差可能被查询独立法过度确信；它不能单独证明两个算法的排序改变。"
            "跨模型案例更接近论文中的算法比较，但跨种子比较同时包含训练随机性。所有类别均完整报告，避免只挑选翻转案例。",
            "",
            "未经校正的区间回答单个预先指定比较是否显著；BH 结果回答在本次 45 项/数据集的穷举检验中控制 FDR 后是否仍翻转。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=ROOT / "results_bqp" / "test_query_ranks"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results_bqp" / "near_tie_experiment"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "docs" / "near_tie_comparison_results.md"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--confirmation-repeats", type=int, default=50000)
    parser.add_argument("--near-tie-threshold", type=float, default=0.005)
    parser.add_argument("--random-seed", type=int, default=20261018)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    if args.confirmation_repeats not in (0,) and args.confirmation_repeats < 1000:
        raise ValueError("confirmation-repeats must be 0 or at least 1000")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    input_audit = json.loads((input_root / "audit.json").read_text(encoding="utf-8"))
    # This analysis consumes the 20 FB15k-237/WN18RR ComplEx/DistMult runs,
    # while the final formal rank audit also contains CP, RotatE, and NELL-995.
    # Accept that superset and validate every consumed run below.
    if input_audit.get("status") != "passed" or input_audit.get("run_count", 0) < 20:
        raise AssertionError("Formal rank inputs must pass an audit containing the 20 required runs")

    comparisons: list[dict[str, Any]] = []
    distributions: dict[str, np.ndarray] = {}
    data_cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    delta_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for dataset in DATASETS:
        dataset_runs = []
        reference = None
        for model, seed in RUNS:
            run = load_run(input_root, dataset, model, seed)
            if reference is None:
                reference = run
            else:
                for key in ("source_ids", "relations", "original_heads", "original_tails"):
                    if not np.array_equal(reference[key], run[key]):
                        raise AssertionError(f"Unaligned {key}: {dataset}/{model}/seed_{seed}")
            data_cache[(dataset, model, seed)] = run
            dataset_runs.append(run)
        tail_matrix = np.stack([run["tail"] for run in dataset_runs])
        head_matrix = np.stack([run["head"] for run in dataset_runs])
        independent_means, cluster_means = bootstrap_all_runs(
            tail_matrix,
            head_matrix,
            args.bootstrap_repeats,
            np.random.default_rng(
                stable_seed(args.random_seed, "all-pairs", dataset, "initial-bootstrap")
            ),
            args.batch_size,
        )
        run_index = {run: index for index, run in enumerate(RUNS)}
        for raw_first, raw_second in itertools.combinations(RUNS, 2):
            first, second = ordered_pair(raw_first, raw_second)
            first_index, second_index = run_index[first], run_index[second]
            first_run = data_cache[(dataset, first[0], first[1])]
            second_run = data_cache[(dataset, second[0], second[1])]
            tail_delta = first_run["tail"] - second_run["tail"]
            head_delta = first_run["head"] - second_run["head"]
            result = summarize_pair(
                dataset,
                first,
                second,
                tail_delta,
                head_delta,
                independent_means[first_index] - independent_means[second_index],
                cluster_means[first_index] - cluster_means[second_index],
                args.near_tie_threshold,
            )
            comparisons.append(result)
            delta_cache[result["comparison_id"]] = (tail_delta, head_delta)
            distributions[f"{result['comparison_id']}__initial_query_independent"] = (
                independent_means[first_index] - independent_means[second_index]
            )
            distributions[f"{result['comparison_id']}__initial_triple_cluster"] = (
                cluster_means[first_index] - cluster_means[second_index]
            )
        print(f"initial bootstrap complete: {dataset}, 45 comparisons", flush=True)

    # Normal-test multiple-comparison control is applied separately within each dataset.
    for dataset in DATASETS:
        group = [item for item in comparisons if item["dataset"] == dataset]
        independent_q = benjamini_hochberg(
            item["inference"]["query_independent_normal_p"] for item in group
        )
        cluster_q = benjamini_hochberg(
            item["inference"]["triple_cluster_normal_p"] for item in group
        )
        independent_bootstrap_q = benjamini_hochberg(
            item["inference"]["query_independent_centered_bootstrap_p"] for item in group
        )
        cluster_bootstrap_q = benjamini_hochberg(
            item["inference"]["triple_cluster_centered_bootstrap_p"] for item in group
        )
        for item, q_independent, q_cluster, q_boot_independent, q_boot_cluster in zip(
            group,
            independent_q,
            cluster_q,
            independent_bootstrap_q,
            cluster_bootstrap_q,
        ):
            inference = item["inference"]
            inference["query_independent_normal_q"] = float(q_independent)
            inference["triple_cluster_normal_q"] = float(q_cluster)
            inference["query_independent_bh_significant"] = bool(q_independent < 0.05)
            inference["triple_cluster_bh_significant"] = bool(q_cluster < 0.05)
            inference["bh_decision_flip"] = bool(
                inference["query_independent_bh_significant"]
                != inference["triple_cluster_bh_significant"]
            )
            inference["query_independent_centered_bootstrap_q"] = float(
                q_boot_independent
            )
            inference["triple_cluster_centered_bootstrap_q"] = float(q_boot_cluster)
            inference["query_independent_centered_bootstrap_bh_significant"] = bool(
                q_boot_independent < 0.05
            )
            inference["triple_cluster_centered_bootstrap_bh_significant"] = bool(
                q_boot_cluster < 0.05
            )
            inference["centered_bootstrap_bh_decision_flip"] = bool(
                inference["query_independent_centered_bootstrap_bh_significant"]
                != inference["triple_cluster_centered_bootstrap_bh_significant"]
            )

    candidate_count = sum(
        item["inference"]["bootstrap_flip_candidate"]
        or item["inference"]["normal_decision_flip"]
        or item["inference"]["centered_bootstrap_p_decision_flip"]
        for item in comparisons
    )
    print(f"boundary candidates requiring confirmation: {candidate_count}", flush=True)
    for item in comparisons:
        inference = item["inference"]
        item["confirmation"] = None
        inference["confirmed_bootstrap_ci_flip"] = False
        inference["confirmed_centered_bootstrap_p_flip"] = False
        requires_confirmation = bool(
            inference["bootstrap_flip_candidate"]
            or inference["normal_decision_flip"]
            or inference["centered_bootstrap_p_decision_flip"]
        )
        inference["requires_confirmation"] = requires_confirmation
        if not requires_confirmation or args.confirmation_repeats == 0:
            continue
        tail_delta, head_delta = delta_cache[item["comparison_id"]]
        independent, cluster = bootstrap_pair_mean(
            tail_delta,
            head_delta,
            args.confirmation_repeats,
            np.random.default_rng(
                stable_seed(args.random_seed, item["comparison_id"], "confirmation-bootstrap")
            ),
            batch_size=args.batch_size,
        )
        confirmation = summarize_bootstrap(independent, cluster)
        confirmation_independent_sig = excludes_zero(confirmation["query_independent_ci95"])
        confirmation_cluster_sig = excludes_zero(confirmation["triple_cluster_ci95"])
        confirmation["query_independent_significant"] = confirmation_independent_sig
        confirmation["triple_cluster_significant"] = confirmation_cluster_sig
        confirmation["decision_flip"] = bool(
            confirmation_independent_sig != confirmation_cluster_sig
        )
        confirmation_independent_p = centered_bootstrap_p(
            independent, float(item["analytic"]["estimate"])
        )
        confirmation_cluster_p = centered_bootstrap_p(
            cluster, float(item["analytic"]["estimate"])
        )
        confirmation["query_independent_centered_p"] = confirmation_independent_p
        confirmation["triple_cluster_centered_p"] = confirmation_cluster_p
        confirmation["centered_p_decision_flip"] = bool(
            (confirmation_independent_p < 0.05) != (confirmation_cluster_p < 0.05)
        )
        item["confirmation"] = confirmation
        inference["confirmed_bootstrap_ci_flip"] = confirmation["decision_flip"]
        inference["confirmed_centered_bootstrap_p_flip"] = confirmation[
            "centered_p_decision_flip"
        ]
        distributions[f"{item['comparison_id']}__confirmation_query_independent"] = independent
        distributions[f"{item['comparison_id']}__confirmation_triple_cluster"] = cluster
        print(
            f"confirmed {item['comparison_id']}: "
            f"CI flip={confirmation['decision_flip']}, "
            f"centered-p flip={confirmation['centered_p_decision_flip']}",
            flush=True,
        )

    rows = [flatten(item) for item in comparisons]
    aggregate_result = aggregate(comparisons)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "all_pair_comparisons.csv"
    distributions_path = output_root / "all_pair_bootstrap_distributions.npz"
    write_csv(csv_path, rows)
    np.savez_compressed(distributions_path, **distributions)
    result = {
        "status": "complete",
        "format": "bqp-near-tie-all-pairs-v1",
        "protocol": {
            "comparison_universe": "all unordered pairs among 10 frozen checkpoints per dataset",
            "comparisons_per_dataset": 45,
            "bootstrap_repeats": args.bootstrap_repeats,
            "confirmation_repeats": args.confirmation_repeats,
            "confidence_level": 0.95,
            "near_tie_absolute_mrr_threshold": args.near_tie_threshold,
            "random_seed": args.random_seed,
            "multiple_testing": "Benjamini-Hochberg separately within each 45-comparison dataset family",
        },
        "comparisons": comparisons,
        "aggregate": aggregate_result,
        "artifacts": {
            "comparisons_csv_sha256": sha256(csv_path),
            "bootstrap_distributions_sha256": sha256(distributions_path),
        },
    }
    atomic_json(output_root / "near_tie_results.json", result)
    plot_results(rows, output_root / "figures")
    write_report(args.report.resolve(), result)
    print(
        "completed all-pair BQP experiment: "
        f"{len(comparisons)} comparisons, "
        f"{aggregate_result['all']['confirmed_centered_bootstrap_p_flips']} "
        "confirmed centered-bootstrap-p flips",
        flush=True,
    )


if __name__ == "__main__":
    main()

