"""Run the preregistered BQP core experiment on formal test-query ranks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import (
    benjamini_hochberg,
    bootstrap_pair_mean,
    paired_statistics,
    relation_centered_correlation,
    relation_stratified_permutation,
    stable_seed,
    summarize_bootstrap,
    two_sided_normal_p,
)


DATASETS = ("FB15k237", "WN18RR", "NELL995")
MODELS = ("complex", "distmult")
SEEDS = (7, 11, 19, 23, 31)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pairs(result_root: Path, dataset: str, model: str, seed: int) -> dict[str, np.ndarray]:
    path = result_root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
    with np.load(path, allow_pickle=False) as archive:
        directions = archive["directions"]
        source_ids = archive["source_triple_ids"]
        reciprocal = archive["reciprocal_ranks"].astype(np.float64)
        relations = archive["base_relations"]
        heads = archive["original_heads"]
        tails = archive["original_tails"]
    count = len(reciprocal) // 2
    if len(reciprocal) != 2 * count:
        raise AssertionError(f"Odd query count in {path}")
    expected_ids = np.arange(count, dtype=source_ids.dtype)
    if not np.array_equal(source_ids[:count], expected_ids) or not np.array_equal(
        source_ids[count:], expected_ids
    ):
        raise AssertionError(f"Broken source pairing in {path}")
    if not np.all(directions[:count] == 0) or not np.all(directions[count:] == 1):
        raise AssertionError(f"Broken direction order in {path}")
    return {
        "tail": reciprocal[:count],
        "head": reciprocal[count:],
        "relations": relations[:count].astype(np.int64, copy=False),
        "original_heads": heads[:count].astype(np.int64, copy=False),
        "original_tails": tails[:count].astype(np.int64, copy=False),
        "source_ids": source_ids[:count].astype(np.int64, copy=False),
    }


def normal_interval(estimate: float, standard_error: float) -> list[float]:
    return [estimate - 1.959963984540054 * standard_error, estimate + 1.959963984540054 * standard_error]


def analyze_pair(
    tail: np.ndarray,
    head: np.ndarray,
    relations: np.ndarray,
    bootstrap_repeats: int,
    permutation_repeats: int,
    bootstrap_rng: np.random.Generator,
    permutation_rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    analytic = paired_statistics(tail, head)
    independent_distribution, cluster_distribution = bootstrap_pair_mean(
        tail, head, bootstrap_repeats, bootstrap_rng
    )
    bootstrap = summarize_bootstrap(independent_distribution, cluster_distribution)
    permutation_distribution, permutation = relation_stratified_permutation(
        tail, head, relations, permutation_repeats, permutation_rng
    )
    analytic_ratio = float(analytic["se_ratio_cluster_over_independent"])
    bootstrap_ratio = float(bootstrap["se_ratio_cluster_over_independent"])
    ratio_difference = abs(analytic_ratio - bootstrap_ratio)
    ratio_tolerance = max(0.03, 2.0 / math.sqrt(bootstrap_repeats))
    if bootstrap_repeats >= 1000 and ratio_difference > ratio_tolerance:
        raise AssertionError(
            f"Analytic/bootstrap SE-ratio disagreement: {analytic_ratio} vs {bootstrap_ratio}"
        )
    estimate = float(analytic["estimate"])
    analytic["query_independent_normal_ci95"] = normal_interval(
        estimate, float(analytic["query_independent_se"])
    )
    analytic["triple_cluster_normal_ci95"] = normal_interval(
        estimate, float(analytic["triple_cluster_se"])
    )
    result = {
        "analytic": analytic,
        "bootstrap": bootstrap,
        "relation_control": {
            "relation_centered_correlation": relation_centered_correlation(
                tail, head, relations
            ),
            **permutation,
        },
        "audit": {
            "analytic_bootstrap_se_ratio_absolute_difference": ratio_difference,
            "tolerance": ratio_tolerance,
            "passed": True,
        },
    }
    distributions = {
        "query_independent_bootstrap": independent_distribution,
        "triple_cluster_bootstrap": cluster_distribution,
        "relation_stratified_null_correlation": permutation_distribution,
    }
    return result, distributions


def flatten_single(result: dict[str, Any]) -> dict[str, Any]:
    analytic = result["analytic"]
    bootstrap = result["bootstrap"]
    control = result["relation_control"]
    return {
        "dataset": result["dataset"],
        "model": result["model"],
        "seed": result["seed"],
        "source_triples": analytic["source_triples"],
        "mrr": analytic["estimate"],
        "tail_mrr": analytic["tail_mean"],
        "head_mrr": analytic["head_mean"],
        "head_tail_covariance": analytic["head_tail_covariance"],
        "head_tail_correlation": analytic["head_tail_correlation"],
        "relation_centered_correlation": control["relation_centered_correlation"],
        "query_independent_se": analytic["query_independent_se"],
        "pooled_query_iid_se": analytic["pooled_query_iid_se"],
        "triple_cluster_se": analytic["triple_cluster_se"],
        "analytic_se_ratio": analytic["se_ratio_cluster_over_independent"],
        "bootstrap_se_ratio": bootstrap["se_ratio_cluster_over_independent"],
        "design_effect": analytic["design_effect"],
        "effective_directional_queries": analytic["effective_directional_queries"],
        "effective_sample_fraction": analytic["effective_sample_fraction"],
        "independent_se_underestimation_fraction": analytic[
            "independent_se_underestimation_fraction"
        ],
        "independent_bootstrap_ci_low": bootstrap["query_independent_ci95"][0],
        "independent_bootstrap_ci_high": bootstrap["query_independent_ci95"][1],
        "cluster_bootstrap_ci_low": bootstrap["triple_cluster_ci95"][0],
        "cluster_bootstrap_ci_high": bootstrap["triple_cluster_ci95"][1],
        "relation_null_mean": control["null_mean"],
        "relation_null_q025": control["null_q025"],
        "relation_null_q975": control["null_q975"],
        "relation_permutation_p": control["p_value_greater"],
        "relation_permutation_q": result["relation_permutation_q"],
    }


def flatten_comparison(result: dict[str, Any]) -> dict[str, Any]:
    row = flatten_single(result)
    row.pop("model")
    row["comparison"] = "complex-minus-distmult"
    row["delta_mrr"] = row.pop("mrr")
    row["tail_delta_mrr"] = row.pop("tail_mrr")
    row["head_delta_mrr"] = row.pop("head_mrr")
    analytic = result["analytic"]
    bootstrap = result["bootstrap"]
    row.update(
        {
            "query_independent_normal_p": result["inference"][
                "query_independent_normal_p"
            ],
            "triple_cluster_normal_p": result["inference"][
                "triple_cluster_normal_p"
            ],
            "independent_bootstrap_significant": result["inference"][
                "query_independent_bootstrap_ci_excludes_zero"
            ],
            "cluster_bootstrap_significant": result["inference"][
                "triple_cluster_bootstrap_ci_excludes_zero"
            ],
            "significance_decision_flip": result["inference"]["decision_flip"],
            "independent_normal_ci_low": analytic["query_independent_normal_ci95"][0],
            "independent_normal_ci_high": analytic["query_independent_normal_ci95"][1],
            "cluster_normal_ci_low": analytic["triple_cluster_normal_ci95"][0],
            "cluster_normal_ci_high": analytic["triple_cluster_normal_ci95"][1],
            "independent_bootstrap_ci_width": bootstrap[
                "query_independent_ci95_width"
            ],
            "cluster_bootstrap_ci_width": bootstrap["triple_cluster_ci95_width"],
        }
    )
    return row


def range_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_single(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"])].append(row)
    output = []
    for (dataset, model), group in sorted(groups.items()):
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "seeds": len(group),
                "head_tail_correlation": range_summary(
                    row["head_tail_correlation"] for row in group
                ),
                "relation_centered_correlation": range_summary(
                    row["relation_centered_correlation"] for row in group
                ),
                "analytic_se_ratio": range_summary(
                    row["analytic_se_ratio"] for row in group
                ),
                "bootstrap_se_ratio": range_summary(
                    row["bootstrap_se_ratio"] for row in group
                ),
                "effective_sample_fraction": range_summary(
                    row["effective_sample_fraction"] for row in group
                ),
                "se_underestimation_fraction": range_summary(
                    row["independent_se_underestimation_fraction"] for row in group
                ),
                "relation_control_significant_count": sum(
                    row["relation_permutation_q"] < 0.05 for row in group
                ),
            }
        )
    return output


def aggregate_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    output = []
    for dataset, group in sorted(groups.items()):
        output.append(
            {
                "dataset": dataset,
                "comparison": "complex-minus-distmult",
                "seeds": len(group),
                "delta_mrr": range_summary(row["delta_mrr"] for row in group),
                "head_tail_delta_correlation": range_summary(
                    row["head_tail_correlation"] for row in group
                ),
                "analytic_se_ratio": range_summary(
                    row["analytic_se_ratio"] for row in group
                ),
                "bootstrap_se_ratio": range_summary(
                    row["bootstrap_se_ratio"] for row in group
                ),
                "se_underestimation_fraction": range_summary(
                    row["independent_se_underestimation_fraction"] for row in group
                ),
                "independent_significant_count": sum(
                    bool(row["independent_bootstrap_significant"]) for row in group
                ),
                "cluster_significant_count": sum(
                    bool(row["cluster_bootstrap_significant"]) for row in group
                ),
                "decision_flip_count": sum(
                    bool(row["significance_decision_flip"]) for row in group
                ),
                "relation_control_significant_count": sum(
                    row["relation_permutation_q"] < 0.05 for row in group
                ),
            }
        )
    return output


def plot_core(single_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_names = {"FB15k237": "FB15k-237", "WN18RR": "WN18RR", "NELL995": "NELL-995"}
    model_names = {"complex": "ComplEx", "distmult": "DistMult"}
    labels = [
        (dataset, model, f"{display_names.get(dataset, dataset)}\n{model_names[model]}")
        for dataset in DATASETS for model in MODELS
    ]
    colors = {7: "#4477AA", 11: "#66CCEE", 19: "#228833", 23: "#CCBB44", 31: "#EE6677"}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.2), constrained_layout=True)
    for x, (dataset, model, _) in enumerate(labels):
        group = sorted(
            [row for row in single_rows if row["dataset"] == dataset and row["model"] == model],
            key=lambda row: row["seed"],
        )
        offsets = np.linspace(-0.16, 0.16, len(group))
        for offset, row in zip(offsets, group):
            axes[0].scatter(
                x + offset,
                row["head_tail_correlation"],
                color=colors[row["seed"]],
                s=28,
                zorder=3,
            )
            axes[0].scatter(
                x + offset,
                row["relation_centered_correlation"],
                facecolor="white",
                edgecolor=colors[row["seed"]],
                s=28,
                zorder=3,
            )
            axes[1].scatter(x + offset, row["analytic_se_ratio"], color=colors[row["seed"]], s=28)
            axes[2].scatter(x + offset, row["effective_sample_fraction"], color=colors[row["seed"]], s=28)
    axes[0].scatter([], [], color="black", label="Raw")
    axes[0].scatter([], [], facecolor="white", edgecolor="black", label="Relation-centered")
    axes[0].set_ylabel("Head-tail Pearson correlation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="0.45", linestyle="--", linewidth=1)
    axes[1].set_ylabel("SE ratio: triple cluster / query independent")
    axes[2].axhline(1.0, color="0.45", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Effective directional-query fraction")
    for axis in axes:
        axis.set_xticks(range(len(labels)), [label for _, _, label in labels])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) Paired dependence")
    axes[1].set_title("(b) Standard-error inflation")
    axes[2].set_title("(c) Effective sample size")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    rows = sorted(comparison_rows, key=lambda row: (row["dataset"], row["seed"]))
    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    positions = np.arange(len(rows))
    for position, row in zip(positions, rows):
        estimate = row["delta_mrr"]
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
            label="Triple cluster" if position == 0 else None,
        )
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1)
    axis.set_yticks(
        positions,
        [f"{row['dataset']} · seed {row['seed']}" for row in rows],
    )
    axis.invert_yaxis()
    axis.set_xlabel("MRR difference: ComplEx − DistMult (95% bootstrap CI)")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output.parent / "model_comparison_ci.png", dpi=220)
    fig.savefig(output.parent / "model_comparison_ci.pdf")
    plt.close(fig)


def format_range(summary: dict[str, float], digits: int = 3) -> str:
    return f"{summary['mean']:.{digits}f} [{summary['min']:.{digits}f}, {summary['max']:.{digits}f}]"


def write_report(
    path: Path,
    single_aggregate: list[dict[str, Any]],
    comparison_aggregate: list[dict[str, Any]],
    single_run_count: int,
    comparison_run_count: int,
    bootstrap_repeats: int,
    permutation_repeats: int,
) -> None:
    lines = [
        "# BQP 正式核心实验结果",
        "",
        "本报告由 `scripts/run_bqp_core_experiment.py` 从冻结的正式测试排名自动生成。",
        f"每项分析使用 {bootstrap_repeats:,} 次 bootstrap 和 {permutation_repeats:,} 次关系内随机重配。",
        "方括号给出五个训练 seed 的最小值和最大值。",
        "",
        "## 单模型 MRR",
        "",
        "| 数据集 | 模型 | 头尾相关性 | 关系中心化相关性 | SE 比率 | 查询独立法低估 SE | 有效样本比例 | 关系控制 q<0.05 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in single_aggregate:
        lines.append(
            "| {dataset} | {model} | {corr} | {centered} | {ratio} | {under} | {effective} | {sig}/5 |".format(
                dataset=row["dataset"],
                model="ComplEx" if row["model"] == "complex" else "DistMult",
                corr=format_range(row["head_tail_correlation"]),
                centered=format_range(row["relation_centered_correlation"]),
                ratio=format_range(row["analytic_se_ratio"]),
                under=format_range(row["se_underestimation_fraction"], digits=3),
                effective=format_range(row["effective_sample_fraction"], digits=3),
                sig=row["relation_control_significant_count"],
            )
        )
    lines.extend(
        [
            "",
            "`SE 比率` 为三元组簇标准误除以保持方向规模的查询独立标准误；大于 1 表示后者低估不确定性。",
            "`查询独立法低估 SE` 是比例值，例如 0.26 表示标准误被低估 26%。",
            "",
            "## ComplEx 与 DistMult 的差值",
            "",
            "| 数据集 | MRR 差值 | 方向差值相关性 | SE 比率 | 查询独立法低估 SE | 独立/簇显著次数 | 判断翻转 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_aggregate:
        lines.append(
            "| {dataset} | {delta} | {corr} | {ratio} | {under} | {independent}/5, {cluster}/5 | {flips}/5 |".format(
                dataset=row["dataset"],
                delta=format_range(row["delta_mrr"], digits=4),
                corr=format_range(row["head_tail_delta_correlation"]),
                ratio=format_range(row["analytic_se_ratio"]),
                under=format_range(row["se_underestimation_fraction"]),
                independent=row["independent_significant_count"],
                cluster=row["cluster_significant_count"],
                flips=row["decision_flip_count"],
            )
        )
    all_single_ratios = [row["analytic_se_ratio"]["min"] for row in single_aggregate]
    all_controls = [row["relation_control_significant_count"] for row in single_aggregate]
    all_flips = sum(row["decision_flip_count"] for row in comparison_aggregate)
    lines.extend(
        [
            "",
            "## 可支持的结论",
            "",
            f"- {len(single_aggregate)} 个数据集–模型组合中，最小单模型 SE 比率为 {min(all_single_ratios):.3f}；请结合逐组合结果判断方向依赖是否总会增大标准误。",
            f"- 关系内随机重配经 BH 校正后，各组合显著 seed 数为 {', '.join(str(x) + '/5' for x in all_controls)}，用于判断同源配对信号是否超出关系组成。",
            "- 效应大小具有明显数据集依赖：WN18RR 的有效查询样本量损失很大，FB15k-237 较小但方向稳定。",
            f"- 本次 ComplEx–DistMult 比较出现 {all_flips}/{comparison_run_count} 次显著性判断翻转。没有翻转不代表查询独立假设正确；它表示当前模型差值离零足够远，区间变宽后结论仍相同。",
            "- 这些结果支撑 BQP 作为评估不确定性问题；它们尚不证明所有 KGC 模型比较都会改变结论。",
            "",
            "## 论文资产",
            "",
            f"- `results_bqp/core_experiment/single_model_runs.csv`：{single_run_count} 个单模型运行；",
            f"- `results_bqp/core_experiment/model_comparisons.csv`：{comparison_run_count} 个同 seed 模型比较；",
            "- `results_bqp/core_experiment/core_results.json`：完整嵌套结果；",
            "- `results_bqp/core_experiment/bootstrap_distributions.npz`：bootstrap 与置换零分布；",
            "- `results_bqp/core_experiment/figures/core_effects.{png,pdf}`；",
            "- `results_bqp/core_experiment/figures/model_comparison_ci.{png,pdf}`。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=ROOT / "results_bqp" / "test_query_ranks"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results_bqp" / "core_experiment"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "docs" / "core_experiment_results.md"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=50000)
    parser.add_argument("--permutation-repeats", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20261018)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    audit = json.loads((input_root / "audit.json").read_text(encoding="utf-8"))
    expected_core_runs = len(DATASETS) * len(MODELS) * len(SEEDS)
    if audit.get("status") != "passed" or audit.get("run_count") < expected_core_runs:
        raise AssertionError(f"Formal test ranks must pass an audit with at least {expected_core_runs} runs first")

    pair_cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    single_results: list[dict[str, Any]] = []
    distributions: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                pair = load_pairs(input_root, dataset, model, seed)
                pair_cache[(dataset, model, seed)] = pair
                result, sampled = analyze_pair(
                    pair["tail"],
                    pair["head"],
                    pair["relations"],
                    args.bootstrap_repeats,
                    args.permutation_repeats,
                    np.random.default_rng(
                        stable_seed(args.random_seed, "single", dataset, model, seed, "bootstrap")
                    ),
                    np.random.default_rng(
                        stable_seed(args.random_seed, "single", dataset, model, seed, "permutation")
                    ),
                )
                result.update({"dataset": dataset, "model": model, "seed": seed})
                single_results.append(result)
                prefix = f"single__{dataset}__{model}__seed_{seed}"
                for name, values in sampled.items():
                    distributions[f"{prefix}__{name}"] = values
                print(
                    f"single {dataset}/{model}/seed_{seed}: "
                    f"corr={result['analytic']['head_tail_correlation']:.3f}, "
                    f"SE ratio={result['analytic']['se_ratio_cluster_over_independent']:.3f}",
                    flush=True,
                )

    single_q = benjamini_hochberg(
        result["relation_control"]["p_value_greater"] for result in single_results
    )
    for result, q_value in zip(single_results, single_q):
        result["relation_permutation_q"] = float(q_value)

    comparison_results: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            complex_pair = pair_cache[(dataset, "complex", seed)]
            distmult_pair = pair_cache[(dataset, "distmult", seed)]
            for key in ("source_ids", "relations", "original_heads", "original_tails"):
                if not np.array_equal(complex_pair[key], distmult_pair[key]):
                    raise AssertionError(f"Unaligned {key} for {dataset}/seed_{seed}")
            tail_delta = complex_pair["tail"] - distmult_pair["tail"]
            head_delta = complex_pair["head"] - distmult_pair["head"]
            result, sampled = analyze_pair(
                tail_delta,
                head_delta,
                complex_pair["relations"],
                args.bootstrap_repeats,
                args.permutation_repeats,
                np.random.default_rng(
                    stable_seed(args.random_seed, "comparison", dataset, seed, "bootstrap")
                ),
                np.random.default_rng(
                    stable_seed(args.random_seed, "comparison", dataset, seed, "permutation")
                ),
            )
            estimate = float(result["analytic"]["estimate"])
            independent_ci = result["bootstrap"]["query_independent_ci95"]
            cluster_ci = result["bootstrap"]["triple_cluster_ci95"]
            independent_significant = independent_ci[0] > 0.0 or independent_ci[1] < 0.0
            cluster_significant = cluster_ci[0] > 0.0 or cluster_ci[1] < 0.0
            result.update(
                {
                    "dataset": dataset,
                    "model": "complex-minus-distmult",
                    "seed": seed,
                    "inference": {
                        "query_independent_normal_p": two_sided_normal_p(
                            estimate, float(result["analytic"]["query_independent_se"])
                        ),
                        "triple_cluster_normal_p": two_sided_normal_p(
                            estimate, float(result["analytic"]["triple_cluster_se"])
                        ),
                        "query_independent_bootstrap_ci_excludes_zero": independent_significant,
                        "triple_cluster_bootstrap_ci_excludes_zero": cluster_significant,
                        "decision_flip": independent_significant != cluster_significant,
                    },
                }
            )
            comparison_results.append(result)
            prefix = f"comparison__{dataset}__seed_{seed}"
            for name, values in sampled.items():
                distributions[f"{prefix}__{name}"] = values
            print(
                f"compare {dataset}/seed_{seed}: delta={estimate:.6f}, "
                f"SE ratio={result['analytic']['se_ratio_cluster_over_independent']:.3f}",
                flush=True,
            )

    comparison_q = benjamini_hochberg(
        result["relation_control"]["p_value_greater"]
        for result in comparison_results
    )
    for result, q_value in zip(comparison_results, comparison_q):
        result["relation_permutation_q"] = float(q_value)

    single_rows = [flatten_single(result) for result in single_results]
    comparison_rows = [flatten_comparison(result) for result in comparison_results]
    single_aggregate = aggregate_single(single_rows)
    comparison_aggregate = aggregate_comparisons(comparison_rows)
    write_csv(output_root / "single_model_runs.csv", single_rows)
    write_csv(output_root / "model_comparisons.csv", comparison_rows)
    np.savez_compressed(output_root / "bootstrap_distributions.npz", **distributions)
    core_result = {
        "status": "complete",
        "format": "bqp-core-experiment-v1",
        "protocol": {
            "bootstrap_repeats": args.bootstrap_repeats,
            "permutation_repeats": args.permutation_repeats,
            "random_seed": args.random_seed,
            "confidence_level": 0.95,
            "multiple_testing": "Benjamini-Hochberg; single-model and model-comparison families separate",
            "primary_se_comparison": "source-triple cluster vs direction-stratified query independent",
        },
        "single_model_runs": single_results,
        "model_comparisons": comparison_results,
        "aggregate": {
            "single_model": single_aggregate,
            "model_comparison": comparison_aggregate,
        },
    }
    atomic_json(output_root / "core_results.json", core_result)
    plot_core(single_rows, comparison_rows, output_root / "figures" / "core_effects")
    write_report(
        args.report.resolve(),
        single_aggregate,
        comparison_aggregate,
        len(single_rows),
        len(comparison_rows),
        args.bootstrap_repeats,
        args.permutation_repeats,
    )
    print(f"completed BQP core experiment: {output_root}", flush=True)


if __name__ == "__main__":
    main()

