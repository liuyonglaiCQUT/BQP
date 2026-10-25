"""Multi-architecture same-seed and algorithm-level BQP inference."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import benjamini_hochberg, paired_statistics, stable_seed, two_sided_normal_p

DATASETS = ("FB15k237", "WN18RR")
MODELS = ("complex", "distmult", "cp", "rotate")
SEEDS = (7, 11, 19, 23, 31)
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
        directions = archive["directions"]
    count = len(rr) // 2
    expected = np.arange(count, dtype=source.dtype)
    if not np.array_equal(source[:count], expected) or not np.array_equal(source[count:], expected):
        raise AssertionError(f"source pairing failed: {path}")
    if not np.all(directions[:count] == 0) or not np.all(directions[count:] == 1):
        raise AssertionError(f"direction ordering failed: {path}")
    return rr[:count], rr[count:]


def excludes_zero(ci: list[float]) -> bool:
    return ci[0] > 0.0 or ci[1] < 0.0


def centered_p(errors: np.ndarray, estimate: float) -> float:
    return float((1 + np.count_nonzero(np.abs(errors) >= abs(estimate))) / (len(errors) + 1))


def wild_checkpoint_distributions(
    tail: np.ndarray,
    head: np.ndarray,
    repeats: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Common Gaussian-multiplier distributions for all models in one seed."""
    means = (tail.mean(axis=1) + head.mean(axis=1)) / 2.0
    pair = (tail + head) / 2.0
    tail_centered = torch.as_tensor(tail - tail.mean(axis=1, keepdims=True), dtype=torch.float32, device=device).t()
    head_centered = torch.as_tensor(head - head.mean(axis=1, keepdims=True), dtype=torch.float32, device=device).t()
    pair_centered = torch.as_tensor(pair - pair.mean(axis=1, keepdims=True), dtype=torch.float32, device=device).t()
    count = tail.shape[1]
    model_count = tail.shape[0]
    independent = np.empty((repeats, model_count), dtype=np.float64)
    cluster = np.empty((repeats, model_count), dtype=np.float64)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        size = stop - start
        wt = torch.randn((size, count), generator=generator, device=device)
        wh = torch.randn((size, count), generator=generator, device=device)
        wc = torch.randn((size, count), generator=generator, device=device)
        ind_error = (wt @ tail_centered + wh @ head_centered) / (2.0 * count)
        clu_error = (wc @ pair_centered) / count
        independent[start:stop] = means + ind_error.cpu().numpy().astype(np.float64)
        cluster[start:stop] = means + clu_error.cpu().numpy().astype(np.float64)
    return independent, cluster


def summarize_checkpoint_pair(
    dataset: str,
    seed: int,
    model_a: str,
    model_b: str,
    tail_delta: np.ndarray,
    head_delta: np.ndarray,
    independent: np.ndarray,
    cluster: np.ndarray,
) -> dict[str, Any]:
    analytic = paired_statistics(tail_delta, head_delta)
    estimate = float(analytic["estimate"])
    ind_se = float(analytic["query_independent_se"])
    clu_se = float(analytic["triple_cluster_se"])
    ind_ci = [estimate - Z975 * ind_se, estimate + Z975 * ind_se]
    clu_ci = [estimate - Z975 * clu_se, estimate + Z975 * clu_se]
    ind_errors = independent - estimate
    clu_errors = cluster - estimate
    ind_boot_ci = [float(value) for value in estimate + np.quantile(ind_errors, [0.025, 0.975])]
    clu_boot_ci = [float(value) for value in estimate + np.quantile(clu_errors, [0.025, 0.975])]
    ind_boot_p = centered_p(ind_errors, estimate)
    clu_boot_p = centered_p(clu_errors, estimate)
    return {
        "dataset": dataset,
        "seed": seed,
        "model_a": model_a,
        "model_b": model_b,
        "comparison_id": f"{dataset}__seed_{seed}__{model_a}__minus__{model_b}",
        "delta_mrr": estimate,
        "tail_delta_mrr": float(analytic["tail_mean"]),
        "head_delta_mrr": float(analytic["head_mean"]),
        "direction_delta_correlation": float(analytic["head_tail_correlation"]),
        "query_independent_se": ind_se,
        "triple_cluster_se": clu_se,
        "se_ratio": clu_se / ind_se,
        "query_independent_normal_ci95": ind_ci,
        "triple_cluster_normal_ci95": clu_ci,
        "query_independent_normal_p": two_sided_normal_p(estimate, ind_se),
        "triple_cluster_normal_p": two_sided_normal_p(estimate, clu_se),
        "normal_flip": excludes_zero(ind_ci) != excludes_zero(clu_ci),
        "query_independent_multiplier_ci95": ind_boot_ci,
        "triple_cluster_multiplier_ci95": clu_boot_ci,
        "query_independent_multiplier_p": ind_boot_p,
        "triple_cluster_multiplier_p": clu_boot_p,
        "multiplier_flip": (ind_boot_p < 0.05) != (clu_boot_p < 0.05),
        "near_tie": abs(estimate) <= 0.005,
    }


def two_way_variance(tail: np.ndarray, head: np.ndarray, independent: bool) -> float:
    """Two-way seed/triple cluster variance with finite-cell subtraction."""
    seeds, triples = tail.shape
    pair = (tail + head) / 2.0
    seed_component = float(np.var(pair.mean(axis=1), ddof=1) / seeds)
    if independent:
        triple_component = float(
            (np.var(tail.mean(axis=0), ddof=1) + np.var(head.mean(axis=0), ddof=1))
            / (4.0 * triples)
        )
        cell_correction = float(
            (np.var(tail, ddof=1) + np.var(head, ddof=1)) / (4.0 * seeds * triples)
        )
    else:
        triple_component = float(np.var(pair.mean(axis=0), ddof=1) / triples)
        cell_correction = float(np.var(pair, ddof=1) / (seeds * triples))
    return max(seed_component + triple_component - cell_correction, 0.0)


def wild_algorithm_distributions(
    tail: np.ndarray,
    head: np.ndarray,
    repeats: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Additive two-way multiplier bootstrap for one algorithm difference."""
    pair = (tail + head) / 2.0
    estimate = float(pair.mean())
    seed_influence = pair.mean(axis=1) - estimate
    tail_triple = tail.mean(axis=0) - float(tail.mean())
    head_triple = head.mean(axis=0) - float(head.mean())
    pair_triple = pair.mean(axis=0) - estimate
    rng = np.random.default_rng(seed)
    independent = np.empty(repeats, dtype=np.float64)
    cluster = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        size = stop - start
        seed_weights = rng.standard_normal((size, len(seed_influence)))
        tail_weights = rng.standard_normal((size, len(tail_triple)))
        head_weights = rng.standard_normal((size, len(head_triple)))
        pair_weights = rng.standard_normal((size, len(pair_triple)))
        seed_error = seed_weights @ seed_influence / len(seed_influence)
        independent[start:stop] = estimate + seed_error + (
            tail_weights @ tail_triple + head_weights @ head_triple
        ) / (2.0 * len(tail_triple))
        cluster[start:stop] = estimate + seed_error + pair_weights @ pair_triple / len(pair_triple)
    return independent, cluster


def summarize_algorithm_pair(
    dataset: str,
    model_a: str,
    model_b: str,
    tail: np.ndarray,
    head: np.ndarray,
    independent: np.ndarray,
    cluster: np.ndarray,
) -> dict[str, Any]:
    pair = (tail + head) / 2.0
    seed_deltas = pair.mean(axis=1)
    estimate = float(seed_deltas.mean())
    seed_se = float(np.std(seed_deltas, ddof=1) / math.sqrt(len(seed_deltas)))
    tcrit = float(scipy_stats.t.ppf(0.975, df=len(seed_deltas) - 1))
    seed_p = float(2.0 * scipy_stats.t.sf(abs(estimate / seed_se), df=len(seed_deltas) - 1)) if seed_se else 0.0
    ind_se = math.sqrt(two_way_variance(tail, head, True))
    clu_se = math.sqrt(two_way_variance(tail, head, False))
    ind_ci = [estimate - Z975 * ind_se, estimate + Z975 * ind_se]
    clu_ci = [estimate - Z975 * clu_se, estimate + Z975 * clu_se]
    ind_errors, clu_errors = independent - estimate, cluster - estimate
    ind_boot_ci = [float(value) for value in estimate + np.quantile(ind_errors, [0.025, 0.975])]
    clu_boot_ci = [float(value) for value in estimate + np.quantile(clu_errors, [0.025, 0.975])]
    ind_boot_p, clu_boot_p = centered_p(ind_errors, estimate), centered_p(clu_errors, estimate)
    return {
        "dataset": dataset,
        "model_a": model_a,
        "model_b": model_b,
        "comparison_id": f"{dataset}__algorithm__{model_a}__minus__{model_b}",
        "delta_mrr": estimate,
        "seed_deltas": [float(value) for value in seed_deltas],
        "seed_only_se": seed_se,
        "seed_only_ci95": [estimate - tcrit * seed_se, estimate + tcrit * seed_se],
        "seed_only_p": seed_p,
        "query_independent_two_way_se": ind_se,
        "triple_cluster_two_way_se": clu_se,
        "two_way_se_ratio": clu_se / ind_se if ind_se else float("nan"),
        "query_independent_two_way_ci95": ind_ci,
        "triple_cluster_two_way_ci95": clu_ci,
        "query_independent_two_way_p": two_sided_normal_p(estimate, ind_se),
        "triple_cluster_two_way_p": two_sided_normal_p(estimate, clu_se),
        "two_way_normal_flip": excludes_zero(ind_ci) != excludes_zero(clu_ci),
        "query_independent_multiplier_ci95": ind_boot_ci,
        "triple_cluster_multiplier_ci95": clu_boot_ci,
        "query_independent_multiplier_p": ind_boot_p,
        "triple_cluster_multiplier_p": clu_boot_p,
        "multiplier_flip": (ind_boot_p < 0.05) != (clu_boot_p < 0.05),
        "near_tie": abs(estimate) <= 0.005,
    }


def flatten_same(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (list, dict))
    }
    confirmation = row.get("confirmation")
    flat["confirmation_repeats"] = confirmation["repeats"] if confirmation else ""
    flat["confirmation_query_independent_p"] = (
        confirmation["query_independent_p"] if confirmation else ""
    )
    flat["confirmation_triple_cluster_p"] = (
        confirmation["triple_cluster_p"] if confirmation else ""
    )
    flat["confirmation_query_independent_ci95"] = (
        ";".join(f"{value:.10g}" for value in confirmation["query_independent_ci95"])
        if confirmation
        else ""
    )
    flat["confirmation_triple_cluster_ci95"] = (
        ";".join(f"{value:.10g}" for value in confirmation["triple_cluster_ci95"])
        if confirmation
        else ""
    )
    flat["confirmation_confirmed_flip"] = (
        confirmation["confirmed_flip"] if confirmation else ""
    )
    return flat


def flatten_algorithm(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if not isinstance(value, list)}
    flat["seed_deltas"] = ";".join(f"{value:.10g}" for value in row["seed_deltas"])
    return flat


def plot_algorithm(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 6), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, DATASETS):
        group = [row for row in rows if row["dataset"] == dataset]
        positions = np.arange(len(group))
        for offset, prefix, label, color in (
            (-0.12, "query_independent_two_way", "query independent", "#d95f02"),
            (0.12, "triple_cluster_two_way", "source-triple cluster", "#1b9e77"),
        ):
            estimates = np.array([row["delta_mrr"] for row in group])
            cis = np.array([row[f"{prefix}_ci95"] for row in group])
            axis.errorbar(estimates, positions + offset, xerr=np.vstack([estimates-cis[:,0], cis[:,1]-estimates]), fmt="o", color=color, label=label, capsize=2)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels([f"{row['model_a']} − {row['model_b']}" for row in group], fontsize=8)
        axis.set_title(dataset)
        axis.set_xlabel("algorithm mean ΔMRR (95% CI)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def make_report(same: list[dict[str, Any]], algorithms: list[dict[str, Any]], output: Path) -> None:
    same_flips = [row for row in same if row["normal_flip"] or row["multiplier_flip"]]
    confirmed_flips = [
        row
        for row in same_flips
        if row.get("confirmation", {}).get("confirmed_flip", False)
    ]
    algorithm_flips = [row for row in algorithms if row["two_way_normal_flip"] or row["multiplier_flip"]]
    dataset_summaries = []
    for dataset in DATASETS:
        group = [row for row in same if row["dataset"] == dataset]
        ratios = np.asarray([row["se_ratio"] for row in group], dtype=np.float64)
        dataset_summaries.append(
            {
                "dataset": dataset,
                "count": len(group),
                "positive_correlation": sum(row["direction_delta_correlation"] > 0 for row in group),
                "ratio_above_one": int(np.count_nonzero(ratios > 1.0)),
                "mean_ratio": float(ratios.mean()),
                "mean_underestimation": float(np.mean(1.0 - 1.0 / ratios)),
                "near_ties": sum(row["near_tie"] for row in group),
            }
        )
    closest = sorted(
        same,
        key=lambda row: abs(row["triple_cluster_multiplier_p"] - 0.05),
    )[:5]
    lines = [
        "# BQP 多架构与算法级扩展结果",
        "",
        f"正式矩阵包含 {len(same)} 项同种子跨架构比较和 {len(algorithms)} 项算法级比较。",
        f"同种子比较中发现 {len(same_flips)} 项解析或乘子bootstrap判断翻转，其中 {len(confirmed_flips)} 项经200,000次bootstrap确认；算法级比较中发现 {len(algorithm_flips)} 项。",
        "",
        "## 标准误影响的稳定性",
        "",
        "| 数据集 | 比较数 | 正方向差值相关 | 簇SE更大 | 平均SE ratio | 查询独立SE平均低估 | 近性能比较 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_summaries:
        lines.append(
            f"| {row['dataset']} | {row['count']} | {row['positive_correlation']}/{row['count']} | "
            f"{row['ratio_above_one']}/{row['count']} | {row['mean_ratio']:.3f} | "
            f"{100.0 * row['mean_underestimation']:.1f}% | {row['near_ties']} |"
        )
    lines += [
        "",
        "四个架构的同种子差值在不同数据集上呈现不同程度的头尾相关；请结合表格中的逐数据集结果解读。",
        "源三元组簇标准误与查询独立标准误的差异跨越双线性、非对称分解和距离模型。",
        (
            "本次翻转只出现在接近显著性边界的checkpoint级比较中，不能升级为算法级排序翻转。"
            if same_flips
            else "未出现翻转是因为效应量没有落在决策边界，而不是协方差项消失。"
        ),
        "",
        "## 同种子翻转",
        "",
        "| 数据集 | seed | 比较 | ΔMRR | SE ratio | iid p | cluster p | 解析翻转 | multiplier翻转 |",
        "|---|---:|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in same_flips:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['model_a']}−{row['model_b']} | {row['delta_mrr']:.6f} | {row['se_ratio']:.3f} | {row['query_independent_normal_p']:.5f} | {row['triple_cluster_normal_p']:.5f} | {row['normal_flip']} | {row['multiplier_flip']} |"
        )
    if not same_flips:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    lines += [
        "",
        "### 200,000次bootstrap确认",
        "",
        "| 数据集 | seed | 比较 | iid p | cluster p | iid 95% CI | cluster 95% CI | 确认翻转 |",
        "|---|---:|---|---:|---:|---|---|:---:|",
    ]
    for row in same_flips:
        confirmation = row.get("confirmation")
        if not confirmation:
            continue
        iid_ci = confirmation["query_independent_ci95"]
        cluster_ci = confirmation["triple_cluster_ci95"]
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['model_a']}−{row['model_b']} | "
            f"{confirmation['query_independent_p']:.5f} | {confirmation['triple_cluster_p']:.5f} | "
            f"[{iid_ci[0]:.6f}, {iid_ci[1]:.6f}] | "
            f"[{cluster_ci[0]:.6f}, {cluster_ci[1]:.6f}] | "
            f"{confirmation['confirmed_flip']} |"
        )
    if not same_flips:
        lines.append("| — | — | — | — | — | — | — | — |")
    lines += [
        "",
        "### 最接近显著性边界的同种子比较",
        "",
        "| 数据集 | seed | 比较 | ΔMRR | SE ratio | multiplier iid p | multiplier cluster p |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in closest:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['model_a']}−{row['model_b']} | "
            f"{row['delta_mrr']:.6f} | {row['se_ratio']:.3f} | "
            f"{row['query_independent_multiplier_p']:.5f} | {row['triple_cluster_multiplier_p']:.5f} |"
        )
    lines += [
        "",
        "## 算法级比较",
        "",
        "| 数据集 | 比较 | ΔMRR | seed-only p | SE ratio | normal iid/cluster p | multiplier iid/cluster p | 翻转 |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in algorithms:
        lines.append(
            f"| {row['dataset']} | {row['model_a']}−{row['model_b']} | {row['delta_mrr']:.6f} | "
            f"{row['seed_only_p']:.5f} | {row['two_way_se_ratio']:.3f} | "
            f"{row['query_independent_two_way_p']:.5f}/{row['triple_cluster_two_way_p']:.5f} | "
            f"{row['query_independent_multiplier_p']:.5f}/{row['triple_cluster_multiplier_p']:.5f} | "
            f"{row['two_way_normal_flip'] or row['multiplier_flip']} |"
        )
    lines += [
        "",
        "![算法级置信区间](../results_bqp/architecture_extension/figures/algorithm_level_ci.png)",
        "",
        "解释边界：同种子结果属于checkpoint级证据；算法级结论必须同时参考五个seed差值、seed-only配对t区间和seed×source-triple两维区间。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "results_bqp" / "test_query_ranks")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_bqp" / "architecture_extension")
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--algorithm-repeats", type=int, default=50_000)
    parser.add_argument("--confirmation-repeats", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    device = torch.device(args.device)
    cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                cache[(dataset, model, seed)] = load_pair(input_root, dataset, model, seed)

    same_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            tail = np.stack([cache[(dataset, model, seed)][0] for model in MODELS])
            head = np.stack([cache[(dataset, model, seed)][1] for model in MODELS])
            independent, cluster = wild_checkpoint_distributions(
                tail, head, args.bootstrap_repeats,
                stable_seed(20261018, "architecture", dataset, seed), device, args.batch_size,
            )
            for index_a, index_b in itertools.combinations(range(len(MODELS)), 2):
                model_a, model_b = MODELS[index_a], MODELS[index_b]
                row = summarize_checkpoint_pair(
                    dataset, seed, model_a, model_b,
                    tail[index_a] - tail[index_b], head[index_a] - head[index_b],
                    independent[:, index_a] - independent[:, index_b],
                    cluster[:, index_a] - cluster[:, index_b],
                )
                same_rows.append(row)
    for dataset in DATASETS:
        group = [row for row in same_rows if row["dataset"] == dataset]
        ind_adj = benjamini_hochberg(row["query_independent_multiplier_p"] for row in group)
        clu_adj = benjamini_hochberg(row["triple_cluster_multiplier_p"] for row in group)
        for row, first, second in zip(group, ind_adj, clu_adj):
            row["query_independent_bh_p"] = float(first)
            row["triple_cluster_bh_p"] = float(second)
            row["bh_flip"] = bool((first < 0.05) != (second < 0.05))

    algorithm_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for index_a, index_b in itertools.combinations(range(len(MODELS)), 2):
            model_a, model_b = MODELS[index_a], MODELS[index_b]
            tail = np.stack([cache[(dataset, model_a, seed)][0] - cache[(dataset, model_b, seed)][0] for seed in SEEDS])
            head = np.stack([cache[(dataset, model_a, seed)][1] - cache[(dataset, model_b, seed)][1] for seed in SEEDS])
            independent, cluster = wild_algorithm_distributions(
                tail, head, args.algorithm_repeats,
                stable_seed(20261018, "algorithm", dataset, model_a, model_b),
                max(10, args.batch_size // 2),
            )
            algorithm_rows.append(summarize_algorithm_pair(dataset, model_a, model_b, tail, head, independent, cluster))

    candidates = [row for row in same_rows if row["normal_flip"] or row["multiplier_flip"]]
    for row in candidates:
        tail_a, head_a = cache[(row["dataset"], row["model_a"], row["seed"])]
        tail_b, head_b = cache[(row["dataset"], row["model_b"], row["seed"])]
        tail = np.stack([tail_a, tail_b])
        head = np.stack([head_a, head_b])
        independent, cluster = wild_checkpoint_distributions(
            tail, head, args.confirmation_repeats,
            stable_seed(20261018, "confirmation", row["comparison_id"]), device, args.batch_size,
        )
        estimate = row["delta_mrr"]
        ind_errors = independent[:, 0] - independent[:, 1] - estimate
        clu_errors = cluster[:, 0] - cluster[:, 1] - estimate
        ind_p, clu_p = centered_p(ind_errors, estimate), centered_p(clu_errors, estimate)
        row["confirmation"] = {
            "repeats": args.confirmation_repeats,
            "query_independent_p": ind_p,
            "triple_cluster_p": clu_p,
            "query_independent_ci95": [float(v) for v in estimate + np.quantile(ind_errors, [0.025, 0.975])],
            "triple_cluster_ci95": [float(v) for v in estimate + np.quantile(clu_errors, [0.025, 0.975])],
            "confirmed_flip": bool((ind_p < 0.05) != (clu_p < 0.05)),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "same_seed_comparisons.csv", [flatten_same(row) for row in same_rows])
    write_csv(output_root / "algorithm_level_comparisons.csv", [flatten_algorithm(row) for row in algorithm_rows])
    result = {
        "format": "bqp-architecture-extension-v1",
        "protocol": "docs/architecture_extension_protocol.md",
        "datasets": list(DATASETS), "models": list(MODELS), "seeds": list(SEEDS),
        "bootstrap_repeats": args.bootstrap_repeats,
        "algorithm_repeats": args.algorithm_repeats,
        "confirmation_repeats": args.confirmation_repeats,
        "same_seed_comparisons": same_rows,
        "algorithm_level_comparisons": algorithm_rows,
        "summary": {
            "same_seed_count": len(same_rows),
            "same_seed_normal_flips": sum(row["normal_flip"] for row in same_rows),
            "same_seed_multiplier_flips": sum(row["multiplier_flip"] for row in same_rows),
            "same_seed_bh_flips": sum(row["bh_flip"] for row in same_rows),
            "confirmed_same_seed_flips": sum(row.get("confirmation", {}).get("confirmed_flip", False) for row in same_rows),
            "algorithm_count": len(algorithm_rows),
            "algorithm_normal_flips": sum(row["two_way_normal_flip"] for row in algorithm_rows),
            "algorithm_multiplier_flips": sum(row["multiplier_flip"] for row in algorithm_rows),
        },
    }
    atomic_json(output_root / "architecture_extension_results.json", result)
    plot_algorithm(algorithm_rows, output_root / "figures" / "algorithm_level_ci")
    make_report(same_rows, algorithm_rows, ROOT / "docs" / "architecture_extension_results.md")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
