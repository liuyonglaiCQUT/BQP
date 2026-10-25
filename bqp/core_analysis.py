"""Statistical estimators for Bidirectional Query Pseudoreplication."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = math.sqrt(
        float(np.dot(first_centered, first_centered))
        * float(np.dot(second_centered, second_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def paired_statistics(tail: np.ndarray, head: np.ndarray) -> dict[str, float | int]:
    tail = np.asarray(tail, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    if tail.ndim != 1 or head.shape != tail.shape or len(tail) < 2:
        raise ValueError("tail and head must be aligned one-dimensional arrays")
    count = len(tail)
    tail_variance = float(np.var(tail, ddof=1))
    head_variance = float(np.var(head, ddof=1))
    covariance = float(np.cov(tail, head, ddof=1)[0, 1])
    pair_means = (tail + head) / 2.0
    estimate = float(pair_means.mean())
    independent_variance = (tail_variance + head_variance) / (4.0 * count)
    cluster_variance = float(np.var(pair_means, ddof=1) / count)
    pooled_variance = float(np.var(np.concatenate([tail, head]), ddof=1) / (2 * count))
    independent_se = math.sqrt(max(independent_variance, 0.0))
    cluster_se = math.sqrt(max(cluster_variance, 0.0))
    pooled_se = math.sqrt(max(pooled_variance, 0.0))
    design_effect = (
        cluster_variance / independent_variance
        if independent_variance > 0.0
        else float("nan")
    )
    se_ratio = (
        cluster_se / independent_se if independent_se > 0.0 else float("nan")
    )
    return {
        "source_triples": count,
        "directional_queries": 2 * count,
        "estimate": estimate,
        "tail_mean": float(tail.mean()),
        "head_mean": float(head.mean()),
        "tail_variance": tail_variance,
        "head_variance": head_variance,
        "head_tail_covariance": covariance,
        "head_tail_correlation": _correlation(tail, head),
        "query_independent_se": independent_se,
        "pooled_query_iid_se": pooled_se,
        "triple_cluster_se": cluster_se,
        "se_ratio_cluster_over_independent": se_ratio,
        "se_ratio_cluster_over_pooled_iid": (
            cluster_se / pooled_se if pooled_se > 0.0 else float("nan")
        ),
        "design_effect": design_effect,
        "effective_directional_queries": (
            2.0 * count / design_effect if design_effect > 0.0 else float("nan")
        ),
        "effective_sample_fraction": (
            1.0 / design_effect if design_effect > 0.0 else float("nan")
        ),
        "independent_se_underestimation_fraction": (
            1.0 - independent_se / cluster_se if cluster_se > 0.0 else float("nan")
        ),
    }


def bootstrap_pair_mean(
    tail: np.ndarray,
    head: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    batch_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direction-independent and source-triple-cluster bootstrap means."""
    tail = np.asarray(tail, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    if repeats < 2:
        raise ValueError("bootstrap repeats must be at least 2")
    count = len(tail)
    pair_means = (tail + head) / 2.0
    independent = np.empty(repeats, dtype=np.float64)
    cluster = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        size = stop - start
        tail_indices = rng.integers(0, count, size=(size, count), dtype=np.int32)
        head_indices = rng.integers(0, count, size=(size, count), dtype=np.int32)
        cluster_indices = rng.integers(0, count, size=(size, count), dtype=np.int32)
        independent[start:stop] = (
            tail[tail_indices].mean(axis=1) + head[head_indices].mean(axis=1)
        ) / 2.0
        cluster[start:stop] = pair_means[cluster_indices].mean(axis=1)
    return independent, cluster


def summarize_bootstrap(
    independent: np.ndarray,
    cluster: np.ndarray,
) -> dict[str, float | int | list[float]]:
    independent_se = float(np.std(independent, ddof=1))
    cluster_se = float(np.std(cluster, ddof=1))
    independent_ci = np.quantile(independent, [0.025, 0.975])
    cluster_ci = np.quantile(cluster, [0.025, 0.975])
    return {
        "repeats": int(len(independent)),
        "query_independent_se": independent_se,
        "triple_cluster_se": cluster_se,
        "se_ratio_cluster_over_independent": (
            cluster_se / independent_se if independent_se > 0.0 else float("nan")
        ),
        "query_independent_ci95": [float(value) for value in independent_ci],
        "triple_cluster_ci95": [float(value) for value in cluster_ci],
        "query_independent_ci95_width": float(independent_ci[1] - independent_ci[0]),
        "triple_cluster_ci95_width": float(cluster_ci[1] - cluster_ci[0]),
    }


def relation_centered_correlation(
    tail: np.ndarray,
    head: np.ndarray,
    relations: np.ndarray,
) -> float:
    tail = np.asarray(tail, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    relations = np.asarray(relations)
    tail_centered = np.empty_like(tail)
    head_centered = np.empty_like(head)
    for relation in np.unique(relations):
        mask = relations == relation
        tail_centered[mask] = tail[mask] - tail[mask].mean()
        head_centered[mask] = head[mask] - head[mask].mean()
    return _correlation(tail_centered, head_centered)


def relation_stratified_permutation(
    tail: np.ndarray,
    head: np.ndarray,
    relations: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    batch_size: int = 50,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Permute head outcomes within relation while preserving both marginals."""
    tail = np.asarray(tail, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    relations = np.asarray(relations)
    tail_centered = tail - tail.mean()
    head_centered = head - head.mean()
    denominator = math.sqrt(
        float(np.dot(tail_centered, tail_centered))
        * float(np.dot(head_centered, head_centered))
    )
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant direction")
    groups = [np.flatnonzero(relations == value) for value in np.unique(relations)]
    movable = sum(len(group) for group in groups if len(group) > 1)
    null = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        size = stop - start
        cross_products = np.zeros(size, dtype=np.float64)
        for indices in groups:
            group_head = head_centered[indices]
            if len(indices) > 1:
                repeated = np.broadcast_to(group_head, (size, len(indices)))
                permuted = rng.permuted(repeated, axis=1)
            else:
                permuted = np.broadcast_to(group_head, (size, 1))
            cross_products += np.sum(
                permuted * tail_centered[indices][None, :], axis=1
            )
        null[start:stop] = cross_products / denominator
    observed = _correlation(tail, head)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (repeats + 1))
    quantiles = np.quantile(null, [0.025, 0.5, 0.975])
    metadata: dict[str, float | int] = {
        "repeats": repeats,
        "observed_correlation": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_q025": float(quantiles[0]),
        "null_median": float(quantiles[1]),
        "null_q975": float(quantiles[2]),
        "p_value_greater": p_value,
        "relation_strata": len(groups),
        "movable_fraction": float(movable / len(tail)),
    }
    return null, metadata


def two_sided_normal_p(estimate: float, standard_error: float) -> float:
    if standard_error <= 0.0:
        return float("nan")
    return float(math.erfc(abs(estimate / standard_error) / math.sqrt(2.0)))


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=np.float64)
    ranked = values[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted

