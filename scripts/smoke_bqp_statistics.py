#!/usr/bin/env python3
"""Fast, data-free sanity checks for the BQP statistical primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import (  # noqa: E402
    bootstrap_pair_mean,
    paired_statistics,
    relation_stratified_permutation,
)


def main() -> None:
    rng = np.random.default_rng(20261018)
    latent = rng.normal(size=2_000)
    tail = 1.0 / (1.0 + np.exp(-(latent + rng.normal(scale=0.35, size=len(latent)))))
    head = 1.0 / (1.0 + np.exp(-(latent + rng.normal(scale=0.35, size=len(latent)))))
    relations = np.arange(len(latent), dtype=np.int64) % 20

    analytic = paired_statistics(tail, head)
    if analytic["head_tail_correlation"] <= 0.0:
        raise AssertionError("Synthetic paired correlation should be positive")
    if analytic["se_ratio_cluster_over_independent"] <= 1.0:
        raise AssertionError("Cluster SE should exceed query-independent SE")

    independent, cluster = bootstrap_pair_mean(
        tail, head, repeats=500, rng=np.random.default_rng(7)
    )
    if np.std(cluster, ddof=1) <= np.std(independent, ddof=1):
        raise AssertionError("Paired bootstrap should preserve the synthetic dependence")

    null, report = relation_stratified_permutation(
        tail, head, relations, repeats=500, rng=np.random.default_rng(11)
    )
    if len(null) != 500 or not (0.0 < report["p_value_greater"] <= 1.0):
        raise AssertionError("Permutation output is malformed")

    print(
        "BQP smoke test passed: "
        f"rho={analytic['head_tail_correlation']:.3f}, "
        f"SE ratio={analytic['se_ratio_cluster_over_independent']:.3f}"
    )


if __name__ == "__main__":
    main()
