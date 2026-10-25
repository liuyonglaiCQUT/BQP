"""Independently audit BQP core statistics against frozen rank arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import (
    benjamini_hochberg,
    paired_statistics,
    relation_centered_correlation,
    summarize_bootstrap,
)
from scripts.run_bqp_core_experiment import DATASETS, MODELS, SEEDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def load_pair(rank_root: Path, dataset: str, model: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = rank_root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
    with np.load(path, allow_pickle=False) as archive:
        reciprocal = archive["reciprocal_ranks"].astype(np.float64)
        relations = archive["base_relations"]
    count = len(reciprocal) // 2
    return reciprocal[:count], reciprocal[count:], relations[:count]


def compare_analytic(
    result: dict[str, Any], tail: np.ndarray, head: np.ndarray, relations: np.ndarray
) -> None:
    recomputed = paired_statistics(tail, head)
    stored = result["analytic"]
    for field in (
        "estimate",
        "head_tail_covariance",
        "head_tail_correlation",
        "query_independent_se",
        "pooled_query_iid_se",
        "triple_cluster_se",
        "se_ratio_cluster_over_independent",
        "design_effect",
        "effective_directional_queries",
        "effective_sample_fraction",
    ):
        assert_close(float(recomputed[field]), float(stored[field]), field)
    assert_close(
        relation_centered_correlation(tail, head, relations),
        float(result["relation_control"]["relation_centered_correlation"]),
        "relation-centered correlation",
    )


def audit_distributions(
    result: dict[str, Any], prefix: str, archive: np.lib.npyio.NpzFile
) -> None:
    independent_key = prefix + "__query_independent_bootstrap"
    cluster_key = prefix + "__triple_cluster_bootstrap"
    null_key = prefix + "__relation_stratified_null_correlation"
    for key in (independent_key, cluster_key, null_key):
        if key not in archive.files:
            raise AssertionError(f"Missing distribution: {key}")
    independent = archive[independent_key]
    cluster = archive[cluster_key]
    null = archive[null_key]
    protocol_bootstrap = int(result["bootstrap"]["repeats"])
    protocol_permutation = int(result["relation_control"]["repeats"])
    if independent.shape != (protocol_bootstrap,) or cluster.shape != (protocol_bootstrap,):
        raise AssertionError(f"Wrong bootstrap shape for {prefix}")
    if null.shape != (protocol_permutation,):
        raise AssertionError(f"Wrong permutation shape for {prefix}")
    recomputed = summarize_bootstrap(independent, cluster)
    stored = result["bootstrap"]
    for field in (
        "query_independent_se",
        "triple_cluster_se",
        "se_ratio_cluster_over_independent",
        "query_independent_ci95_width",
        "triple_cluster_ci95_width",
    ):
        assert_close(float(recomputed[field]), float(stored[field]), field)
    observed = float(result["relation_control"]["observed_correlation"])
    p_value = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    assert_close(p_value, float(result["relation_control"]["p_value_greater"]), "permutation p")
    ratio_difference = abs(
        float(result["analytic"]["se_ratio_cluster_over_independent"])
        - float(result["bootstrap"]["se_ratio_cluster_over_independent"])
    )
    if ratio_difference > 0.03 or not result["audit"]["passed"]:
        raise AssertionError(f"SE-ratio audit failed for {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-root", type=Path, default=ROOT / "results_bqp" / "test_query_ranks"
    )
    parser.add_argument(
        "--core-root", type=Path, default=ROOT / "results_bqp" / "core_experiment"
    )
    args = parser.parse_args()
    rank_root = args.rank_root.resolve()
    core_root = args.core_root.resolve()
    result_path = core_root / "core_results.json"
    distributions_path = core_root / "bootstrap_distributions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("format") != "bqp-core-experiment-v1":
        raise AssertionError("Incomplete or unsupported core result")
    expected_single_count = len(DATASETS) * len(MODELS) * len(SEEDS)
    expected_comparison_count = len(DATASETS) * len(SEEDS)
    if len(result["single_model_runs"]) != expected_single_count or len(result["model_comparisons"]) != expected_comparison_count:
        raise AssertionError(
            f"Expected {expected_single_count} single runs and {expected_comparison_count} comparisons"
        )
    protocol = result["protocol"]
    if protocol["bootstrap_repeats"] != 50000 or protocol["permutation_repeats"] != 5000:
        raise AssertionError("Formal resampling counts do not match the protocol")

    cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(distributions_path, allow_pickle=False) as archive:
        expected_distribution_count = 3 * (expected_single_count + expected_comparison_count)
        if len(archive.files) != expected_distribution_count:
            raise AssertionError(f"Expected {expected_distribution_count} stored distributions")
        for item in result["single_model_runs"]:
            identity = (item["dataset"], item["model"], int(item["seed"]))
            if identity in cache:
                raise AssertionError(f"Duplicate single-run identity: {identity}")
            tail, head, relations = load_pair(rank_root, *identity)
            cache[identity] = (tail, head, relations)
            compare_analytic(item, tail, head, relations)
            audit_distributions(
                item,
                f"single__{identity[0]}__{identity[1]}__seed_{identity[2]}",
                archive,
            )
        for item in result["model_comparisons"]:
            dataset = item["dataset"]
            seed = int(item["seed"])
            complex_tail, complex_head, relations = cache[(dataset, "complex", seed)]
            distmult_tail, distmult_head, other_relations = cache[(dataset, "distmult", seed)]
            if not np.array_equal(relations, other_relations):
                raise AssertionError(f"Comparison relation mismatch: {dataset}/seed_{seed}")
            tail = complex_tail - distmult_tail
            head = complex_head - distmult_head
            compare_analytic(item, tail, head, relations)
            audit_distributions(item, f"comparison__{dataset}__seed_{seed}", archive)

    for family_name in ("single_model_runs", "model_comparisons"):
        family = result[family_name]
        recomputed_q = benjamini_hochberg(
            item["relation_control"]["p_value_greater"] for item in family
        )
        for item, q_value in zip(family, recomputed_q):
            assert_close(float(q_value), float(item["relation_permutation_q"]), "BH q")

    csv_expectations = {
        "single_model_runs.csv": expected_single_count,
        "model_comparisons.csv": expected_comparison_count,
    }
    for filename, expected_rows in csv_expectations.items():
        with (core_root / filename).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            raise AssertionError(f"{filename}: expected {expected_rows} rows")
    asset_paths = [
        core_root / "figures" / "core_effects.png",
        core_root / "figures" / "core_effects.pdf",
        core_root / "figures" / "model_comparison_ci.png",
        core_root / "figures" / "model_comparison_ci.pdf",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in asset_paths):
        raise AssertionError("Missing or empty figure asset")
    report = {
        "status": "passed",
        "format": "bqp-core-experiment-audit-v1",
        "single_model_runs": expected_single_count,
        "model_comparisons": expected_comparison_count,
        "stored_distributions": expected_distribution_count,
        "bootstrap_repeats_per_analysis": protocol["bootstrap_repeats"],
        "permutation_repeats_per_analysis": protocol["permutation_repeats"],
        "maximum_analytic_bootstrap_se_ratio_difference": max(
            item["audit"]["analytic_bootstrap_se_ratio_absolute_difference"]
            for item in result["single_model_runs"] + result["model_comparisons"]
        ),
        "files": {
            "core_results_sha256": sha256(result_path),
            "bootstrap_distributions_sha256": sha256(distributions_path),
            **{path.name + "_sha256": sha256(path) for path in asset_paths},
        },
    }
    output = core_root / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"core audit passed: {expected_single_count} runs, {expected_comparison_count} comparisons, {expected_distribution_count} distributions, "
        f"max SE-ratio drift={report['maximum_analytic_bootstrap_se_ratio_difference']:.4f}"
    )


if __name__ == "__main__":
    main()

