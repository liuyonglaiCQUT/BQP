"""Audit the exhaustive near-tie BQP comparison experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("FB15k237", "WN18RR")
MODELS = ("complex", "distmult")
SEEDS = (7, 11, 19, 23, 31)
RUNS = tuple((model, seed) for model in MODELS for seed in SEEDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ordered_pair(first: tuple[str, int], second: tuple[str, int]) -> tuple[tuple[str, int], tuple[str, int]]:
    if first[0] != second[0]:
        return (first, second) if first[0] == "complex" else (second, first)
    return (first, second) if first[1] < second[1] else (second, first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path, default=ROOT / "results_bqp" / "near_tie_experiment"
    )
    parser.add_argument(
        "--rank-root", type=Path, default=ROOT / "results_bqp" / "test_query_ranks"
    )
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    rank_root = args.rank_root.resolve()
    result_path = result_root / "near_tie_results.json"
    csv_path = result_root / "all_pair_comparisons.csv"
    distributions_path = result_root / "all_pair_bootstrap_distributions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("format") != "bqp-near-tie-all-pairs-v1":
        raise AssertionError("Unexpected result format")
    comparisons = result["comparisons"]
    if len(comparisons) != 90:
        raise AssertionError(f"Expected 90 comparisons, found {len(comparisons)}")
    dataset_counts = Counter(item["dataset"] for item in comparisons)
    if dataset_counts != Counter({"FB15k237": 45, "WN18RR": 45}):
        raise AssertionError(f"Unexpected dataset counts: {dataset_counts}")
    category_counts = Counter(item["category"] for item in comparisons)
    expected_categories = Counter(
        {
            "same_model_cross_seed": 40,
            "cross_model_same_seed": 10,
            "cross_model_cross_seed": 40,
        }
    )
    if category_counts != expected_categories:
        raise AssertionError(f"Unexpected category counts: {category_counts}")
    identifiers = [item["comparison_id"] for item in comparisons]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Duplicate comparison identifiers")
    for dataset in DATASETS:
        observed = {
            ((item["model_a"], item["seed_a"]), (item["model_b"], item["seed_b"]))
            for item in comparisons
            if item["dataset"] == dataset
        }
        expected = {ordered_pair(first, second) for first, second in itertools.combinations(RUNS, 2)}
        if observed != expected:
            raise AssertionError(f"Incomplete pair universe: {dataset}")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 90:
        raise AssertionError(f"Expected 90 CSV rows, found {len(csv_rows)}")
    if {row["comparison_id"] for row in csv_rows} != set(identifiers):
        raise AssertionError("CSV/JSON comparison identifiers disagree")
    if result["artifacts"]["comparisons_csv_sha256"] != sha256(csv_path):
        raise AssertionError("CSV hash mismatch")
    if result["artifacts"]["bootstrap_distributions_sha256"] != sha256(distributions_path):
        raise AssertionError("Bootstrap distribution hash mismatch")

    with np.load(distributions_path, allow_pickle=False) as distributions:
        expected_initial = {
            f"{identifier}__initial_{kind}"
            for identifier in identifiers
            for kind in ("query_independent", "triple_cluster")
        }
        if not expected_initial.issubset(set(distributions.files)):
            raise AssertionError("Missing initial bootstrap distributions")
        repeats = result["protocol"]["bootstrap_repeats"]
        for key in expected_initial:
            if distributions[key].shape != (repeats,):
                raise AssertionError(f"Unexpected distribution shape: {key}")
        for item in comparisons:
            confirmation = item["confirmation"]
            requires_confirmation = bool(
                item["inference"]["bootstrap_flip_candidate"]
                or item["inference"]["normal_decision_flip"]
                or item["inference"]["centered_bootstrap_p_decision_flip"]
            )
            if bool(confirmation) != requires_confirmation:
                raise AssertionError(f"Confirmation coverage mismatch: {item['comparison_id']}")
            if confirmation:
                for kind in ("query_independent", "triple_cluster"):
                    key = f"{item['comparison_id']}__confirmation_{kind}"
                    if distributions[key].shape != (result["protocol"]["confirmation_repeats"],):
                        raise AssertionError(f"Unexpected confirmation shape: {key}")
                if bool(confirmation["decision_flip"]) != bool(
                    item["inference"]["confirmed_bootstrap_ci_flip"]
                ):
                    raise AssertionError(f"Confirmation decision mismatch: {item['comparison_id']}")
                if bool(confirmation["centered_p_decision_flip"]) != bool(
                    item["inference"]["confirmed_centered_bootstrap_p_flip"]
                ):
                    raise AssertionError(f"Confirmation p-value mismatch: {item['comparison_id']}")

    # Recompute every point estimate directly from frozen reciprocal-rank arrays.
    cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for dataset in DATASETS:
        for model, seed in RUNS:
            path = rank_root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
            with np.load(path, allow_pickle=False) as archive:
                reciprocal = archive["reciprocal_ranks"].astype(np.float64)
            count = len(reciprocal) // 2
            cache[(dataset, model, seed)] = (reciprocal[:count], reciprocal[count:])
    maximum_estimate_drift = 0.0
    for item in comparisons:
        tail_a, head_a = cache[(item["dataset"], item["model_a"], item["seed_a"])]
        tail_b, head_b = cache[(item["dataset"], item["model_b"], item["seed_b"])]
        estimate = float(np.mean(np.concatenate([tail_a - tail_b, head_a - head_b])))
        drift = abs(estimate - float(item["analytic"]["estimate"]))
        maximum_estimate_drift = max(maximum_estimate_drift, drift)
        if drift > 1e-12:
            raise AssertionError(f"Point-estimate mismatch: {item['comparison_id']}")

    for name in (
        "all_pair_significance_map.png",
        "all_pair_significance_map.pdf",
        "boundary_comparison_ci.png",
        "boundary_comparison_ci.pdf",
    ):
        path = result_root / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing figure: {path}")

    audit = {
        "status": "passed",
        "format": "bqp-near-tie-audit-v1",
        "comparisons": len(comparisons),
        "dataset_counts": dict(dataset_counts),
        "category_counts": dict(category_counts),
        "near_ties": sum(item["near_tie"] for item in comparisons),
        "bootstrap_flip_candidates": sum(
            item["inference"]["bootstrap_flip_candidate"] for item in comparisons
        ),
        "confirmed_bootstrap_ci_flips": sum(
            item["inference"]["confirmed_bootstrap_ci_flip"] for item in comparisons
        ),
        "centered_bootstrap_p_flips": sum(
            item["inference"]["centered_bootstrap_p_decision_flip"]
            for item in comparisons
        ),
        "confirmed_centered_bootstrap_p_flips": sum(
            item["inference"]["confirmed_centered_bootstrap_p_flip"]
            for item in comparisons
        ),
        "normal_decision_flips": sum(
            item["inference"]["normal_decision_flip"] for item in comparisons
        ),
        "bh_decision_flips": sum(
            item["inference"]["bh_decision_flip"] for item in comparisons
        ),
        "maximum_point_estimate_drift": maximum_estimate_drift,
        "hashes": {
            "near_tie_results_sha256": sha256(result_path),
            "all_pair_comparisons_sha256": sha256(csv_path),
            "all_pair_bootstrap_distributions_sha256": sha256(distributions_path),
        },
    }
    atomic_json(result_root / "audit.json", audit)
    print(
        "near-tie audit passed: "
        f"{audit['comparisons']} comparisons, {audit['near_ties']} near ties, "
        f"{audit['confirmed_centered_bootstrap_p_flips']} confirmed centered-p flips",
        flush=True,
    )


if __name__ == "__main__":
    main()

