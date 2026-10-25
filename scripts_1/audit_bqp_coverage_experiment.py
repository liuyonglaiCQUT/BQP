"""Audit identities, targets, coverage counts, and artifacts for BQP coverage."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bqp_coverage_experiment import DATASETS, MODELS_BY_DATASET, SAMPLE_SIZES, SEEDS, wilson


def load(root: Path, dataset: str, model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
    with np.load(path, allow_pickle=False) as archive:
        rr = archive["reciprocal_ranks"].astype(np.float64)
    count = len(rr) // 2
    return rr[:count], rr[count:]


def main() -> None:
    result_root = ROOT / "results_bqp" / "coverage_experiment"
    rank_root = ROOT / "results_bqp" / "test_query_ranks"
    result = json.loads((result_root / "coverage_results.json").read_text(encoding="utf-8"))
    repeats = int(result["repeats"])
    if repeats != 3000 or result["sample_sizes"] != list(SAMPLE_SIZES):
        raise AssertionError("formal repetition/sample-size protocol mismatch")
    expected_model_map = {dataset: list(MODELS_BY_DATASET[dataset]) for dataset in DATASETS}
    if result.get("models_by_dataset") != expected_model_map:
        raise AssertionError("dataset-specific architecture matrix does not match the frozen protocol")
    single = result["single_model_rows"]
    comparisons = result["comparison_rows"]
    controls = result["control_rows"]
    expected_rows = (
        sum(len(MODELS_BY_DATASET[d]) for d in DATASETS) * len(SEEDS) * len(SAMPLE_SIZES),
        sum(math.comb(len(MODELS_BY_DATASET[d]), 2) for d in DATASETS) * len(SEEDS),
        sum(2 * len(MODELS_BY_DATASET[d]) for d in DATASETS) * len(SEEDS),
    )
    if (len(single), len(comparisons), len(controls)) != expected_rows:
        raise AssertionError(f"row count mismatch: {len(single)}, {len(comparisons)}, {len(controls)}")
    cache = {
        (dataset, model, seed): load(rank_root, dataset, model, seed)
        for dataset in DATASETS for model in MODELS_BY_DATASET[dataset] for seed in SEEDS
    }
    max_target_drift = 0.0
    checked_hits = 0
    identities = set()
    for row in single:
        identity = (row["dataset"], row["model"], int(row["seed"]), int(row["sample_size"]))
        if identity in identities:
            raise AssertionError(f"duplicate single row: {identity}")
        identities.add(identity)
        tail, head = cache[(identity[0], identity[1], identity[2])]
        target = float(((tail + head) / 2.0).mean())
        max_target_drift = max(max_target_drift, abs(target - float(row["target"])))
        checked_hits += check_coverage_fields(row, repeats)
    expected_single_identities = sum(len(MODELS_BY_DATASET[d]) for d in DATASETS) * len(SEEDS) * len(SAMPLE_SIZES)
    if len(identities) != expected_single_identities:
        raise AssertionError("incomplete single-model matrix")

    identities = set()
    for row in comparisons:
        identity = (row["dataset"], row["model_a"], row["model_b"], int(row["seed"]))
        if identity in identities:
            raise AssertionError(f"duplicate comparison row: {identity}")
        identities.add(identity)
        tail_a, head_a = cache[(identity[0], identity[1], identity[3])]
        tail_b, head_b = cache[(identity[0], identity[2], identity[3])]
        target = float(((tail_a - tail_b + head_a - head_b) / 2.0).mean())
        max_target_drift = max(max_target_drift, abs(target - float(row["target"])))
        checked_hits += check_coverage_fields(row, repeats)
    expected_comparison_identities = sum(math.comb(len(MODELS_BY_DATASET[d]), 2) for d in DATASETS) * len(SEEDS)
    if len(identities) != expected_comparison_identities:
        raise AssertionError("incomplete comparison matrix")

    identities = set()
    for row in controls:
        identity = (row["dataset"], row["condition"], row["model"], int(row["seed"]))
        if identity in identities:
            raise AssertionError(f"duplicate control row: {identity}")
        identities.add(identity)
        tail, head = cache[(identity[0], identity[2], identity[3])]
        target = float(((tail + head) / 2.0).mean())
        max_target_drift = max(max_target_drift, abs(target - float(row["target"])))
        checked_hits += check_coverage_fields(row, repeats)
    expected_control_identities = sum(2 * len(MODELS_BY_DATASET[d]) for d in DATASETS) * len(SEEDS)
    if len(identities) != expected_control_identities:
        raise AssertionError("incomplete control matrix")
    if max_target_drift > 1e-12:
        raise AssertionError(f"target drift: {max_target_drift}")

    required = [
        result_root / "single_model_coverage.csv",
        result_root / "comparison_coverage.csv",
        result_root / "depairing_control.csv",
        result_root / "figures" / "single_model_coverage.png",
        result_root / "figures" / "single_model_coverage.pdf",
        result_root / "figures" / "depairing_control.png",
        result_root / "figures" / "depairing_control.pdf",
        ROOT / "docs" / "coverage_experiment_results.md",
        ROOT / "docs" / "coverage_experiment_protocol.md",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing artifact: {path}")
    summary = {
        "status": "passed",
        "rank_runs": len(cache),
        "single_model_rows": len(single),
        "comparison_rows": len(comparisons),
        "control_rows": len(controls),
        "coverage_fields_checked": checked_hits,
        "max_target_recompute_drift": max_target_drift,
    }
    (result_root / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def check_coverage_fields(row: dict, repeats: int) -> int:
    checked = 0
    for prefix in ("query_independent", "triple_cluster"):
        hits = int(row[f"{prefix}_hits"])
        coverage = float(row[f"{prefix}_coverage"])
        if not 0 <= hits <= repeats or not math.isclose(coverage, hits / repeats, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"coverage/hit mismatch: {prefix}")
        if not np.allclose(row[f"{prefix}_wilson95"], wilson(hits, repeats), rtol=0.0, atol=1e-15):
            raise AssertionError(f"Wilson interval mismatch: {prefix}")
        checked += 1
    gap = float(row["triple_cluster_coverage"] - row["query_independent_coverage"])
    if not math.isclose(gap, float(row["coverage_gap_cluster_minus_independent"]), rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("coverage gap mismatch")
    return checked


if __name__ == "__main__":
    main()
