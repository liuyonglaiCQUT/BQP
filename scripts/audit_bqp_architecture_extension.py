"""Independent structural and numerical audit of the BQP architecture extension."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.core_analysis import paired_statistics

DATASETS = ("FB15k237", "WN18RR")
MODELS = ("complex", "distmult", "cp", "rotate")
SEEDS = (7, 11, 19, 23, 31)


def load(root: Path, dataset: str, model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / dataset / model / f"seed_{seed}" / "test_query_ranks.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        rr = archive["reciprocal_ranks"].astype(np.float64)
        source = archive["source_triple_ids"]
        direction = archive["directions"]
    count = len(rr) // 2
    expected = np.arange(count, dtype=source.dtype)
    if not np.array_equal(source[:count], expected) or not np.array_equal(source[count:], expected):
        raise AssertionError(f"unaligned source ids: {path}")
    if not np.all(direction[:count] == 0) or not np.all(direction[count:] == 1):
        raise AssertionError(f"unaligned directions: {path}")
    return rr[:count], rr[count:]


def close(first: float, second: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    rank_root = ROOT / "results_bqp" / "test_query_ranks"
    result_root = ROOT / "results_bqp" / "architecture_extension"
    result_path = result_root / "architecture_extension_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["datasets"] != list(DATASETS) or result["models"] != list(MODELS) or result["seeds"] != list(SEEDS):
        raise AssertionError("matrix identity mismatch")
    same = result["same_seed_comparisons"]
    algorithms = result["algorithm_level_comparisons"]
    expected_same = len(DATASETS) * len(SEEDS) * 6
    expected_algorithms = len(DATASETS) * 6
    if len(same) != expected_same or len(algorithms) != expected_algorithms:
        raise AssertionError(f"wrong comparison counts: {len(same)}, {len(algorithms)}")
    cache = {
        (dataset, model, seed): load(rank_root, dataset, model, seed)
        for dataset in DATASETS for model in MODELS for seed in SEEDS
    }
    seen_same: set[tuple[str, int, str, str]] = set()
    max_delta_drift = 0.0
    max_se_drift = 0.0
    confirmed_flip_count = 0
    for row in same:
        key = (row["dataset"], int(row["seed"]), row["model_a"], row["model_b"])
        if key in seen_same:
            raise AssertionError(f"duplicate same-seed comparison: {key}")
        seen_same.add(key)
        ta, ha = cache[(key[0], key[2], key[1])]
        tb, hb = cache[(key[0], key[3], key[1])]
        recomputed = paired_statistics(ta - tb, ha - hb)
        max_delta_drift = max(max_delta_drift, abs(float(recomputed["estimate"]) - row["delta_mrr"]))
        max_se_drift = max(
            max_se_drift,
            abs(float(recomputed["query_independent_se"]) - row["query_independent_se"]),
            abs(float(recomputed["triple_cluster_se"]) - row["triple_cluster_se"]),
        )
        if row["normal_flip"] or row["multiplier_flip"]:
            confirmation = row.get("confirmation")
            if not confirmation or int(confirmation["repeats"]) != int(result["confirmation_repeats"]):
                raise AssertionError(f"missing confirmation: {row['comparison_id']}")
            recomputed_confirmation_flip = bool(
                (float(confirmation["query_independent_p"]) < 0.05)
                != (float(confirmation["triple_cluster_p"]) < 0.05)
            )
            if bool(confirmation["confirmed_flip"]) != recomputed_confirmation_flip:
                raise AssertionError(f"confirmation decision mismatch: {row['comparison_id']}")
            confirmed_flip_count += int(recomputed_confirmation_flip)
    if max_delta_drift > 1e-11 or max_se_drift > 1e-11:
        raise AssertionError(f"numerical drift: delta={max_delta_drift}, se={max_se_drift}")
    if confirmed_flip_count != int(result["summary"]["confirmed_same_seed_flips"]):
        raise AssertionError("confirmed flip summary mismatch")

    seen_algorithms: set[tuple[str, str, str]] = set()
    for row in algorithms:
        key = (row["dataset"], row["model_a"], row["model_b"])
        if key in seen_algorithms:
            raise AssertionError(f"duplicate algorithm comparison: {key}")
        seen_algorithms.add(key)
        seed_deltas = []
        for seed in SEEDS:
            ta, ha = cache[(key[0], key[1], seed)]
            tb, hb = cache[(key[0], key[2], seed)]
            seed_deltas.append(float(((ta - tb + ha - hb) / 2.0).mean()))
        if not np.allclose(seed_deltas, row["seed_deltas"], rtol=0.0, atol=1e-11):
            raise AssertionError(f"seed deltas mismatch: {key}")
        if not close(float(np.mean(seed_deltas)), row["delta_mrr"]):
            raise AssertionError(f"algorithm estimate mismatch: {key}")
    required = [
        result_root / "same_seed_comparisons.csv",
        result_root / "algorithm_level_comparisons.csv",
        result_root / "figures" / "algorithm_level_ci.png",
        result_root / "figures" / "algorithm_level_ci.pdf",
        ROOT / "docs" / "architecture_extension_results.md",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing output: {path}")
    summary = {
        "status": "passed",
        "rank_runs": len(cache),
        "same_seed_comparisons": len(same),
        "algorithm_level_comparisons": len(algorithms),
        "max_delta_recompute_drift": max_delta_drift,
        "max_se_recompute_drift": max_se_drift,
        "confirmed_same_seed_flips": result["summary"]["confirmed_same_seed_flips"],
        "algorithm_multiplier_flips": result["summary"]["algorithm_multiplier_flips"],
    }
    (result_root / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
