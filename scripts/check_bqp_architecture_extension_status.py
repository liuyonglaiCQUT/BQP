"""Report missing checkpoints/rank exports for the four-model BQP matrix."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("FB15k237", "WN18RR")
MODELS = ("complex", "distmult", "cp", "rotate")
SEEDS = (7, 11, 19, 23, 31)


def complete_checkpoint(dataset: str, model: str, seed: int) -> bool:
    run = ROOT / "checkpoints" / "bqp_baselines" / dataset / model / f"seed_{seed}"
    required = (run / "model.pt", run / "config.json", run / "manifest.json")
    if not all(path.is_file() for path in required):
        return False
    try:
        return json.loads((run / "manifest.json").read_text(encoding="utf-8")).get("status", "complete") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def complete_rank(dataset: str, model: str, seed: int) -> bool:
    run = ROOT / "results_bqp" / "test_query_ranks" / dataset / model / f"seed_{seed}"
    required = (run / "test_query_ranks.npz", run / "test_metrics.json")
    if not all(path.is_file() for path in required):
        return False
    try:
        return json.loads((run / "test_metrics.json").read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    checkpoint_matrix = [(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS]
    rank_matrix = [(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS]
    missing_checkpoints = ["/".join((d, m, f"seed_{s}")) for d, m, s in checkpoint_matrix if not complete_checkpoint(d, m, s)]
    missing_ranks = ["/".join((d, m, f"seed_{s}")) for d, m, s in rank_matrix if not complete_rank(d, m, s)]
    report = {
        "checkpoint_matrix": len(checkpoint_matrix),
        "complete_checkpoints": len(checkpoint_matrix) - len(missing_checkpoints),
        "missing_checkpoints": missing_checkpoints,
        "rank_matrix": len(rank_matrix),
        "complete_rank_exports": len(rank_matrix) - len(missing_ranks),
        "missing_rank_exports": missing_ranks,
        "checkpoint_archive_complete": not missing_checkpoints,
        # Statistical analysis consumes the frozen rank exports.  Checkpoints
        # are needed to reproduce those ranks, but need not be present on the
        # machine that performs the downstream analysis.
        "ready_for_analysis": not missing_ranks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
