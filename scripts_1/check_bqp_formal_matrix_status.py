"""Show completeness of the BQP core, architecture, checkpoint, and rank matrices."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 11, 19, 23, 31)
MODELS_BY_DATASET = {
    "FB15k237": ("complex", "distmult", "cp", "rotate"),
    "WN18RR": ("complex", "distmult", "cp", "rotate"),
    "NELL995": ("complex", "distmult"),
}
CORE_MODELS = ("complex", "distmult")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_complete(dataset: str, model: str, seed: int) -> tuple[bool, str | None]:
    run = ROOT / "checkpoints" / "bqp_baselines" / dataset / model / f"seed_{seed}"
    required = (run / "model.pt", run / "config.json", run / "manifest.json")
    if not all(path.is_file() for path in required):
        return False, None
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") not in {None, "complete"} or "selected_epoch" not in manifest:
        return False, None
    config = read_json(run / "config.json")
    if (
        config.get("dataset", {}).get("name") != dataset
        or config.get("model", {}).get("name") != model
        or int(manifest.get("seed", seed)) != seed
    ):
        return False, "identity_mismatch"
    if dataset == "NELL995":
        dataset_manifest = ROOT / "data" / dataset / "dataset_manifest.json"
        recorded = config.get("dataset", {}).get("split_manifest_sha256")
        if not dataset_manifest.is_file():
            return False, "split_manifest_missing"
        import hashlib

        digest = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
        if recorded is None:
            return False, "split_manifest_hash_missing"
        if recorded != digest:
            return False, "split_manifest_hash_mismatch"
    return True, None


def rank_complete(dataset: str, model: str, seed: int) -> bool:
    run = ROOT / "results_bqp" / "test_query_ranks" / dataset / model / f"seed_{seed}"
    return (
        (run / "test_query_ranks.npz").is_file()
        and read_json(run / "test_metrics.json").get("status") == "complete"
    )


def main() -> None:
    identities = [
        (dataset, model, seed)
        for dataset, models in MODELS_BY_DATASET.items()
        for model in models
        for seed in SEEDS
    ]
    core_identities = [
        row for row in identities if row[1] in CORE_MODELS
    ]
    checkpoint_errors: dict[str, str] = {}
    complete_checkpoints = []
    complete_ranks = []
    historical_hash_matches = []
    historical_hash_missing = []
    locally_retrained_checkpoints = []
    historical_identities = [
        (dataset, model, seed)
        for dataset in ("FB15k237", "WN18RR")
        for model in ("complex", "distmult", "cp", "rotate")
        for seed in SEEDS
    ]
    historical_identity_set = set(historical_identities)
    for dataset, model, seed in identities:
        label = f"{dataset}/{model}/seed_{seed}"
        run_dir = ROOT / "checkpoints" / "bqp_baselines" / dataset / model / f"seed_{seed}"
        complete, error = checkpoint_complete(dataset, model, seed)
        if complete:
            complete_checkpoints.append(label)
            if (dataset, model, seed) in historical_identity_set:
                historical_rank_dir = ROOT / "results_bqp" / "historical_architecture_ranks" / dataset / model / f"seed_{seed}"
                if not historical_rank_dir.is_dir():
                    historical_rank_dir = ROOT / "results_bqp" / "test_query_ranks" / dataset / model / f"seed_{seed}"
                historical_metrics = read_json(historical_rank_dir / "test_metrics.json")
                expected_hash = historical_metrics.get("files", {}).get("checkpoint_sha256")
                if expected_hash and (run_dir / "model.pt").is_file():
                    if sha256(run_dir / "model.pt") == expected_hash:
                        historical_hash_matches.append(label)
                    elif read_json(run_dir / "manifest.json").get("training_origin") == "local_retraining":
                        locally_retrained_checkpoints.append(label)
                    else:
                        historical_hash_missing.append(label)
                else:
                    historical_hash_missing.append(label)
        elif error:
            checkpoint_errors[label] = error
        if rank_complete(dataset, model, seed):
            complete_ranks.append(label)

    nell_audit = read_json(ROOT / "data" / "NELL995" / "split_audit.json")
    missing_checkpoints = sorted(set(f"{d}/{m}/seed_{s}" for d, m, s in identities) - set(complete_checkpoints))
    missing_ranks = sorted(set(f"{d}/{m}/seed_{s}" for d, m, s in identities) - set(complete_ranks))
    core_rank_count = sum(label.split("/")[1] in CORE_MODELS for label in complete_ranks)
    report = {
        "formal_matrix_runs": len(identities),
        "formal_checkpoints_complete": len(complete_checkpoints),
        "formal_ranks_complete": len(complete_ranks),
        "missing_checkpoints": missing_checkpoints,
        "missing_ranks": missing_ranks,
        "checkpoint_identity_errors": checkpoint_errors,
        "historical_original_checkpoint_hash_matches": len(historical_hash_matches),
        "historical_original_checkpoint_hash_matches_expected": len(historical_identities),
        "historical_checkpoint_hash_unavailable_or_mismatched": historical_hash_missing,
        "locally_retrained_checkpoint_count": len(locally_retrained_checkpoints),
        "locally_retrained_checkpoints": locally_retrained_checkpoints,
        "core_rank_runs": core_rank_count,
        "core_rank_runs_expected": len(core_identities),
        "core_analysis_ready": core_rank_count == len(core_identities),
        "full_rank_matrix_ready": not missing_ranks,
        "historical_checkpoint_archive_complete": len(historical_hash_matches) == len(historical_identities),
        "nell_split_audit_status": nell_audit.get("status", "missing"),
        "nell_all_entities_in_train": nell_audit.get("num_entities_absent_from_train") == 0,
    }
    output_path = ROOT / "results_bqp" / "formal_matrix_status.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
