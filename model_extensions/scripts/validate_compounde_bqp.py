"""Independently audit a CompoundE adaptation run, then perform BQP inference."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "adapters"), str(ROOT / "scripts")]
from bqp_statistics import (bootstrap_pair_mean, paired_statistics,
                            relation_stratified_permutation, summarize_bootstrap)
from compounde import CompoundE
from run_compounde import digest, evaluate, filters, metrics_from_ranks, read_data


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_selection(path: Path) -> tuple[int, float, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("Validation trajectory is empty")
    steps = [int(row["step"]) for row in rows]
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError("Validation trajectory steps are duplicated or unordered")
    maximum = max(float(row["mrr"]) for row in rows)
    selected = next(row for row in rows if float(row["mrr"]) == maximum)
    selected_flags = [row for row in rows if row.get("selected")]
    if not selected_flags or int(selected_flags[-1]["step"]) != int(selected["step"]):
        raise ValueError("Recorded validation selections do not identify the first maximum MRR")
    return int(selected["step"]), maximum, rows


def validate_export_arrays(archive, expected_count: int) -> None:
    required = {
        "ranks_harmonized_average", "triples_hrt", "raw_triples_hrt",
        "source_triple_ids", "directions", "filter_protocol", "tie_policy",
        "selected_checkpoint_step", "selected_checkpoint_sha256",
    }
    missing = required.difference(archive.files)
    if missing:
        raise ValueError("Rank export is missing fields: " + ", ".join(sorted(missing)))
    ranks = archive["ranks_harmonized_average"]
    if ranks.shape != (expected_count, 2) or not np.isfinite(ranks).all() or np.any(ranks < 1):
        raise ValueError(f"Invalid rank matrix: {ranks.shape}")
    if archive["triples_hrt"].shape != (expected_count, 3):
        raise ValueError("Invalid integer triple matrix")
    if archive["raw_triples_hrt"].shape != (expected_count, 3):
        raise ValueError("Invalid raw triple matrix")
    if not np.array_equal(archive["source_triple_ids"], np.arange(expected_count)):
        raise ValueError("Source triple IDs do not preserve test order")
    if archive["directions"].tolist() != ["head", "tail"]:
        raise ValueError("Expected rank columns [head, tail]")
    if str(archive["filter_protocol"].item()) != "all_known_train_valid_test":
        raise ValueError("Unexpected filter protocol")
    if str(archive["tie_policy"].item()) != "average":
        raise ValueError("Unexpected tie policy")


def analyze(ranks: np.ndarray, relations: np.ndarray, repeats: int, seed: int):
    head, tail = 1.0 / ranks[:, 0], 1.0 / ranks[:, 1]
    independent, cluster = bootstrap_pair_mean(
        tail, head, repeats, np.random.default_rng(seed))
    null, permutation = relation_stratified_permutation(
        tail, head, relations, repeats, np.random.default_rng(seed + 1))
    report = {
        "analytic": paired_statistics(tail, head),
        "bootstrap": summarize_bootstrap(independent, cluster),
        "within_relation_repairing": permutation,
    }
    samples = {
        "bootstrap_independent": independent,
        "bootstrap_cluster": cluster,
        "within_relation_null_correlation": null,
    }
    return report, samples


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["WN18RR", "FB15k237"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-reference-gap", type=float, default=0.03)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--entity-chunk", type=int, default=1024)
    arguments = parser.parse_args()
    if arguments.repeats < 100:
        raise ValueError("Use at least 100 repeats")
    if arguments.max_reference_gap < 0:
        raise ValueError("max-reference-gap must be nonnegative")
    if str(arguments.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run = ROOT / "runs" / "CompoundE" / arguments.dataset / f"seed_{arguments.seed}"
    paths = {
        name: run / filename for name, filename in {
            "config": "config.json", "vocabulary": "vocabulary.json",
            "checkpoint": "best.pt", "trajectory": "valid_trajectory.jsonl",
            "rank_export": "test_triple_ranks.npz", "test_metrics": "test_metrics.json",
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_json(paths["config"])
    vocabulary = load_json(paths["vocabulary"])
    logged_metrics = load_json(paths["test_metrics"])
    arrays, entities, relations, raw, raw_hashes = read_data(arguments.dataset)
    if raw_hashes != config.get("raw_split_sha256"):
        raise ValueError("Raw split hash mismatch")
    if vocabulary != {"entities": entities, "relations": relations}:
        raise ValueError("Vocabulary labels/order no longer match the training run")
    if config.get("entities") != len(entities) or config.get("relations") != len(relations):
        raise ValueError("Config vocabulary sizes do not match the data")
    current_training_sources = {
        "adapter": digest(ROOT / "adapters" / "compounde.py"),
        "runner": digest(ROOT / "scripts" / "run_compounde.py"),
    }
    if config.get("training_source_sha256") != current_training_sources:
        raise ValueError("Training implementation changed after the checkpoint was created")

    selected_step, selected_valid_mrr, trajectory = trajectory_selection(paths["trajectory"])
    checkpoint = torch.load(paths["checkpoint"], map_location=arguments.device)
    if checkpoint.get("format") != "compounde-table11-adaptation-checkpoint-v2":
        raise ValueError("Unsupported CompoundE checkpoint format")
    if int(checkpoint["step"]) != selected_step or int(checkpoint["best_step"]) != selected_step:
        raise ValueError("best.pt does not match the validation-selected step")
    if abs(float(checkpoint["best"]) - selected_valid_mrr) > 1e-12:
        raise ValueError("best.pt validation MRR does not match the trajectory")
    checkpoint_hash = digest(paths["checkpoint"])

    archive = np.load(paths["rank_export"])
    validate_export_arrays(archive, len(arrays["test"]))
    exported_ranks = archive["ranks_harmonized_average"].astype(np.float64)
    exported_triples = archive["triples_hrt"].astype(np.int64)
    exported_raw = archive["raw_triples_hrt"].astype(str)
    if not np.array_equal(exported_triples, arrays["test"]):
        raise ValueError("Exported integer triples do not match the source test split")
    if not np.array_equal(exported_raw, np.asarray(raw["test"], dtype=str)):
        raise ValueError("Exported raw triples do not preserve test identity/order")
    if int(archive["selected_checkpoint_step"].item()) != selected_step:
        raise ValueError("Rank export checkpoint step mismatch")
    if str(archive["selected_checkpoint_sha256"].item()) != checkpoint_hash:
        raise ValueError("Rank export checkpoint hash mismatch")
    if logged_metrics.get("selected_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Test metric checkpoint hash mismatch")

    model = CompoundE(len(entities), len(relations), int(config["dim"]),
                      float(config["gamma"]), config["variant"]).to(arguments.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    known = filters(np.concatenate([arrays[split] for split in ("train", "valid", "test")]))
    recomputed_ranks, recomputed_metrics = evaluate(
        model, arrays["test"], known, arguments.eval_batch, arguments.entity_chunk,
        arguments.device, progress=True)
    rank_drift = float(np.max(np.abs(recomputed_ranks - exported_ranks)))
    metric_names = ("mrr", "hits@1", "hits@3", "hits@10")
    metric_drift = {name: abs(recomputed_metrics[name] - float(logged_metrics[name]))
                    for name in metric_names}
    metric_audit_passed = rank_drift == 0.0 and max(metric_drift.values()) <= 1e-12
    paper_reference = {name: float(value)
                       for name, value in config["reference_metrics"].items()}
    reference_gaps = {name: recomputed_metrics[name] - paper_reference[name]
                      for name in metric_names}
    performance_plausibility_passed = (
        abs(reference_gaps["mrr"]) <= arguments.max_reference_gap)

    output = (ROOT / "results" / "BQP_CompoundE" / arguments.dataset /
              f"seed_{arguments.seed}" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    audit = {
        "model": "CompoundE",
        "implementation": "paper Table-11 architecture/training adaptation",
        "dataset": arguments.dataset,
        "seed": arguments.seed,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": digest(paths["config"]),
        "training_source_sha256": current_training_sources,
        "selected_checkpoint_step": selected_step,
        "selected_validation_mrr": selected_valid_mrr,
        "validation_points": len(trajectory),
        "selection_audit": "first maximum validation MRR; test metrics not used for selection",
        "raw_split_sha256": raw_hashes,
        "source_triples": len(arrays["test"]),
        "directional_queries": 2 * len(arrays["test"]),
        "source_identity_audit": "exact raw and mapped test row identity/order",
        "protocols": {
            "official_public_repository": {
                "available_for_dataset": False,
                "reason": "the public repository entry point evaluates ogbl-wikikg2, not this benchmark",
            },
            "harmonized_average": {
                "filter": "all known train+valid+test facts",
                "tie_policy": "average",
                "rank_columns": ["head", "tail"],
            },
        },
        "recomputed_harmonized_metrics": recomputed_metrics,
        "logged_harmonized_metrics": {name: logged_metrics[name] for name in metric_names},
        "metric_abs_drift": metric_drift,
        "max_rank_abs_drift": rank_drift,
        "metric_audit_passed": metric_audit_passed,
        "paper_reference_metrics": paper_reference,
        "reference_metric_gaps": reference_gaps,
        "max_allowed_reference_gap": arguments.max_reference_gap,
        "performance_plausibility_passed": performance_plausibility_passed,
        "audit_passed": metric_audit_passed and performance_plausibility_passed,
        "scope": ("Single validation-selected checkpoint; source-triple sampling uncertainty "
                  "conditional on this graph/model. This is not training-seed uncertainty, and "
                  "source-triple clustering does not resolve cross-triple graph dependence."),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "test_triple_ranks.npz",
        ranks_harmonized_average=recomputed_ranks,
        triples_hrt=arrays["test"],
        raw_triples_hrt=np.asarray(raw["test"], dtype=str),
        source_triple_ids=np.arange(len(arrays["test"]), dtype=np.int64),
        directions=np.asarray(["head", "tail"]),
    )
    if not metric_audit_passed:
        raise RuntimeError(f"Independent rank/metric audit failed; BQP inference stopped: {output}")
    if not performance_plausibility_passed:
        raise RuntimeError(
            f"Test MRR differs from the paper reference by {reference_gaps['mrr']:+.4f}, exceeding "
            f"{arguments.max_reference_gap:.4f}; diagnose reproduction before BQP: {output}")

    report, samples = analyze(recomputed_ranks, arrays["test"][:, 1],
                              arguments.repeats, arguments.seed)
    result = {
        "protocol": "harmonized_average",
        "BQP": report,
        "official_dataset_specific_protocol_available": False,
    }
    (output / "bqp_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output / "harmonized_average_resampling.npz", **samples)
    print(json.dumps({
        "output": str(output),
        "audit_passed": True,
        "recomputed_metrics": recomputed_metrics,
        "paper_reference_metrics": paper_reference,
        "reference_metric_gaps": reference_gaps,
        "BQP": report,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
