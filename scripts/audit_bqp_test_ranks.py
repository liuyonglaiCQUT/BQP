"""Audit the formal BQP test-rank export matrix without running the models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.data import load_dataset


REQUIRED_ARRAYS = {
    "source_triple_ids",
    "directions",
    "original_heads",
    "base_relations",
    "original_tails",
    "query_heads",
    "query_relations",
    "target_entities",
    "filtered_ranks",
    "reciprocal_ranks",
    "selected_epoch",
    "seed",
    "num_entities",
    "num_relations",
    "validation_filter_protocol",
    "test_filter_protocol",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_value(value: np.ndarray) -> Any:
    return value.item() if value.ndim == 0 else value


def csv_data_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in handle) - 1


def assert_close(actual: float, expected: float, label: str, atol: float = 2e-7) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def audit_run(result_root: Path, run: dict[str, Any]) -> dict[str, Any]:
    dataset_name = run["dataset"]
    model_name = run["model"]
    seed = int(run["seed"])
    run_dir = result_root / dataset_name / model_name / f"seed_{seed}"
    checkpoint_dir = (
        ROOT / "checkpoints" / "bqp_baselines" / dataset_name / model_name / f"seed_{seed}"
    )
    report_path = run_dir / "test_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    dataset_path = Path(config["dataset"]["root"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    dataset = load_dataset(dataset_path, dataset_name)

    npz_path = run_dir / report["files"]["query_npz"]
    query_csv_path = run_dir / report["files"]["query_csv_gz"]
    pair_csv_path = run_dir / report["files"]["paired_triple_csv_gz"]
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise AssertionError(f"{run_dir}: missing arrays {sorted(missing)}")
        values = {name: archive[name] for name in archive.files}

    source_count = len(dataset.test)
    query_count = source_count * 2
    vector_names = [
        "source_triple_ids",
        "directions",
        "original_heads",
        "base_relations",
        "original_tails",
        "query_heads",
        "query_relations",
        "target_entities",
        "filtered_ranks",
        "reciprocal_ranks",
    ]
    for name in vector_names:
        if values[name].shape != (query_count,):
            raise AssertionError(f"{run_dir}: {name} has shape {values[name].shape}")

    ids = np.arange(source_count, dtype=values["source_triple_ids"].dtype)
    directions = np.concatenate(
        [np.zeros(source_count, dtype=np.int8), np.ones(source_count, dtype=np.int8)]
    )
    triples = dataset.test
    expected_sources = np.concatenate([ids, ids])
    if not np.array_equal(values["source_triple_ids"], expected_sources):
        raise AssertionError(f"{run_dir}: source triple pairing is broken")
    if not np.array_equal(values["directions"], directions):
        raise AssertionError(f"{run_dir}: direction order is broken")
    for name, column in (
        ("original_heads", triples[:, 0]),
        ("base_relations", triples[:, 1]),
        ("original_tails", triples[:, 2]),
    ):
        if not np.array_equal(values[name], np.concatenate([column, column])):
            raise AssertionError(f"{run_dir}: {name} does not match test.txt")

    num_relations = dataset.num_relations
    if not np.array_equal(values["query_heads"][:source_count], triples[:, 0]):
        raise AssertionError(f"{run_dir}: malformed tail queries")
    if not np.array_equal(values["query_relations"][:source_count], triples[:, 1]):
        raise AssertionError(f"{run_dir}: malformed tail-query relations")
    if not np.array_equal(values["target_entities"][:source_count], triples[:, 2]):
        raise AssertionError(f"{run_dir}: malformed tail-query targets")
    if not np.array_equal(values["query_heads"][source_count:], triples[:, 2]):
        raise AssertionError(f"{run_dir}: malformed head queries")
    if not np.array_equal(
        values["query_relations"][source_count:], triples[:, 1] + num_relations
    ):
        raise AssertionError(f"{run_dir}: malformed reciprocal relations")
    if not np.array_equal(values["target_entities"][source_count:], triples[:, 0]):
        raise AssertionError(f"{run_dir}: malformed head-query targets")

    ranks = values["filtered_ranks"].astype(np.float64)
    reciprocal = values["reciprocal_ranks"].astype(np.float64)
    if not np.all(np.isfinite(ranks)) or np.any(ranks < 1.0):
        raise AssertionError(f"{run_dir}: invalid ranks")
    if np.any(ranks > dataset.num_entities):
        raise AssertionError(f"{run_dir}: ranks exceed entity count")
    if not np.allclose(reciprocal, 1.0 / ranks, rtol=1e-6, atol=1e-8):
        raise AssertionError(f"{run_dir}: reciprocal ranks do not match ranks")

    global_metrics = report["test_evaluation"]["global"]
    assert_close(float(reciprocal.mean()), float(global_metrics["mrr"]), "MRR")
    assert_close(float(ranks.mean()), float(global_metrics["mr"]), "MR", atol=1e-3)
    for cutoff in (1, 3, 10):
        assert_close(
            float(np.mean(ranks <= cutoff)),
            float(global_metrics[f"hits@{cutoff}"]),
            f"Hits@{cutoff}",
        )
    assert_close(float(run["test_mrr"]), float(global_metrics["mrr"]), "manifest MRR")

    if data_value(values["validation_filter_protocol"]) != "all_known":
        raise AssertionError(f"{run_dir}: invalid validation filter protocol")
    if data_value(values["test_filter_protocol"]) != "train+valid+test":
        raise AssertionError(f"{run_dir}: invalid test filter protocol")
    if int(data_value(values["seed"])) != seed:
        raise AssertionError(f"{run_dir}: seed mismatch")
    if int(data_value(values["selected_epoch"])) != int(run["selected_epoch"]):
        raise AssertionError(f"{run_dir}: selected epoch mismatch")

    hashes = report["files"]
    for path, key in (
        (npz_path, "query_npz_sha256"),
        (query_csv_path, "query_csv_gz_sha256"),
        (pair_csv_path, "paired_triple_csv_gz_sha256"),
    ):
        if sha256(path) != hashes[key]:
            raise AssertionError(f"{run_dir}: SHA-256 mismatch for {path.name}")
    if csv_data_rows(query_csv_path) != query_count:
        raise AssertionError(f"{run_dir}: query CSV row count mismatch")
    if csv_data_rows(pair_csv_path) != source_count:
        raise AssertionError(f"{run_dir}: pair CSV row count mismatch")
    reproduction = report["reproduction_check"]
    if not reproduction["passed"] or reproduction["absolute_difference"] > reproduction["absolute_tolerance"]:
        raise AssertionError(f"{run_dir}: archived-MRR reproduction check failed")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "source_triples": source_count,
        "directional_queries": query_count,
        "test_mrr": float(global_metrics["mrr"]),
        "archived_mrr_drift": float(reproduction["absolute_difference"]),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results_bqp" / "test_query_ranks",
    )
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    manifest_path = result_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest["runs"]
    identities = {(x["dataset"], x["model"], int(x["seed"])) for x in runs}
    if len(identities) != len(runs):
        raise AssertionError("Duplicate run identities in manifest")
    reports = [audit_run(result_root, run) for run in runs]
    summary = {
        "status": "passed",
        "format": "bqp-formal-test-ranks-audit-v1",
        "run_count": len(reports),
        "source_triples_across_runs": sum(x["source_triples"] for x in reports),
        "directional_queries_across_runs": sum(x["directional_queries"] for x in reports),
        "maximum_archived_mrr_drift": max(x["archived_mrr_drift"] for x in reports),
        "runs": reports,
    }
    output_path = result_root / "audit.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"audit passed: {len(reports)} runs, "
        f"{summary['directional_queries_across_runs']} directional queries, "
        f"max archived-MRR drift={summary['maximum_archived_mrr_drift']:.3g}"
    )


if __name__ == "__main__":
    main()
