"""Export formal paired head/tail test-query ranks for the BQP study.

Validation-selected inference checkpoints are evaluated exactly once on the test
split.  Outputs retain both a query-level representation and an explicitly paired
source-triple representation so downstream statistics cannot accidentally treat
the two directions as unrelated observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.data import KGDataset, load_dataset
from bqp.metrics import evaluate, metric_dict
from bqp.models import build_model


DEFAULT_CHECKPOINT_ROOT = ROOT / "checkpoints" / "bqp_baselines"
DEFAULT_OUTPUT_ROOT = ROOT / "results_bqp" / "test_query_ranks"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_gzip_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def discover_runs(
    checkpoint_root: Path,
    datasets: set[str] | None,
    models: set[str] | None,
    seeds: set[int] | None,
) -> list[Path]:
    runs = sorted(checkpoint_root.glob("*/*/seed_*"))
    selected: list[Path] = []
    for run in runs:
        try:
            seed = int(run.name[len("seed_") :])
        except ValueError:
            continue
        if datasets is not None and run.parents[1].name not in datasets:
            continue
        if models is not None and run.parent.name.lower() not in models:
            continue
        if seeds is not None and seed not in seeds:
            continue
        selected.append(run)
    return selected


def query_arrays(dataset: KGDataset, ranks: np.ndarray) -> dict[str, np.ndarray]:
    triples = dataset.test.astype(np.int64, copy=False)
    queries = dataset.queries("test").numpy().astype(np.int64, copy=False)
    count = len(triples)
    if ranks.shape != (2 * count,):
        raise ValueError(f"Expected {2 * count} ranks, found shape {ranks.shape}")
    expected_tail = triples
    expected_head = triples[:, [2, 1, 0]].copy()
    expected_head[:, 1] += dataset.num_relations
    if not np.array_equal(queries[:count], expected_tail):
        raise ValueError("Tail-query rows are not aligned with source test triples")
    if not np.array_equal(queries[count:], expected_head):
        raise ValueError("Head-query rows are not aligned with source test triples")

    source_ids = np.tile(np.arange(count, dtype=np.int64), 2)
    directions = np.concatenate(
        [np.zeros(count, dtype=np.uint8), np.ones(count, dtype=np.uint8)]
    )
    originals = triples[source_ids]
    return {
        "source_triple_ids": source_ids,
        "directions": directions,
        "original_heads": originals[:, 0],
        "base_relations": originals[:, 1],
        "original_tails": originals[:, 2],
        "query_heads": queries[:, 0],
        "query_relations": queries[:, 1],
        "target_entities": queries[:, 2],
        "filtered_ranks": ranks.astype(np.float32, copy=False),
        "reciprocal_ranks": (1.0 / ranks).astype(np.float32, copy=False),
    }


def query_csv_rows(
    arrays: dict[str, np.ndarray], dataset: KGDataset
) -> Iterable[list[object]]:
    id_to_entity = {index: entity for entity, index in dataset.entity_to_id.items()}
    id_to_relation = {
        index: relation for relation, index in dataset.relation_to_id.items()
    }
    for index in range(len(arrays["source_triple_ids"])):
        original_head = int(arrays["original_heads"][index])
        relation = int(arrays["base_relations"][index])
        original_tail = int(arrays["original_tails"][index])
        direction = int(arrays["directions"][index])
        yield [
            int(arrays["source_triple_ids"][index]),
            "head" if direction else "tail",
            original_head,
            relation,
            original_tail,
            id_to_entity[original_head],
            id_to_relation[relation],
            id_to_entity[original_tail],
            int(arrays["query_heads"][index]),
            int(arrays["query_relations"][index]),
            int(arrays["target_entities"][index]),
            float(arrays["filtered_ranks"][index]),
            float(arrays["reciprocal_ranks"][index]),
        ]


def pair_csv_rows(
    arrays: dict[str, np.ndarray], dataset: KGDataset
) -> Iterable[list[object]]:
    count = len(dataset.test)
    id_to_entity = {index: entity for entity, index in dataset.entity_to_id.items()}
    id_to_relation = {
        index: relation for relation, index in dataset.relation_to_id.items()
    }
    for source_id, (head, relation, tail) in enumerate(dataset.test.tolist()):
        tail_rank = float(arrays["filtered_ranks"][source_id])
        head_rank = float(arrays["filtered_ranks"][count + source_id])
        yield [
            source_id,
            head,
            relation,
            tail,
            id_to_entity[head],
            id_to_relation[relation],
            id_to_entity[tail],
            tail_rank,
            head_rank,
            1.0 / tail_rank,
            1.0 / head_rank,
        ]


def export_run(
    run_dir: Path,
    output_root: Path,
    device: torch.device,
    batch_size_override: int | None,
    force: bool,
    reproduction_atol: float,
) -> dict[str, Any]:
    dataset_name = run_dir.parents[1].name
    model_name = run_dir.parent.name.lower()
    seed = int(run_dir.name[len("seed_") :])
    output_dir = output_root / dataset_name / model_name / run_dir.name
    metrics_path = output_dir / "test_metrics.json"
    if metrics_path.exists() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"skip complete {dataset_name}/{model_name}/seed_{seed}", flush=True)
            return existing

    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "model.pt"
    source_manifest_path = run_dir / "manifest.json"
    for path in (config_path, checkpoint_path, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset = load_dataset(dataset_root, config["dataset"]["name"])
    if dataset.name != dataset_name:
        raise ValueError(f"Dataset path mismatch for {run_dir}")

    checkpoint = torch_load(checkpoint_path, device)
    if checkpoint.get("format") != "bqp-inference-v1":
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    if int(checkpoint["seed"]) != seed:
        raise ValueError(f"Checkpoint seed mismatch in {checkpoint_path}")
    if checkpoint["validation_filter_protocol"] != "all_known":
        raise ValueError("Formal BQP runs require all_known validation selection")

    model = build_model(config["model"], dataset.num_entities, dataset.num_relations)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else config.get("inference", {}).get(
            "eval_batch_size", config.get("training", {}).get("eval_batch_size", 128)
        )
    )
    started = time.time()
    evaluation = evaluate(
        model,
        dataset.queries("test"),
        dataset.filter_true("test"),
        dataset.num_relations,
        batch_size,
        device,
    )
    elapsed = time.time() - started
    arrays = query_arrays(dataset, evaluation.ranks.astype(np.float64))
    query_count = len(arrays["filtered_ranks"])
    source_count = len(dataset.test)

    recorded_mrr = float(source_manifest["recorded_test_mrr_for_migration_check"])
    reproduced_mrr = float(evaluation.metrics["global"]["mrr"])
    if not np.isclose(
        reproduced_mrr, recorded_mrr, rtol=0.0, atol=reproduction_atol
    ):
        raise ValueError(
            f"Test MRR reproduction failed for {run_dir}: "
            f"{reproduced_mrr} vs recorded {recorded_mrr}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "test_query_ranks.npz"
    query_csv_path = output_dir / "test_queries.csv.gz"
    pair_csv_path = output_dir / "test_triple_pairs.csv.gz"
    atomic_npz(
        npz_path,
        **arrays,
        selected_epoch=np.asarray(int(checkpoint["selected_epoch"]), dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int32),
        num_entities=np.asarray(dataset.num_entities, dtype=np.int32),
        num_relations=np.asarray(dataset.num_relations, dtype=np.int32),
        validation_filter_protocol=np.asarray("all_known"),
        test_filter_protocol=np.asarray("train+valid+test"),
    )
    atomic_gzip_csv(
        query_csv_path,
        [
            "source_triple_id",
            "direction",
            "original_head_id",
            "base_relation_id",
            "original_tail_id",
            "original_head",
            "base_relation",
            "original_tail",
            "query_head_id",
            "query_relation_id",
            "target_entity_id",
            "filtered_rank",
            "reciprocal_rank",
        ],
        query_csv_rows(arrays, dataset),
    )
    atomic_gzip_csv(
        pair_csv_path,
        [
            "source_triple_id",
            "head_id",
            "relation_id",
            "tail_id",
            "head",
            "relation",
            "tail",
            "tail_rank",
            "head_rank",
            "tail_reciprocal_rank",
            "head_reciprocal_rank",
        ],
        pair_csv_rows(arrays, dataset),
    )

    ranks = arrays["filtered_ranks"]
    directions = arrays["directions"]
    report = {
        "status": "complete",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "checkpoint_selection": {
            "split": "validation",
            "metric": checkpoint["selection_metric"],
            "filter_protocol": checkpoint["validation_filter_protocol"],
            "selected_epoch": int(checkpoint["selected_epoch"]),
            "validation_mrr": float(checkpoint["validation_mrr"]),
        },
        "test_evaluation": {
            "evaluated_once_by_this_export": True,
            "filter_protocol": "train+valid+test",
            "tie_policy": "average rank",
            "source_triples": source_count,
            "directional_queries": query_count,
            "global": evaluation.metrics["global"],
            "tail": metric_dict(torch.from_numpy(ranks[directions == 0])),
            "head": metric_dict(torch.from_numpy(ranks[directions == 1])),
            "per_relation": evaluation.metrics["per_relation"],
            "per_relation_direction": evaluation.metrics["per_relation_direction"],
        },
        "reproduction_check": {
            "recorded_test_mrr": recorded_mrr,
            "exported_test_mrr": reproduced_mrr,
            "absolute_difference": abs(recorded_mrr - reproduced_mrr),
            "absolute_tolerance": reproduction_atol,
            "passed": True,
        },
        "files": {
            "query_npz": npz_path.name,
            "query_csv_gz": query_csv_path.name,
            "paired_triple_csv_gz": pair_csv_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "query_npz_sha256": sha256(npz_path),
            "query_csv_gz_sha256": sha256(query_csv_path),
            "paired_triple_csv_gz_sha256": sha256(pair_csv_path),
        },
        "device": str(device),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
    }
    atomic_json(metrics_path, report)
    print(
        f"exported {dataset_name}/{model_name}/seed_{seed}: "
        f"test_mrr={reproduced_mrr:.6f}, queries={query_count}, "
        f"seconds={elapsed:.1f}",
        flush=True,
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Export validation-selected checkpoints' paired BQP test ranks"
    )
    result.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--dataset", action="append", dest="datasets")
    result.add_argument("--model", action="append", dest="models")
    result.add_argument("--seed", action="append", type=int, dest="seeds")
    result.add_argument("--device", default="auto")
    result.add_argument("--batch-size", type=int)
    result.add_argument(
        "--reproduction-atol",
        type=float,
        default=1e-4,
        help="Maximum allowed MRR drift from the archived result across hardware",
    )
    result.add_argument("--force", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def collect_complete_reports(output_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/*/seed_*/test_metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            reports.append(report)
    return reports


def main() -> None:
    args = parser().parse_args()
    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    runs = discover_runs(
        checkpoint_root,
        set(args.datasets) if args.datasets else None,
        {model.lower() for model in args.models} if args.models else None,
        set(args.seeds) if args.seeds else None,
    )
    if not runs:
        raise FileNotFoundError(f"No baseline runs found under {checkpoint_root}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint_root": str(checkpoint_root),
                    "output_root": str(output_root),
                    "runs": [str(path.relative_to(checkpoint_root)) for path in runs],
                },
                indent=2,
            )
        )
        return

    device = resolve_device(args.device)
    reports = [
        export_run(
            run,
            output_root,
            device,
            args.batch_size,
            args.force,
            args.reproduction_atol,
        )
        for run in runs
    ]
    complete_reports = collect_complete_reports(output_root)
    manifest = {
        "status": "complete",
        "format": "bqp-formal-test-ranks-v1",
        "selection_rule": "validation micro MRR under all_known filtering",
        "test_filter_protocol": "train+valid+test",
        "run_count": len(complete_reports),
        "runs": [
            {
                "dataset": report["dataset"],
                "model": report["model"],
                "seed": report["seed"],
                "selected_epoch": report["checkpoint_selection"]["selected_epoch"],
                "test_mrr": report["test_evaluation"]["global"]["mrr"],
            }
            for report in complete_reports
        ],
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(
        f"processed {len(reports)} run(s); manifest contains "
        f"{len(complete_reports)} complete formal BQP exports",
        flush=True,
    )


if __name__ == "__main__":
    main()
