"""Train validation-selected BQP extension and local-reproduction checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bqp.data import load_dataset
from bqp.metrics import evaluate
from bqp.models import build_model


DEFAULT_OUTPUT = ROOT / "checkpoints" / "bqp_baselines"
SEEDS = (7, 11, 19, 23, 31)


PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    # These FB/WN settings define a new, fully documented local reproduction
    # recipe for the legacy core runs whose original training settings were
    # lost. They do not claim to describe or recreate the historical weights.
    ("FB15k237", "complex"): {
        "model": {"name": "complex", "rank": 200, "init_size": 0.001, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    ("FB15k237", "distmult"): {
        "model": {"name": "distmult", "rank": 400, "init_size": 0.001, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    ("WN18RR", "complex"): {
        "model": {"name": "complex", "rank": 200, "init_size": 0.001, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    ("WN18RR", "distmult"): {
        "model": {"name": "distmult", "rank": 400, "init_size": 0.001, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    # NELL995 is a new, explicitly derived split. Its core settings below are
    # frozen before any NELL test ranking is exported.
    ("NELL995", "complex"): {
        "model": {"name": "complex", "rank": 200, "init_size": 0.05, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    ("NELL995", "distmult"): {
        "model": {"name": "distmult", "rank": 400, "init_size": 0.05, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    ("NELL995", "cp"): {
        "model": {"name": "cp", "rank": 200, "init_size": 0.05, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-4, "eval_every": 5, "patience": 8},
    },
    ("NELL995", "rotate"): {
        "model": {"name": "rotate", "rank": 200, "init_size": 0.05, "initializer": "uniform", "margin": 9.0, "score_scale": 1.0},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-6, "eval_every": 5, "patience": 8},
    },
    ("FB15k237", "cp"): {
        "model": {"name": "cp", "rank": 200, "init_size": 0.05, "initializer": "normal"},
        "training": {"epochs": 80, "batch_size": 2048, "negative_samples": 128, "learning_rate": 0.002, "regularizer": 1e-4, "eval_every": 5, "patience": 5},
    },
    ("WN18RR", "cp"): {
        "model": {"name": "cp", "rank": 200, "init_size": 0.05, "initializer": "normal"},
        "training": {"epochs": 100, "batch_size": 2048, "negative_samples": 128, "learning_rate": 0.002, "regularizer": 1e-4, "eval_every": 5, "patience": 6},
    },
    ("FB15k237", "rotate"): {
        "model": {"name": "rotate", "rank": 200, "init_size": 0.05, "initializer": "uniform", "margin": 9.0, "score_scale": 1.0},
        "training": {"epochs": 60, "batch_size": 4096, "negative_samples": 32, "learning_rate": 0.004, "regularizer": 1e-6, "eval_every": 5, "patience": 5},
    },
    ("WN18RR", "rotate"): {
        "model": {"name": "rotate", "rank": 200, "init_size": 0.05, "initializer": "uniform", "margin": 9.0, "score_scale": 1.0},
        "training": {"epochs": 100, "batch_size": 1024, "negative_samples": 128, "learning_rate": 0.001, "regularizer": 1e-6, "eval_every": 5, "patience": 6},
    },
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one(
    dataset_name: str,
    model_name: str,
    seed: int,
    output_root: Path,
    device: torch.device,
    epoch_override: int | None,
    force: bool,
) -> dict[str, Any]:
    output = output_root / dataset_name / model_name / f"seed_{seed}"
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") == "complete":
            print(f"skip complete {dataset_name}/{model_name}/seed_{seed}", flush=True)
            return current

    seed_everything(seed)
    dataset = load_dataset(ROOT / "data" / dataset_name, dataset_name)
    preset = json.loads(json.dumps(PRESETS[(dataset_name, model_name)]))
    if epoch_override is not None:
        preset["training"]["epochs"] = int(epoch_override)
        preset["training"]["patience"] = max(1, int(epoch_override) // int(preset["training"]["eval_every"]))
    training = preset["training"]
    model = build_model(preset["model"], dataset.num_entities, dataset.num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    queries = dataset.train_queries
    validation = dataset.queries("valid")
    validation_filter = dataset.filter_true("valid", "all_known")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch_size = int(training["batch_size"])
    negatives = int(training["negative_samples"])
    regularizer = float(training["regularizer"])
    epochs = int(training["epochs"])
    eval_every = int(training["eval_every"])
    patience = int(training["patience"])
    trajectory: list[dict[str, float | int]] = []
    best_mrr = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(queries), generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(queries), batch_size):
            batch = queries[permutation[start : start + batch_size]].to(device)
            candidates = torch.randint(
                dataset.num_entities,
                (len(batch), negatives),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            positive = model.score_triples(batch)
            negative = model.score_candidates(batch[:, 0], batch[:, 1], candidates)
            loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
            if regularizer:
                loss = loss + regularizer * model.n3_penalty(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            seen += len(batch)

        if epoch % eval_every == 0 or epoch == epochs:
            valid = evaluate(
                model,
                validation,
                validation_filter,
                dataset.num_relations,
                128,
                device,
            )
            valid_mrr = float(valid.metrics["global"]["mrr"])
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "validation_mrr": valid_mrr,
                "elapsed_seconds": time.time() - started,
            }
            trajectory.append(row)
            print(
                f"{dataset_name}/{model_name}/seed_{seed} epoch={epoch} "
                f"loss={row['train_loss']:.5f} val_mrr={valid_mrr:.6f}",
                flush=True,
            )
            if valid_mrr > best_mrr + 1e-7:
                best_mrr = valid_mrr
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(
        model,
        dataset.queries("test"),
        dataset.filter_true("test"),
        dataset.num_relations,
        128,
        device,
    )
    test_mrr = float(test.metrics["global"]["mrr"])
    output.mkdir(parents=True, exist_ok=True)
    dataset_info = {"name": dataset_name, "root": f"data/{dataset_name}"}
    dataset_dir = ROOT / "data" / dataset_name
    dataset_info["files_sha256"] = {
        name: sha256(dataset_dir / name)
        for name in ("train.txt", "valid.txt", "test.txt")
        if (dataset_dir / name).is_file()
    }
    dataset_manifest_path = ROOT / "data" / dataset_name / "dataset_manifest.json"
    if dataset_manifest_path.is_file():
        dataset_info["split_manifest_sha256"] = sha256(dataset_manifest_path)
    config = {
        "format": "bqp-baseline-config-v2",
        "dataset": dataset_info,
        "model": preset["model"],
        "training": training,
        "provenance": {
            "training_origin": "local_retraining",
            "checkpoint_is_original_historical_weight": False,
            "trainer": "scripts/train_bqp_additional_architectures.py",
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "evaluation": {
            "checkpoint_selection_split": "validation",
            "checkpoint_selection_metric": "micro_mrr",
            "validation_filter_protocol": "all_known",
            "test_filter_protocol": "train+valid+test",
            "tie_policy": "average_rank",
        },
        "inference": {"eval_batch_size": 128},
    }
    atomic_json(output / "config.json", config)
    checkpoint = {
        "format": "bqp-inference-v1",
        "model_state_dict": best_state,
        "selected_epoch": best_epoch,
        "selection_metric": "validation_micro_mrr",
        "validation_filter_protocol": "all_known",
        "validation_mrr": best_mrr,
        "seed": seed,
        "source_checkpoint_sha256": None,
    }
    temporary_checkpoint = output / "model.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, output / "model.pt")
    atomic_json(output / "trajectory.json", trajectory)
    manifest = {
        "status": "complete",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selection_metric": "validation_micro_mrr",
        "validation_filter_protocol": "all_known",
        "validation_mrr": best_mrr,
        "recorded_test_mrr_for_migration_check": test_mrr,
        "training_seconds": time.time() - started,
        "training_queries_per_epoch": len(queries),
        "test_was_evaluated_once_after_selection": True,
        "training_origin": "local_retraining",
        "checkpoint_is_original_historical_weight": False,
        "checkpoint_sha256": sha256(output / "model.pt"),
        "trajectory_sha256": sha256(output / "trajectory.json"),
        "config_sha256": sha256(output / "config.json"),
    }
    atomic_json(manifest_path, manifest)
    print(
        f"complete {dataset_name}/{model_name}/seed_{seed}: best_epoch={best_epoch}, "
        f"valid_mrr={best_mrr:.6f}, test_mrr={test_mrr:.6f}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=("FB15k237", "WN18RR", "NELL995"))
    parser.add_argument("--model", action="append", choices=("complex", "distmult", "cp", "rotate"))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, help="debug override; do not use for formal matrix")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = args.dataset or ["FB15k237", "WN18RR"]
    models = args.model or ["cp", "rotate"]
    seeds = args.seed or list(SEEDS)
    device = torch.device(args.device)
    reports = []
    for dataset_name in datasets:
        for model_name in models:
            for seed in seeds:
                reports.append(
                    train_one(dataset_name, model_name, seed, args.output_root.resolve(), device, args.epochs, args.force)
                )
    complete = []
    for model_name in ("cp", "rotate"):
        for path in sorted(args.output_root.resolve().glob(f"*/{model_name}/seed_*/manifest.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("status") == "complete":
                complete.append(item)
    atomic_json(
        args.output_root.resolve() / "architecture_extension_manifest.json",
        {
            "format": "bqp-architecture-checkpoints-v1",
            "protocol": "docs/architecture_extension_protocol.md",
            "run_count": len(complete),
            "runs": complete,
        },
    )
    print(json.dumps({"completed": len(reports), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
