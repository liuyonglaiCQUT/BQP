"""Train the paper-described CompoundE FB/WN variants and export paired ranks.

The public CompoundE repository only exposes an ogbl-wikikg2 entry point. This
runner is therefore an explicit Table-11 adaptation for FB15k-237 and WN18RR,
not a claim that an unpublished author training recipe was recovered. The
public loss, frequency weighting, alternating directions, initialization, and
learning-rate schedule are retained. Benchmark negative sampling excludes
known training facts, following the RotatE/PairRE recipe cited by the paper.
Model selection uses validation MRR only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapters"))
from compounde import CompoundE


SPECS = {
    "FB15k237": {
        "dim": 600,
        "lr": 5e-5,
        "negatives": 125,
        "alpha": 1.0,
        "variant": "fb_complete_srt",
        "reference_metrics": {"mrr": 0.357, "hits@1": 0.264,
                              "hits@3": 0.393, "hits@10": 0.545},
        "split_sizes": {"train": 272115, "valid": 17535, "test": 20466},
    },
    "WN18RR": {
        "dim": 500,
        "lr": 7e-5,
        "negatives": 256,
        "alpha": 0.5,
        "variant": "wn_rst_st",
        "reference_metrics": {"mrr": 0.491, "hits@1": 0.450,
                              "hits@3": 0.508, "hits@10": 0.576},
        "split_sizes": {"train": 86835, "valid": 3034, "test": 3134},
    },
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_data(dataset: str):
    folder = ROOT / "repos" / "SimKGC" / "data" / dataset
    paths = {split: folder / f"{split}.txt" for split in ("train", "valid", "test")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = {
        split: [tuple(line.split("\t")) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        for split, path in paths.items()
    }
    for split, rows in raw.items():
        malformed = next((index for index, row in enumerate(rows, 1) if len(row) != 3), None)
        if malformed is not None:
            raise ValueError(f"Malformed {dataset}/{split}.txt row {malformed}")
        expected = SPECS[dataset]["split_sizes"][split]
        if len(rows) != expected:
            raise ValueError(f"Unexpected {dataset} {split} size: {len(rows)} != {expected}")
    entities = sorted({entity for rows in raw.values() for h, _, t in rows for entity in (h, t)})
    relations = sorted({relation for rows in raw.values() for _, relation, _ in rows})
    entity_to_id = {value: index for index, value in enumerate(entities)}
    relation_to_id = {value: index for index, value in enumerate(relations)}
    arrays = {
        split: np.asarray([[entity_to_id[h], relation_to_id[r], entity_to_id[t]]
                           for h, r, t in rows], dtype=np.int64)
        for split, rows in raw.items()
    }
    hashes = {split: digest(path) for split, path in paths.items()}
    return arrays, entities, relations, raw, hashes


def filters(rows: np.ndarray):
    head, tail = defaultdict(set), defaultdict(set)
    for h, r, t in rows:
        head[int(r), int(t)].add(int(h))
        tail[int(h), int(r)].add(int(t))
    return head, tail


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        **{f"hits@{cutoff}": float(np.mean(ranks <= cutoff)) for cutoff in (1, 3, 10)},
    }


@torch.no_grad()
def evaluate(model, rows, known, batch_size, chunk, device, progress=False):
    """All-known filtered ranks with average tie handling; columns are head, tail."""
    model.eval()
    heads, tails = known
    result = []
    for direction in ("head", "tail"):
        ranks = []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch = torch.as_tensor(batch_rows, device=device)
            target = model(batch).squeeze(1)
            greater = torch.zeros(len(batch), dtype=torch.int64, device=device)
            equal = torch.zeros(len(batch), dtype=torch.int64, device=device)
            for begin in range(0, model.entity.num_embeddings, chunk):
                end = min(begin + chunk, model.entity.num_embeddings)
                candidates = torch.arange(begin, end, device=device).expand(len(batch), -1)
                scores = model(batch, candidates, direction)
                for index, (h, r, t) in enumerate(batch_rows):
                    banned = heads[int(r), int(t)] if direction == "head" else tails[int(h), int(r)]
                    gold = int(h if direction == "head" else t)
                    banned_in_chunk = [candidate - begin for candidate in banned
                                       if begin <= candidate < end and candidate != gold]
                    if banned_in_chunk:
                        scores[index, banned_in_chunk] = -torch.inf
                    if begin <= gold < end:
                        scores[index, gold - begin] = target[index]
                greater += (scores > target[:, None]).sum(1)
                equal += (scores == target[:, None]).sum(1)
            ranks.extend((1.0 + greater.double() + (equal.double() - 1.0) / 2.0).cpu().tolist())
            if progress and start % max(batch_size * 100, 1) == 0:
                print(f"EVAL {direction} {start}/{len(rows)}", flush=True)
        result.append(ranks)
    ranks = np.asarray(result, dtype=np.float64).T
    return ranks, metrics_from_ranks(ranks)


class ShuffledBatcher:
    """Stateful no-replacement batches, including the final short batch."""

    def __init__(self, size: int, batch_size: int, rng: np.random.Generator):
        self.size = size
        self.batch_size = batch_size
        self.rng = rng
        self.order = rng.permutation(size)
        self.offset = 0

    def next(self) -> np.ndarray:
        if self.offset >= self.size:
            self.order = self.rng.permutation(self.size)
            self.offset = 0
        stop = min(self.offset + self.batch_size, self.size)
        indices = self.order[self.offset:stop]
        self.offset = stop
        return indices

    def state_dict(self) -> dict:
        return {"order": self.order, "offset": self.offset,
                "size": self.size, "batch_size": self.batch_size}

    def load_state_dict(self, state: dict) -> None:
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Training batcher configuration changed during resume")
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (self.size,) or len(np.unique(order)) != self.size:
            raise ValueError("Invalid saved training permutation")
        self.order = order
        self.offset = int(state["offset"])
        if not 0 <= self.offset <= self.size:
            raise ValueError("Invalid saved training permutation offset")


def frequency_weights(train: np.ndarray) -> np.ndarray:
    """Match public code: defaultdict(4) for (h,r) and (t,-r-1)."""
    head_frequency, tail_frequency = Counter(), Counter()
    for h, r, t in train:
        head_frequency[int(h), int(r)] += 1
        tail_frequency[int(t), int(r)] += 1
    return np.asarray([
        1.0 / np.sqrt(head_frequency[int(h), int(r)] + 4
                      + tail_frequency[int(t), int(r)] + 4)
        for h, r, t in train
    ], dtype=np.float32)


def sample_negatives(rows: np.ndarray, direction: str, entity_count: int,
                     negative_count: int, training_filters,
                     rng: np.random.Generator, protocol: str) -> np.ndarray:
    negative = rng.integers(entity_count, size=(len(rows), negative_count), dtype=np.int64)
    if protocol == "uniform_unfiltered":
        return negative
    heads, tails = training_filters
    for index, (h, r, t) in enumerate(rows):
        banned = heads[int(r), int(t)] if direction == "head" else tails[int(h), int(r)]
        if len(banned) >= entity_count:
            raise ValueError("No valid negative candidates")
        banned_array = np.fromiter(banned, dtype=np.int64, count=len(banned))
        collision = np.isin(negative[index], banned_array)
        while np.any(collision):
            negative[index, collision] = rng.integers(
                entity_count, size=int(collision.sum()), dtype=np.int64)
            collision = np.isin(negative[index], banned_array)
    return negative


def checkpoint_state(model, optimizer, step, best, best_step, next_decay,
                     rng, head_batcher, tail_batcher) -> dict:
    return {
        "format": "compounde-table11-adaptation-checkpoint-v2",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best": best,
        "best_step": best_step,
        "next_decay": next_decay,
        "numpy_generator_state": rng.bit_generator.state,
        "numpy_legacy_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (torch.cuda.get_rng_state_all()
                                      if torch.cuda.is_available() else None),
        "head_batcher": head_batcher.state_dict(),
        "tail_batcher": tail_batcher.state_dict(),
    }


def restore_rng_and_batchers(state, rng, head_batcher, tail_batcher, device) -> None:
    rng.bit_generator.state = state["numpy_generator_state"]
    np.random.set_state(state["numpy_legacy_state"])
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    if str(device).startswith("cuda") and state.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_rng_state_all"]])
    head_batcher.load_state_dict(state["head_batcher"])
    tail_batcher.load_state_dict(state["tail_batcher"])


def training_config(arguments, preset, entities, relations, raw_hashes) -> dict:
    return {
        "format": "compounde-table11-adaptation-config-v2",
        "dataset": arguments.dataset,
        "seed": arguments.seed,
        "device_requested": arguments.device,
        "steps": arguments.steps,
        "valid_every": arguments.valid_every,
        "checkpoint_every": arguments.checkpoint_every,
        "microbatch": arguments.microbatch,
        "eval_batch": arguments.eval_batch,
        "entity_chunk": arguments.entity_chunk,
        **preset,
        "entities": len(entities),
        "relations": len(relations),
        "raw_split_sha256": raw_hashes,
        "training_source_sha256": {
            "adapter": digest(ROOT / "adapters" / "compounde.py"),
            "runner": digest(Path(__file__).resolve()),
        },
        "selection": "maximum validation MRR; first step wins exact ties",
        "evaluation_filter": "all known train+valid+test facts",
        "tie_policy": "average",
        "negative_sampling": arguments.negative_sampling,
        "negative_sampling_scope": (
            "filtered excludes known training facts, following the RotatE/PairRE benchmark recipe; "
            "the public OGB-only CompoundE dataloader uses uniform_unfiltered"
        ),
        "batch_sampling": "separate shuffled no-replacement head/tail streams; tail first",
        "relation_parameterization": "five d-blocks for FB/WN; FB head/tail rotation shared",
        "implementation": "paper-table-11-adaptation",
        "public_repository_scope": "official public entry point is ogbl-wikikg2 only",
    }


def check_resume_config(saved: dict, expected: dict) -> None:
    keys = (
        "format", "dataset", "seed", "steps", "valid_every", "checkpoint_every",
        "dim", "lr", "negatives", "alpha", "variant", "gamma", "batch",
        "entities", "relations", "raw_split_sha256", "training_source_sha256",
        "negative_sampling",
        "negative_sampling_scope",
        "batch_sampling", "relation_parameterization", "implementation",
    )
    for key in keys:
        if saved.get(key) != expected.get(key):
            raise ValueError(f"Resume configuration mismatch for {key}: "
                             f"{saved.get(key)!r} != {expected.get(key)!r}")


def train(arguments) -> None:
    if arguments.steps < 1 or arguments.valid_every < 1 or arguments.checkpoint_every < 1:
        raise ValueError("steps, valid-every and checkpoint-every must be positive")
    if arguments.microbatch < 1 or arguments.eval_batch < 1 or arguments.entity_chunk < 1:
        raise ValueError("batch and chunk sizes must be positive")
    if str(arguments.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    rng = np.random.default_rng(arguments.seed)
    arrays, entities, relations, raw, raw_hashes = read_data(arguments.dataset)
    preset = dict(SPECS[arguments.dataset])
    preset.pop("split_sizes")
    preset.update(gamma=6.0, batch=1024)
    if arguments.smoke:
        preset.update(dim=16, batch=8, negatives=8)
        arrays = {split: values[:32] for split, values in arrays.items()}
        raw = {split: values[:32] for split, values in raw.items()}
        arguments.steps, arguments.valid_every, arguments.checkpoint_every = 2, 1, 1

    output = (ROOT / ("smoke" if arguments.smoke else "runs") / "CompoundE" /
              arguments.dataset / f"seed_{arguments.seed}")
    if output.exists() and not arguments.resume:
        raise SystemExit(f"Refusing to overwrite {output}; use --resume or move the directory")
    output.mkdir(parents=True, exist_ok=True)

    config = training_config(arguments, preset, entities, relations, raw_hashes)
    config_path = output / "config.json"
    if arguments.resume:
        if not config_path.is_file() or not (output / "last.pt").is_file():
            raise FileNotFoundError("Resume requires config.json and last.pt in " + str(output))
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        check_resume_config(saved_config, config)
        config = saved_config
        with (output / "resume_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     "eval_batch": arguments.eval_batch,
                                     "entity_chunk": arguments.entity_chunk,
                                     "microbatch": arguments.microbatch}) + "\n")
    else:
        atomic_text(config_path, json.dumps(config, indent=2, ensure_ascii=False))
        atomic_text(output / "vocabulary.json", json.dumps(
            {"entities": entities, "relations": relations}, ensure_ascii=False))

    model = CompoundE(len(entities), len(relations), preset["dim"],
                      preset["gamma"], preset["variant"]).to(arguments.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=preset["lr"])
    known = filters(np.concatenate([arrays[split] for split in ("train", "valid", "test")]))
    training_filters = filters(arrays["train"])
    weights_by_row = frequency_weights(arrays["train"])
    head_batcher = ShuffledBatcher(len(arrays["train"]), preset["batch"], rng)
    tail_batcher = ShuffledBatcher(len(arrays["train"]), preset["batch"], rng)
    best, best_step, first, next_decay = -1.0, None, 1, arguments.steps // 2

    if arguments.resume:
        state = torch.load(output / "last.pt", map_location=arguments.device)
        if state.get("format") != "compounde-table11-adaptation-checkpoint-v2":
            raise ValueError("Legacy checkpoint lacks complete resume state; preserve it and start a new run")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        best, best_step = float(state["best"]), state["best_step"]
        first, next_decay = int(state["step"]) + 1, int(state["next_decay"])
        restore_rng_and_batchers(state, rng, head_batcher, tail_batcher, arguments.device)
        print(f"RESUME step={first} best_validation_mrr={best:.8f} best_step={best_step}", flush=True)

    started = time.time()
    for step in range(first, arguments.steps + 1):
        model.train()
        direction = "tail" if step % 2 else "head"
        indices = (tail_batcher if direction == "tail" else head_batcher).next()
        rows = arrays["train"][indices]
        negative = sample_negatives(
            rows, direction, len(entities), preset["negatives"], training_filters,
            rng, arguments.negative_sampling)
        weights = torch.as_tensor(weights_by_row[indices], device=arguments.device)
        weights = weights / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for begin in range(0, len(rows), arguments.microbatch):
            end = min(begin + arguments.microbatch, len(rows))
            positive = torch.as_tensor(rows[begin:end], device=arguments.device)
            candidates = torch.as_tensor(negative[begin:end], device=arguments.device)
            negative_scores = model(positive, candidates, direction)
            negative_loss = -(F.softmax(negative_scores * preset["alpha"], 1).detach()
                              * F.logsigmoid(-negative_scores)).sum(1)
            positive_loss = -F.logsigmoid(model(positive).squeeze(1))
            loss = ((positive_loss + negative_loss) * 0.5 * weights[begin:end]).sum()
            loss.backward()
            total_loss += float(loss.detach())
        optimizer.step()

        if step >= next_decay and step < arguments.steps:
            for group in optimizer.param_groups:
                group["lr"] /= 10.0
            next_decay *= 3
            print(f"LR_DECAY step={step} lr={optimizer.param_groups[0]['lr']}", flush=True)
        if step % 100 == 0 or arguments.smoke:
            print(json.dumps({"step": step, "loss": total_loss,
                              "lr": optimizer.param_groups[0]["lr"],
                              "seconds": time.time() - started}), flush=True)

        validation_due = step % arguments.valid_every == 0 or step == arguments.steps
        if validation_due:
            _, validation = evaluate(model, arrays["valid"], known, arguments.eval_batch,
                                     arguments.entity_chunk, arguments.device)
            improved = validation["mrr"] > best
            if improved:
                best, best_step = validation["mrr"], step
            state = checkpoint_state(model, optimizer, step, best, best_step, next_decay,
                                     rng, head_batcher, tail_batcher)
            atomic_torch_save(state, output / "last.pt")
            if improved:
                atomic_torch_save(state, output / "best.pt")
            with (output / "valid_trajectory.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, **validation,
                                         "selected": improved}) + "\n")
            print(f"VALID step={step} selected={improved} {json.dumps(validation)}", flush=True)
        elif step % arguments.checkpoint_every == 0:
            state = checkpoint_state(model, optimizer, step, best, best_step, next_decay,
                                     rng, head_batcher, tail_batcher)
            atomic_torch_save(state, output / "last.pt")
            print(f"CHECKPOINT step={step}", flush=True)

    if not (output / "best.pt").is_file():
        raise RuntimeError("Training finished without a validation-selected checkpoint")
    selected = torch.load(output / "best.pt", map_location=arguments.device)
    model.load_state_dict(selected["model"], strict=True)
    ranks, test_metrics = evaluate(model, arrays["test"], known, arguments.eval_batch,
                                   arguments.entity_chunk, arguments.device, progress=True)
    checkpoint_hash = digest(output / "best.pt")
    np.savez_compressed(
        output / "test_triple_ranks.npz",
        ranks_harmonized_average=ranks,
        triples_hrt=arrays["test"],
        raw_triples_hrt=np.asarray(raw["test"], dtype=str),
        source_triple_ids=np.arange(len(arrays["test"]), dtype=np.int64),
        directions=np.asarray(["head", "tail"]),
        filter_protocol=np.asarray("all_known_train_valid_test"),
        tie_policy=np.asarray("average"),
        selected_checkpoint_step=np.asarray(selected["step"]),
        selected_checkpoint_sha256=np.asarray(checkpoint_hash),
    )
    result = {
        "format": "compounde-table11-adaptation-test-v2",
        "dataset": arguments.dataset,
        "seed": arguments.seed,
        "selected_checkpoint": "best.pt",
        "selected_checkpoint_step": int(selected["step"]),
        "selected_checkpoint_sha256": checkpoint_hash,
        "paper_reference_metrics": preset["reference_metrics"],
        "protocol": "harmonized all-known filtered, average ties",
        **test_metrics,
    }
    atomic_text(output / "test_metrics.json", json.dumps(result, indent=2))
    print("TEST " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--valid-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--entity-chunk", type=int, default=1024)
    parser.add_argument("--negative-sampling", choices=["filtered", "uniform_unfiltered"],
                        default="filtered")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    train(parser.parse_args())
