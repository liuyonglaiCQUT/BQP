"""Create a leakage-free, documented NELL-995 random-stratified split.

The bundled ``train.json`` is the complete graph with reciprocal edges, and
the bundled ``valid.json`` is entirely contained in ``test.json``.  Neither is
therefore treated as a train/validation/test partition.  Instead, the script
reconstructs the 154,213-fact graph from ``raw.kb`` and creates a fresh,
relation-stratified split with the widely reported NELL-995 split sizes.  The
source directory is never modified, and this derived split is explicitly not
claimed to reproduce the canonical benchmark's particular fact assignment.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "NELL-995"
OUTPUT = ROOT / "data" / "NELL995"
SPLIT_SEED = 20261018


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_graph(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 tab-separated fields")
            head, relation, tail = fields
            if ":" not in relation:
                raise ValueError(f"{path}:{line_number}: unexpected relation id {relation!r}")
            triples.append((head.replace(":", "_"), relation.split(":", 1)[1], tail.replace(":", "_")))
    return triples


def read_source_json(path: Path) -> list[tuple[str, str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        (str(row["head_id"]), "".join(str(row["relation"]).split()), str(row["tail_id"]))
        for row in rows
    ]


def read_source_train_as_base_facts(path: Path) -> set[tuple[str, str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    normalized: set[tuple[str, str, str]] = set()
    for row in rows:
        head = str(row["head_id"])
        relation = str(row["relation"])
        tail = str(row["tail_id"])
        if relation.endswith(" inv"):
            head, tail = tail, head
            relation = relation[:-4]
        normalized.add((head, "".join(relation.split()), tail))
    return normalized


def write_triples(path: Path, triples: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "".join(f"{head}\t{relation}\t{tail}\n" for head, relation, tail in sorted(triples)),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--train-size", type=int, default=149678)
    parser.add_argument("--valid-size", type=int, default=543)
    parser.add_argument("--test-size", type=int, default=3992)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = SOURCE
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    raw_rows = read_raw_graph(source / "raw.kb")
    raw = set(raw_rows)
    if len(raw) != len(raw_rows):
        raise ValueError("raw.kb contains duplicate triples")

    if args.train_size + args.valid_size + args.test_size != len(raw):
        raise ValueError(
            "Requested split sizes must sum to the number of raw.kb facts "
            f"({args.train_size}+{args.valid_size}+{args.test_size} != {len(raw)})"
        )

    raw_sorted = sorted(raw)
    relation_groups: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for triple in raw_sorted:
        relation_groups[triple[1]].append(triple)
    relations_sorted = sorted(relation_groups)

    def proportional_capped_counts(
        total: int, weights: dict[str, int], minimum: int, capacities: dict[str, int]
    ) -> dict[str, int]:
        if total < minimum * len(weights) or any(capacities[key] < minimum for key in weights):
            raise ValueError("Split size cannot provide the requested minimum per relation")
        counts = {relation: minimum for relation in weights}
        remaining = total - sum(counts.values())
        while remaining:
            active = [relation for relation in weights if counts[relation] < capacities[relation]]
            if not active:
                raise ValueError("Available transductive split capacity is smaller than requested split size")
            weight_sum = sum(weights[relation] for relation in active)
            quotas = {relation: remaining * weights[relation] / weight_sum for relation in active}
            saturated = [
                relation
                for relation in active
                if quotas[relation] >= capacities[relation] - counts[relation]
            ]
            if saturated:
                for relation in saturated:
                    room = capacities[relation] - counts[relation]
                    counts[relation] += room
                    remaining -= room
                continue
            floors = {relation: int(np.floor(quotas[relation])) for relation in active}
            for relation, value in floors.items():
                counts[relation] += value
            remaining -= sum(floors.values())
            order = sorted(active, key=lambda relation: (-(quotas[relation] % 1.0), relation))
            for relation in order:
                if remaining == 0:
                    break
                if counts[relation] < capacities[relation]:
                    counts[relation] += 1
                    remaining -= 1
        return counts

    # Keep a reproducible random edge cover in train: for every entity, choose
    # one incident raw fact and protect the union of those facts. Holdout facts
    # are sampled only from the remaining pool, so every split is transductive
    # by construction without repeatedly repairing a random split.
    rng = np.random.default_rng(args.seed)
    incident_indices: dict[str, list[int]] = collections.defaultdict(list)
    for index, (head, _, tail) in enumerate(raw_sorted):
        incident_indices[head].append(index)
        if tail != head:
            incident_indices[tail].append(index)
    protected_indices = {
        indices[int(rng.integers(len(indices)))] for indices in incident_indices.values()
    }
    candidate_groups: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for index, triple in enumerate(raw_sorted):
        if index not in protected_indices:
            candidate_groups[triple[1]].append(triple)
    candidate_counts = {relation: len(candidate_groups[relation]) for relation in relations_sorted}
    if any(candidate_counts[relation] == 0 for relation in relations_sorted):
        raise RuntimeError("At least one relation has no held-out candidate under the entity-coverage constraint")

    test_counts = proportional_capped_counts(
        args.test_size,
        {relation: len(relation_groups[relation]) for relation in relations_sorted},
        minimum=1,
        capacities=candidate_counts,
    )
    valid_relations = [relation for relation in relations_sorted if candidate_counts[relation] - test_counts[relation] > 0]
    valid_weights = {
        relation: len(relation_groups[relation]) - test_counts[relation] for relation in valid_relations
    }
    valid_capacity = {
        relation: candidate_counts[relation] - test_counts[relation] for relation in valid_relations
    }
    valid_counts = proportional_capped_counts(
        args.valid_size,
        valid_weights,
        minimum=1,
        capacities=valid_capacity,
    )

    train: list[tuple[str, str, str]] = []
    valid: list[tuple[str, str, str]] = []
    test: list[tuple[str, str, str]] = []
    heldout: set[tuple[str, str, str]] = set()
    for relation in relations_sorted:
        candidates = candidate_groups[relation]
        order = rng.permutation(len(candidates))
        shuffled = [candidates[int(index)] for index in order]
        n_test = test_counts[relation]
        n_valid = valid_counts.get(relation, 0)
        test.extend(shuffled[:n_test])
        valid.extend(shuffled[n_test : n_test + n_valid])
        heldout.update(shuffled[: n_test + n_valid])
    train.extend(triple for triple in raw_sorted if triple not in heldout)
    train_set, valid_set, test_set = set(train), set(valid), set(test)
    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        raise AssertionError("Derived split is not pairwise disjoint")

    entities = {entity for h, _, t in raw for entity in (h, t)}
    training_entities = {entity for h, _, t in train for entity in (h, t)}
    if training_entities != entities:
        raise AssertionError(
            f"Transductive entity coverage failed: {len(entities - training_entities)} entities absent from train"
        )
    relations = {relation for _, relation, _ in raw}
    if len(entities) != 75492 or len(relations) != 200:
        raise ValueError(f"Unexpected raw NELL graph dimensions: {len(entities)} entities, {len(relations)} relations")
    test_relations = {relation for _, relation, _ in test_set}
    valid_relations = {relation for _, relation, _ in valid_set}
    if len(test_relations) != 200:
        raise AssertionError("Expected at least one test fact for each of 200 relations")
    if len(valid_relations) < 190:
        raise AssertionError(f"Validation relation coverage is unexpectedly low: {len(valid_relations)}")

    source_valid = read_source_json(source / "valid.json")
    source_test = read_source_json(source / "test.json")
    source_valid_subset_test = set(source_valid) <= set(source_test)
    if not source_valid_subset_test:
        raise ValueError("The source valid/test overlap changed; re-audit this input before using the script")
    source_train_matches_raw = read_source_train_as_base_facts(source / "train.json") == raw
    if not source_train_matches_raw:
        raise ValueError("train.json no longer normalizes to the full raw.kb graph; re-audit this input")

    output.mkdir(parents=True, exist_ok=True)
    for split_name, triples in (("train", train), ("valid", valid), ("test", test)):
        write_triples(output / f"{split_name}.txt", triples)

    metadata: dict[str, Any] = {
        "format": "bqp-nell995-derived-split-v2",
        "dataset_name": "NELL995",
        "source": "local data/NELL-995/raw.kb",
        "source_files_sha256": {
            name: sha256(source / name)
            for name in ("raw.kb", "valid.json", "test.json", "train.json")
        },
        "derived_split_seed": int(args.seed),
        "derived_split_rule": (
            "Fresh split from raw.kb. For each entity, protect one randomly selected incident fact in train; "
            "the union of protected facts is a random edge cover. Allocate test and validation sizes over the "
            "remaining candidates proportionally to relation frequency, capped by each relation's candidate count. "
            "The remaining facts form train, so every entity is represented in train by construction."
        ),
        "requested_split_sizes": {"train": args.train_size, "valid": args.valid_size, "test": args.test_size},
        "source_split_audit": {
            "raw_kb_is_complete_graph": True,
            "train_json_normalizes_to_complete_raw_graph": source_train_matches_raw,
            "source_valid_rows": len(source_valid),
            "source_test_rows": len(source_test),
            "source_valid_is_subset_of_source_test": source_valid_subset_test,
            "source_valid_and_test_used_as_facts": False,
        },
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "num_entities": len(entities),
        "num_training_entities": len(training_entities),
        "num_entities_absent_from_train": len(entities - training_entities),
        "num_protected_entity_cover_facts": len(protected_indices),
        "num_base_relations": len(relations),
        "num_test_relations": len({relation for _, relation, _ in test_set}),
        "num_valid_relations": len({relation for _, relation, _ in valid_set}),
        "test_relation_names": sorted({relation for _, relation, _ in test_set}),
        "num_distinct_entities_in_test": len({e for h, _, t in test_set for e in (h, t)}),
        "test_relation_counts": dict(sorted(collections.Counter(r for _, r, _ in test).items())),
        "pairwise_split_intersections": {"train_valid": 0, "train_test": 0, "valid_test": 0},
        "target_population": (
            "The empirical distribution of held-out positive triples in this explicitly seeded, entity-covered, "
            "relation-stratified NELL-995 split, conditional on the fixed raw graph, split, and trained checkpoint. "
            "The test split represents all 200 relations; validation coverage follows the available candidate facts. "
            "It is not the original canonical fact assignment and does not represent arbitrary NELL facts or knowledge graphs in general."
        ),
        "inference_scope": (
            "Bootstrap intervals quantify hypothetical resampling of source triples from this derived benchmark test "
            "distribution; they do not quantify uncertainty over graphs, relation populations, alternative split seeds, "
            "or training seeds."
        ),
        "output_files_sha256": {
            f"{name}.txt": sha256(output / f"{name}.txt")
            for name in ("train", "valid", "test")
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "dataset_name": metadata["dataset_name"],
        "split_sizes": metadata["split_sizes"],
        "num_entities": metadata["num_entities"],
        "num_base_relations": metadata["num_base_relations"],
        "num_test_relations": metadata["num_test_relations"],
        "derived_split_seed": metadata["derived_split_seed"],
        "pairwise_split_intersections": metadata["pairwise_split_intersections"],
        "target_population": metadata["target_population"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
