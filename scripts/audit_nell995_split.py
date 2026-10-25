"""Independent integrity audit for the derived NELL-995 BQP split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "NELL995"
SOURCE = ROOT / "data" / "NELL-995"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [tuple(line.rstrip("\r\n").split("\t")) for line in handle if line.strip()]
    if any(len(row) != 3 for row in rows):
        raise AssertionError(f"Malformed triple row in {path}")
    return rows


def read_raw_facts(path: Path) -> set[tuple[str, str, str]]:
    facts: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3 or ":" not in fields[1]:
                raise AssertionError(f"Malformed source fact at {path}:{line_number}")
            head, relation, tail = fields
            facts.add((head.replace(":", "_"), relation.split(":", 1)[1], tail.replace(":", "_")))
    return facts


def main() -> None:
    manifest = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = {name: read_triples(DATA / f"{name}.txt") for name in ("train", "valid", "test")}
    sets = {name: set(rows) for name, rows in splits.items()}
    for name, rows in splits.items():
        if len(rows) != len(sets[name]):
            raise AssertionError(f"Duplicate triples in {name}")
        if len(rows) != manifest["split_sizes"][name]:
            raise AssertionError(f"Split count mismatch in {name}")
        expected_hash = manifest["output_files_sha256"][f"{name}.txt"]
        if sha256(DATA / f"{name}.txt") != expected_hash:
            raise AssertionError(f"Hash mismatch in {name}.txt")
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]:
        raise AssertionError("Derived split files overlap")

    all_rows = sum(splits.values(), [])
    all_facts = set(all_rows)
    raw_facts = read_raw_facts(SOURCE / "raw.kb")
    if all_facts != raw_facts:
        raise AssertionError(
            f"Derived splits do not reconstruct raw.kb: missing={len(raw_facts - all_facts)}, "
            f"unexpected={len(all_facts - raw_facts)}"
        )
    source_hashes = manifest["source_files_sha256"]
    for name, expected_hash in source_hashes.items():
        if sha256(SOURCE / name) != expected_hash:
            raise AssertionError(f"Source hash mismatch for {name}")
    entities = {entity for head, _, tail in all_rows for entity in (head, tail)}
    training_entities = {entity for head, _, tail in splits["train"] for entity in (head, tail)}
    relations_by_split = {
        name: {relation for _, relation, _ in rows} for name, rows in splits.items()
    }
    if len(entities) != manifest["num_entities"]:
        raise AssertionError("Entity count mismatch")
    if training_entities != entities:
        raise AssertionError(
            f"Transductive entity coverage failed: {len(entities - training_entities)} entities absent from train"
        )
    if len(relations_by_split["train"]) != 200 or len(relations_by_split["test"]) != 200:
        raise AssertionError("Expected all 200 relations in train and test")
    if len(relations_by_split["valid"]) < 190:
        raise AssertionError("Validation relation coverage is unexpectedly low")
    if len(relations_by_split["test"]) != manifest["num_test_relations"]:
        raise AssertionError("Test relation coverage differs from manifest")
    if len(relations_by_split["valid"]) != manifest["num_valid_relations"]:
        raise AssertionError("Validation relation coverage differs from manifest")
    summary: dict[str, Any] = {
        "status": "passed",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "num_entities": len(entities),
        "num_training_entities": len(training_entities),
        "num_entities_absent_from_train": len(entities - training_entities),
        "num_relations": len(relations_by_split["train"]),
        "relations_per_split": {name: len(values) for name, values in relations_by_split.items()},
        "pairwise_intersections": {"train_valid": 0, "train_test": 0, "valid_test": 0},
        "output_file_hashes_verified": True,
        "source_file_hashes_verified": True,
        "raw_graph_reconstructed_exactly": True,
    }
    (DATA / "split_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
