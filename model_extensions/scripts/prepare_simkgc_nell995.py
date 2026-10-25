#!/usr/bin/env python3
"""Prepare the audited BQP NELL-995 split for SimKGC.

This adapter deliberately does not use ``data/NELL-995/*.json`` because those
legacy files contain the full graph in train and overlapping validation/test
facts.  The source bundled under ``model_extensions/data/NELL995_source`` is an
exact copy of the split used by the existing BQP baselines.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "NELL995_source"
DESTINATION = ROOT / "repos" / "SimKGC" / "data" / "NELL995"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entity_text(entity_id: str) -> tuple[str, str]:
    if not entity_id.startswith("concept_"):
        return entity_id, "NELL entity."
    type_and_name = entity_id[len("concept_") :]
    entity_type, separator, raw_name = type_and_name.partition("_")
    name_source = raw_name if separator else type_and_name
    name = " ".join(name_source.replace("_", " ").split())
    return name, f"NELL entity. Type: {entity_type}."


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed triple at {path}:{line_number}")
            triples.append((fields[0], fields[1], fields[2]))
    if len(triples) != len(set(triples)):
        raise ValueError(f"Duplicate triples in {path}")
    return triples


def example(triple: tuple[str, str, str]) -> dict[str, str]:
    head_id, relation, tail_id = triple
    head, _ = entity_text(head_id)
    tail, _ = entity_text(tail_id)
    return {
        "head_id": head_id,
        "head": head,
        "relation": relation,
        "tail_id": tail_id,
        "tail": tail,
    }


def main() -> None:
    manifest_path = SOURCE / "dataset_manifest.json"
    audit_path = SOURCE / "split_audit.json"
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            "Missing bundled NELL995 source. Copy the audited data/NELL995 files "
            "into model_extensions/data/NELL995_source."
        )
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_audit: dict[str, Any] = json.loads(audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status") != "passed":
        raise ValueError("The bundled NELL995 split audit did not pass")
    if manifest.get("format") != "bqp-nell995-derived-split-v2":
        raise ValueError("Unexpected NELL995 split format")

    expected_sizes = {"train": 149678, "valid": 543, "test": 3992}
    triples_by_split: dict[str, list[tuple[str, str, str]]] = {}
    for split, expected_size in expected_sizes.items():
        source_path = SOURCE / f"{split}.txt"
        expected_hash = manifest["output_files_sha256"][f"{split}.txt"]
        actual_hash = sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Source hash mismatch for {source_path}: {actual_hash} != {expected_hash}"
            )
        triples_by_split[split] = read_triples(source_path)
        if len(triples_by_split[split]) != expected_size:
            raise ValueError(
                f"Unexpected {split} size: {len(triples_by_split[split])} != {expected_size}"
            )

    sets = {name: set(rows) for name, rows in triples_by_split.items()}
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]:
        raise ValueError("NELL995 source splits overlap")

    entities = sorted(
        {entity for rows in triples_by_split.values() for h, _, t in rows for entity in (h, t)}
    )
    training_entities = {entity for h, _, t in triples_by_split["train"] for entity in (h, t)}
    relations = sorted({relation for rows in triples_by_split.values() for _, relation, _ in rows})
    if len(entities) != 75492 or set(entities) != training_entities or len(relations) != 200:
        raise ValueError(
            "Unexpected NELL995 dimensions or non-transductive split: "
            f"entities={len(entities)}, training_entities={len(training_entities)}, "
            f"relations={len(relations)}"
        )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for split, rows in triples_by_split.items():
        shutil.copyfile(SOURCE / f"{split}.txt", DESTINATION / f"{split}.txt")
        with (DESTINATION / f"{split}.txt.json").open("w", encoding="utf-8") as stream:
            json.dump([example(row) for row in rows], stream, ensure_ascii=False)

    entity_rows = []
    for entity_id in entities:
        name, description = entity_text(entity_id)
        entity_rows.append(
            {"entity_id": entity_id, "entity": name, "entity_desc": description}
        )
    with (DESTINATION / "entities.json").open("w", encoding="utf-8") as stream:
        json.dump(entity_rows, stream, ensure_ascii=False)
    with (DESTINATION / "relations.json").open("w", encoding="utf-8") as stream:
        json.dump({relation: relation for relation in relations}, stream, ensure_ascii=False, indent=2)

    adapter_manifest = {
        "format": "simkgc-nell995-bqp-adapter-v1",
        "source_manifest_format": manifest["format"],
        "source_derived_split_seed": manifest["derived_split_seed"],
        "source_split_sizes": manifest["split_sizes"],
        "source_output_files_sha256": manifest["output_files_sha256"],
        "source_split_audit_status": source_audit["status"],
        "num_entities": len(entities),
        "num_relations": len(relations),
        "entity_text_rule": (
            "For concept_<type>_<name>, collapse underscores/whitespace in <name> "
            "and use 'NELL entity. Type: <type>.'; literals retain their ID as name."
        ),
        "relation_text_rule": "Use the audited base-relation identifier unchanged.",
        "output_files_sha256": {
            path.name: sha256(path)
            for path in sorted(DESTINATION.iterdir())
            if path.is_file() and path.name != "adapter_manifest.json"
        },
    }
    (DESTINATION / "adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "passed",
        "output": str(DESTINATION),
        "split_sizes": expected_sizes,
        "entities": len(entities),
        "relations": len(relations),
        "source_split_seed": manifest["derived_split_seed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
