from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 columns, found {len(fields)}")
            triples.append((fields[0], fields[1], fields[2]))
    return triples


def _encode(
    triples: Iterable[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> np.ndarray:
    return np.asarray(
        [(entity_to_id[h], relation_to_id[r], entity_to_id[t]) for h, r, t in triples],
        dtype=np.int64,
    ).reshape(-1, 3)


def reciprocal(triples: np.ndarray, num_relations: int) -> np.ndarray:
    reverse = triples[:, [2, 1, 0]].copy()
    reverse[:, 1] += num_relations
    return np.concatenate([triples, reverse], axis=0)


def build_filter(triples: Iterable[np.ndarray], num_relations: int) -> dict[tuple[int, int], set[int]]:
    result: dict[tuple[int, int], set[int]] = {}
    arrays = [reciprocal(array, num_relations) for array in triples]
    for head, relation, tail in np.concatenate(arrays, axis=0):
        result.setdefault((int(head), int(relation)), set()).add(int(tail))
    return result


@dataclass
class KGDataset:
    name: str
    root: Path
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)

    @property
    def train_queries(self) -> torch.Tensor:
        return torch.from_numpy(reciprocal(self.train, self.num_relations)).long()

    def queries(self, split: str) -> torch.Tensor:
        array = getattr(self, split)
        return torch.from_numpy(reciprocal(array, self.num_relations)).long()

    def filter_true(
        self,
        split: str,
        protocol: str = "strict",
    ) -> dict[tuple[int, int], set[int]]:
        if split == "valid":
            if protocol == "strict":
                arrays = [self.train, self.valid]
            elif protocol == "all_known":
                arrays = [self.train, self.valid, self.test]
            else:
                raise ValueError("Validation filter protocol must be strict or all_known")
        elif split == "test":
            arrays = [self.train, self.valid, self.test]
        else:
            raise ValueError("Filtered evaluation is supported for valid and test splits")
        return build_filter(arrays, self.num_relations)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "root": str(self.root),
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "split_sizes": {
                "train": int(len(self.train)),
                "valid": int(len(self.valid)),
                "test": int(len(self.test)),
            },
            "evaluation_queries": {
                "valid": int(2 * len(self.valid)),
                "test": int(2 * len(self.test)),
            },
        }

    def relation_statistics(self) -> list[dict[str, object]]:
        id_to_relation = {value: key for key, value in self.relation_to_id.items()}
        statistics: list[dict[str, object]] = []
        for relation_id in range(self.num_relations):
            rows = self.train[self.train[:, 1] == relation_id]
            by_head: dict[int, set[int]] = {}
            by_tail: dict[int, set[int]] = {}
            for head, _, tail in rows:
                by_head.setdefault(int(head), set()).add(int(tail))
                by_tail.setdefault(int(tail), set()).add(int(head))
            tails_per_head = float(np.mean([len(v) for v in by_head.values()])) if by_head else 0.0
            heads_per_tail = float(np.mean([len(v) for v in by_tail.values()])) if by_tail else 0.0
            if tails_per_head < 1.5 and heads_per_tail < 1.5:
                mapping = "1-1"
            elif tails_per_head >= 1.5 and heads_per_tail < 1.5:
                mapping = "1-N"
            elif tails_per_head < 1.5 and heads_per_tail >= 1.5:
                mapping = "N-1"
            else:
                mapping = "N-N"
            statistics.append(
                {
                    "relation_id": relation_id,
                    "relation": id_to_relation[relation_id],
                    "train_count": int(len(rows)),
                    "unique_heads": len(by_head),
                    "unique_tails": len(by_tail),
                    "mean_tails_per_head": tails_per_head,
                    "mean_heads_per_tail": heads_per_tail,
                    "mapping_type": mapping,
                }
            )
        return statistics


def load_dataset(root: str | Path, name: str | None = None) -> KGDataset:
    root = Path(root)
    raw = {split: _read_triples(root / f"{split}.txt") for split in ("train", "valid", "test")}
    all_rows = raw["train"] + raw["valid"] + raw["test"]
    entities = sorted({h for h, _, _ in all_rows} | {t for _, _, t in all_rows})
    relations = sorted({r for _, r, _ in all_rows})
    entity_to_id = {entity: index for index, entity in enumerate(entities)}
    relation_to_id = {relation: index for index, relation in enumerate(relations)}
    encoded = {
        split: _encode(rows, entity_to_id, relation_to_id)
        for split, rows in raw.items()
    }
    return KGDataset(
        name=name or root.name,
        root=root,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        train=encoded["train"],
        valid=encoded["valid"],
        test=encoded["test"],
    )
