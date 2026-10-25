from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


def metric_dict(ranks: torch.Tensor) -> dict[str, float | int]:
    ranks = ranks.float()
    if ranks.numel() == 0:
        return {"count": 0, "mrr": float("nan"), "mr": float("nan"), "hits@1": float("nan"), "hits@3": float("nan"), "hits@10": float("nan")}
    return {
        "count": int(ranks.numel()),
        "mrr": float((1.0 / ranks).mean().item()),
        "mr": float(ranks.mean().item()),
        "hits@1": float((ranks <= 1).float().mean().item()),
        "hits@3": float((ranks <= 3).float().mean().item()),
        "hits@10": float((ranks <= 10).float().mean().item()),
    }


def filtered_ranks(
    scores: torch.Tensor,
    triples: torch.Tensor,
    all_true: dict[tuple[int, int], set[int]],
) -> torch.Tensor:
    filtered = scores.clone()
    targets = triples[:, 2]
    for row_index, (head, relation, tail) in enumerate(triples.tolist()):
        known = all_true.get((head, relation), set())
        if known:
            blocked = [entity for entity in known if entity != tail]
            if blocked:
                filtered[row_index, torch.tensor(blocked, device=filtered.device)] = -torch.inf
    target_scores = filtered.gather(1, targets.unsqueeze(1))
    greater = (filtered > target_scores).sum(dim=1).float()
    equal = (filtered == target_scores).sum(dim=1).float()
    return 1.0 + greater + 0.5 * (equal - 1.0)


def summarize_ranks(
    ranks: torch.Tensor,
    triples: torch.Tensor,
    num_base_relations: int,
) -> dict[str, object]:
    ranks = ranks.detach().cpu()
    triples = triples.detach().cpu()
    relation_ids = triples[:, 1]
    base_relations = relation_ids.remainder(num_base_relations)
    directions = (relation_ids >= num_base_relations).long()
    result: dict[str, object] = {
        "global": metric_dict(ranks),
        "per_relation": {},
        "per_relation_direction": {},
    }
    for relation in torch.unique(base_relations, sorted=True).tolist():
        relation_mask = base_relations == relation
        result["per_relation"][str(relation)] = metric_dict(ranks[relation_mask])
        for direction_id, direction_name in ((0, "tail"), (1, "head")):
            mask = relation_mask & (directions == direction_id)
            if mask.any():
                result["per_relation_direction"][f"{relation}:{direction_name}"] = metric_dict(ranks[mask])
    return result


@dataclass
class EvaluationOutput:
    metrics: dict[str, object]
    ranks: np.ndarray


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    queries: torch.Tensor,
    all_true: dict[tuple[int, int], set[int]],
    num_base_relations: int,
    batch_size: int,
    device: torch.device,
) -> EvaluationOutput:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size].to(device)
        scores = model.score_tails(batch[:, 0], batch[:, 1])
        chunks.append(filtered_ranks(scores, batch, all_true).cpu())
    ranks = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
    return EvaluationOutput(
        metrics=summarize_ranks(ranks, queries, num_base_relations),
        ranks=ranks.numpy().astype(np.float32, copy=False),
    )


@torch.no_grad()
def evaluate_variants(
    model: torch.nn.Module,
    queries: torch.Tensor,
    filters: dict[str, dict[tuple[int, int], set[int]]],
    num_base_relations: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, EvaluationOutput]:
    """Evaluate multiple filtering protocols while sharing model score computation."""
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in filters}
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size].to(device)
        scores = model.score_tails(batch[:, 0], batch[:, 1])
        for name, all_true in filters.items():
            chunks[name].append(filtered_ranks(scores, batch, all_true).cpu())
    outputs: dict[str, EvaluationOutput] = {}
    for name, rank_chunks in chunks.items():
        ranks = torch.cat(rank_chunks, dim=0) if rank_chunks else torch.empty(0)
        outputs[name] = EvaluationOutput(
            metrics=summarize_ranks(ranks, queries, num_base_relations),
            ranks=ranks.numpy().astype(np.float32, copy=False),
        )
    return outputs
