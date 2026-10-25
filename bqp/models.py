from __future__ import annotations

import math

import torch
from torch import nn


def _initialize_embedding(
    embedding: nn.Embedding,
    init_size: float,
    initializer: str,
) -> None:
    """Initialize embeddings while keeping legacy checkpoints reproducible.

    ``normal`` matches the initialization used by the reference KBC
    implementation from Lacroix et al.; ``uniform`` is retained for the
    project's earlier experiments.
    """
    if initializer == "normal":
        nn.init.normal_(embedding.weight, mean=0.0, std=init_size)
    elif initializer == "uniform":
        nn.init.uniform_(embedding.weight, -init_size, init_size)
    else:
        raise ValueError("initializer must be normal or uniform")


class DistMult(nn.Module):
    """DistMult with independent reciprocal-relation embeddings."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        rank: int,
        init_size: float = 0.001,
        initializer: str = "uniform",
    ):
        super().__init__()
        self.rank = int(rank)
        self.entity = nn.Embedding(num_entities, self.rank)
        self.relation = nn.Embedding(2 * num_relations, self.rank)
        _initialize_embedding(self.entity, init_size, initializer)
        _initialize_embedding(self.relation, init_size, initializer)

    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        query = self.entity(heads) * self.relation(relations)
        return query @ self.entity.weight.t()

    def score_candidates(
        self, heads: torch.Tensor, relations: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        query = self.entity(heads) * self.relation(relations)
        return (query.unsqueeze(1) * self.entity(candidates)).sum(dim=-1)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        return (
            self.entity(triples[:, 0])
            * self.relation(triples[:, 1])
            * self.entity(triples[:, 2])
        ).sum(dim=-1)

    def n3_penalty(self, triples: torch.Tensor) -> torch.Tensor:
        heads = self.entity(triples[:, 0])
        relations = self.relation(triples[:, 1])
        tails = self.entity(triples[:, 2])
        return (heads.abs().pow(3).sum() + relations.abs().pow(3).sum() + tails.abs().pow(3).sum()) / triples.shape[0]


class ComplEx(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        rank: int,
        init_size: float = 0.001,
        initializer: str = "uniform",
    ):
        super().__init__()
        self.rank = int(rank)
        self.entity = nn.Embedding(num_entities, 2 * self.rank)
        self.relation = nn.Embedding(2 * num_relations, 2 * self.rank)
        _initialize_embedding(self.entity, init_size, initializer)
        _initialize_embedding(self.relation, init_size, initializer)

    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        head_re, head_im = self.entity(heads).chunk(2, dim=-1)
        rel_re, rel_im = self.relation(relations).chunk(2, dim=-1)
        ent_re, ent_im = self.entity.weight.chunk(2, dim=-1)
        query_re = head_re * rel_re - head_im * rel_im
        query_im = head_re * rel_im + head_im * rel_re
        return query_re @ ent_re.t() + query_im @ ent_im.t()

    def score_candidates(
        self, heads: torch.Tensor, relations: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        head_re, head_im = self.entity(heads).chunk(2, dim=-1)
        rel_re, rel_im = self.relation(relations).chunk(2, dim=-1)
        candidate_re, candidate_im = self.entity(candidates).chunk(2, dim=-1)
        query_re = head_re * rel_re - head_im * rel_im
        query_im = head_re * rel_im + head_im * rel_re
        return (
            query_re.unsqueeze(1) * candidate_re
            + query_im.unsqueeze(1) * candidate_im
        ).sum(dim=-1)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        return self.score_candidates(
            triples[:, 0], triples[:, 1], triples[:, 2].unsqueeze(1)
        ).squeeze(1)

    def n3_penalty(self, triples: torch.Tensor) -> torch.Tensor:
        heads_re, heads_im = self.entity(triples[:, 0]).chunk(2, dim=-1)
        relations_re, relations_im = self.relation(triples[:, 1]).chunk(2, dim=-1)
        tails_re, tails_im = self.entity(triples[:, 2]).chunk(2, dim=-1)
        heads = torch.sqrt(heads_re.square() + heads_im.square())
        relations = torch.sqrt(relations_re.square() + relations_im.square())
        tails = torch.sqrt(tails_re.square() + tails_im.square())
        return (
            heads.abs().pow(3).sum()
            + relations.abs().pow(3).sum()
            + tails.abs().pow(3).sum()
        ) / triples.shape[0]


class CP(nn.Module):
    """Canonical polyadic factorization with distinct subject/object entities.

    The distinction is material for BQP: unlike DistMult, CP does not impose a
    symmetric entity role.  Reciprocal relations are still trained as separate
    query types so its predictions use exactly the same evaluation protocol as
    the existing baselines.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        rank: int,
        init_size: float = 0.05,
        initializer: str = "normal",
    ):
        super().__init__()
        self.rank = int(rank)
        self.subject = nn.Embedding(num_entities, self.rank)
        self.object = nn.Embedding(num_entities, self.rank)
        self.relation = nn.Embedding(2 * num_relations, self.rank)
        for embedding in (self.subject, self.object, self.relation):
            _initialize_embedding(embedding, init_size, initializer)

    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        query = self.subject(heads) * self.relation(relations)
        return query @ self.object.weight.t()

    def score_candidates(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        query = self.subject(heads) * self.relation(relations)
        return (query.unsqueeze(1) * self.object(candidates)).sum(dim=-1)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        return (
            self.subject(triples[:, 0])
            * self.relation(triples[:, 1])
            * self.object(triples[:, 2])
        ).sum(dim=-1)

    def n3_penalty(self, triples: torch.Tensor) -> torch.Tensor:
        heads = self.subject(triples[:, 0])
        relations = self.relation(triples[:, 1])
        tails = self.object(triples[:, 2])
        return (
            heads.abs().pow(3).sum()
            + relations.abs().pow(3).sum()
            + tails.abs().pow(3).sum()
        ) / triples.shape[0]


class RotatE(nn.Module):
    """RotatE scored against every entity under the shared 1-vs-all objective.

    Reciprocal relations receive separate phases, matching the reciprocal-query
    treatment used by the other baselines in this repository.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        rank: int,
        init_size: float = 0.001,
        score_scale: float = 1.0,
        margin: float = 9.0,
        initializer: str = "uniform",
    ):
        super().__init__()
        self.rank = int(rank)
        self.score_scale = float(score_scale)
        self.margin = float(margin)
        self.entity = nn.Embedding(num_entities, 2 * self.rank)
        self.relation = nn.Embedding(2 * num_relations, self.rank)
        _initialize_embedding(self.entity, init_size, initializer)
        nn.init.uniform_(self.relation.weight, -math.pi, math.pi)

    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        head_re, head_im = self.entity(heads).chunk(2, dim=-1)
        phase = self.relation(relations)
        rel_re = torch.cos(phase)
        rel_im = torch.sin(phase)
        query_re = head_re * rel_re - head_im * rel_im
        query_im = head_re * rel_im + head_im * rel_re
        ent_re, ent_im = self.entity.weight.chunk(2, dim=-1)
        # Squared complex L2 distance, expanded algebraically to avoid a
        # batch x entities x rank intermediate.  Squaring preserves the entity
        # ordering induced by L2 distance while making full ranking practical.
        query_norm = query_re.square().sum(dim=-1, keepdim=True) + query_im.square().sum(dim=-1, keepdim=True)
        entity_norm = ent_re.square().sum(dim=-1) + ent_im.square().sum(dim=-1)
        dot = query_re @ ent_re.t() + query_im @ ent_im.t()
        distance_squared = (query_norm + entity_norm.unsqueeze(0) - 2.0 * dot).clamp_min(0.0)
        return self.margin - self.score_scale * distance_squared

    def score_candidates(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        head_re, head_im = self.entity(heads).chunk(2, dim=-1)
        phase = self.relation(relations)
        rel_re, rel_im = torch.cos(phase), torch.sin(phase)
        query_re = head_re * rel_re - head_im * rel_im
        query_im = head_re * rel_im + head_im * rel_re
        candidate_re, candidate_im = self.entity(candidates).chunk(2, dim=-1)
        distance_squared = (
            (query_re.unsqueeze(1) - candidate_re).square()
            + (query_im.unsqueeze(1) - candidate_im).square()
        ).sum(dim=-1)
        return self.margin - self.score_scale * distance_squared

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        return self.score_candidates(
            triples[:, 0], triples[:, 1], triples[:, 2].unsqueeze(1)
        ).squeeze(1)

    def n3_penalty(self, triples: torch.Tensor) -> torch.Tensor:
        heads = self.entity(triples[:, 0])
        relations = self.relation(triples[:, 1])
        tails = self.entity(triples[:, 2])
        return (heads.abs().pow(3).sum() + relations.abs().pow(3).sum() + tails.abs().pow(3).sum()) / triples.shape[0]


def build_model(config: dict, num_entities: int, num_relations: int) -> nn.Module:
    name = config["name"].lower()
    common = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "rank": int(config["rank"]),
        "init_size": float(config.get("init_size", 0.001)),
        "initializer": str(config.get("initializer", "uniform")).lower(),
    }
    if name == "complex":
        return ComplEx(**common)
    if name == "distmult":
        return DistMult(**common)
    if name == "cp":
        return CP(**common)
    if name == "rotate":
        return RotatE(
            **common,
            score_scale=float(config.get("score_scale", 1.0)),
            margin=float(config.get("margin", 9.0)),
        )
    raise ValueError(f"Unsupported model: {name}")
