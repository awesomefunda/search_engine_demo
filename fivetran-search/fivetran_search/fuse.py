"""Candidate fusion.

Reciprocal Rank Fusion: each retriever contributes 1/(k + rank). It is used
instead of adding the raw scores because BM25 and cosine live on completely
different scales (and SQLite's bm25() output is not even bounded), so only the
ranks are safely comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import SETTINGS


@dataclass
class Candidate:
    page_id: int
    rrf: float = 0.0
    lexical_score: float = 0.0
    lexical_rank: int | None = None
    semantic_score: float = 0.0
    semantic_rank: int | None = None
    sources: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    lexical: list[tuple[int, float]],
    semantic: list[tuple[int, float]],
    k: int | None = None,
    limit: int | None = None,
) -> list[Candidate]:
    k = SETTINGS.rrf_k if k is None else k
    limit = SETTINGS.fusion_candidates if limit is None else limit
    by_id: dict[int, Candidate] = {}

    for rank, (pid, score) in enumerate(lexical, start=1):
        c = by_id.setdefault(pid, Candidate(page_id=pid))
        c.lexical_score = score
        c.lexical_rank = rank
        c.rrf += 1.0 / (k + rank)
        c.sources.append("keyword")

    for rank, (pid, score) in enumerate(semantic, start=1):
        c = by_id.setdefault(pid, Candidate(page_id=pid))
        c.semantic_score = score
        c.semantic_rank = rank
        c.rrf += 1.0 / (k + rank)
        c.sources.append("semantic")

    out = sorted(by_id.values(), key=lambda c: (-c.rrf, c.page_id))
    return out[:limit]


def minmax(values: list[float]) -> list[float]:
    """Scale to [0,1] within the candidate set. Identical values -> all 0.5."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
