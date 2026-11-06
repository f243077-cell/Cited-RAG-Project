"""Retrieval-only scoring: hit@k and MRR with zero LLM requests (PRD F-EVA-7). Pure: the caller passes the search."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from citerag.eval.metrics import hit_at_k, is_hit, reciprocal_rank
from citerag.rag.context import Chunk
from citerag.schemas import GoldenItem

SCORED_TYPES = {"answerable", "multi_chunk", "image"}


class ItemResult(BaseModel):
    id: str
    question: str
    type: str
    scored: bool
    first_hit_rank: int | None
    top_cosine: float
    top_sources: list[str]


class RetrievalReport(BaseModel):
    n_items: int
    n_scored: int
    ks: list[int]
    mrr_k: int
    hits: dict[int, int]
    hit_rate: dict[int, float]
    mrr: float
    answerable_cosines: list[float]
    unanswerable_cosines: list[float]
    items: list[ItemResult]


def _label(chunk: Chunk) -> str:
    return chunk.doc_name if chunk.page is None else f"{chunk.doc_name} p{chunk.page}"


def score_retrieval(
    items: list[GoldenItem],
    rank: Callable[[str], list[Chunk]],
    top_cosine: Callable[[str], float],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    mrr_k: int = 10,
) -> RetrievalReport:
    results: list[ItemResult] = []
    hits = dict.fromkeys(ks, 0)
    reciprocal_sum = 0.0
    for item in items:
        chunks = rank(item.question)
        relevance = [is_hit(c.doc_name, c.page, item.gold_sources) for c in chunks]
        scored = item.type in SCORED_TYPES
        if scored:
            for k in ks:
                hits[k] += int(hit_at_k(relevance, k))
            reciprocal_sum += reciprocal_rank(relevance, mrr_k)
        results.append(
            ItemResult(
                id=item.id,
                question=item.question,
                type=item.type,
                scored=scored,
                first_hit_rank=relevance.index(True) + 1 if True in relevance else None,
                top_cosine=top_cosine(item.question),
                top_sources=[_label(c) for c in chunks[:5]],
            )
        )
    n_scored = sum(r.scored for r in results)
    return RetrievalReport(
        n_items=len(items),
        n_scored=n_scored,
        ks=list(ks),
        mrr_k=mrr_k,
        hits=hits,
        hit_rate={k: hits[k] / n_scored if n_scored else 0.0 for k in ks},
        mrr=reciprocal_sum / n_scored if n_scored else 0.0,
        answerable_cosines=[r.top_cosine for r in results if r.scored],
        unanswerable_cosines=[r.top_cosine for r in results if r.type == "unanswerable"],
        items=results,
    )
