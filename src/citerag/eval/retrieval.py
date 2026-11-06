"""Retrieval-only scoring: hit@k and MRR with zero LLM requests (PRD F-EVA-7). Pure: the caller passes the search."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from citerag.eval.metrics import full_coverage_at_k, hit_at_k, is_hit, reciprocal_rank
from citerag.rag.context import Chunk
from citerag.schemas import GoldenItem, GoldenSource

SCORED_TYPES = {"answerable", "multi_chunk", "image"}


class ItemResult(BaseModel):
    id: str
    question: str
    type: str
    scored: bool
    first_hit_rank: int | None
    all_parts_by_rank: int | None = None  # multi_chunk: rank where the last gold part appears
    top_cosine: float
    top_sources: list[str]


class PartRank(BaseModel):
    part: int
    sources: list[str]
    first_rank: int | None  # rank of the first chunk of this part; None when it is not in the list
    missed: bool  # not in the top k


class RetrievalReport(BaseModel):
    n_items: int
    n_scored: int
    ks: list[int]
    mrr_k: int
    hits: dict[int, int]
    hit_rate: dict[int, float]
    mrr: float
    n_multi_chunk: int = 0
    full_coverage: dict[int, int] = {}
    answerable_cosines: list[float]
    unanswerable_cosines: list[float]
    items: list[ItemResult]


def label(chunk: Chunk) -> str:
    return chunk.doc_name if chunk.page is None else f"{chunk.doc_name} p{chunk.page}"


def first_gold_rank(chunks: list[Chunk], gold: list[GoldenSource]) -> int | None:
    """Rank of the first chunk on a gold page, counting from 1. None when the list has no gold chunk."""
    for rank, c in enumerate(chunks, start=1):
        if is_hit(c.doc_name, c.page, gold):
            return rank
    return None


def part_ranks(chunks: list[Chunk], gold: list[GoldenSource], k: int) -> list[PartRank]:
    """Sources with the same part are alternatives, so a part ranks where its earliest source appears."""
    rows = []
    for part in sorted({g.part for g in gold}):
        sources = [g for g in gold if g.part == part]
        rank = first_gold_rank(chunks, sources)
        labels = [g.doc if g.page is None else f"{g.doc} p{g.page}" for g in sources]
        rows.append(PartRank(part=part, sources=labels, first_rank=rank, missed=rank is None or rank > k))
    return rows


def rank_line(name: str, chunks: list[Chunk], gold: list[GoldenSource], show: int = 3) -> str:
    rank = first_gold_rank(chunks, gold)
    found = f"not found in {len(chunks)}" if rank is None else f"rank {rank} of {len(chunks)}"
    return f"{name}: {found}, top: {[label(c) for c in chunks[:show]]}"


def score_retrieval(
    items: list[GoldenItem],
    rank: Callable[[str], list[Chunk]],
    top_cosine: Callable[[str], float],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    mrr_k: int = 10,
) -> RetrievalReport:
    results: list[ItemResult] = []
    hits = dict.fromkeys(ks, 0)
    covered = dict.fromkeys(ks, 0)
    n_multi_chunk = 0
    reciprocal_sum = 0.0
    for item in items:
        chunks = rank(item.question)
        relevance = [is_hit(c.doc_name, c.page, item.gold_sources) for c in chunks]
        found = [(c.doc_name, c.page) for c in chunks]
        all_parts = None
        if item.type == "multi_chunk":
            n_multi_chunk += 1
            for k in ks:
                covered[k] += int(full_coverage_at_k(found, item.gold_sources, k))
            ranks = range(1, len(found) + 1)
            all_parts = next((r for r in ranks if full_coverage_at_k(found, item.gold_sources, r)), None)
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
                all_parts_by_rank=all_parts,
                top_cosine=top_cosine(item.question),
                top_sources=[label(c) for c in chunks[:5]],
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
        n_multi_chunk=n_multi_chunk,
        full_coverage=covered if n_multi_chunk else {},
        answerable_cosines=[r.top_cosine for r in results if r.scored],
        unanswerable_cosines=[r.top_cosine for r in results if r.type == "unanswerable"],
        items=results,
    )
