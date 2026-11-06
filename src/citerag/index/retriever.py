"""Hybrid retrieval (ARCHITECTURE section 8). LlamaIndex stays in here: callers get Chunk objects and plain floats."""

from __future__ import annotations

import logging
from typing import Protocol

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from citerag.config import Settings
from citerag.index.store import IndexStore
from citerag.rag.context import Chunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle
    ) -> list[NodeWithScore]: ...


def make_reranker(settings: Settings) -> Reranker:
    """Imported here because it loads PyTorch and only runs when RERANK_ENABLED is true."""
    from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

    return SentenceTransformerRerank(model=settings.rerank_model, top_n=settings.top_k_context)


def _to_chunk(item: NodeWithScore) -> Chunk:
    meta = item.node.metadata
    return Chunk(
        chunk_id=item.node.node_id,
        doc_id=meta["doc_id"],
        doc_name=meta["doc_name"],
        page=meta.get("page"),
        text=item.node.get_content(),
        score=float(item.score or 0.0),
        kind=meta.get("kind", "text"),
        image_path=meta.get("image_path"),
    )


class _MatchingBM25(BaseRetriever):
    """BM25 fills its list with score-0 chunks that share no term with the query, and rank fusion would
    reward those ranks. Keep only real keyword matches."""

    def __init__(self, inner: BM25Retriever) -> None:
        super().__init__()
        self._inner = inner

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [n for n in self._inner.retrieve(query_bundle) if (n.score or 0.0) > 0.0]


class HybridRetriever:
    def __init__(self, store: IndexStore, settings: Settings, reranker: Reranker | None = None) -> None:
        self._store, self._settings, self._reranker = store, settings, reranker
        self._bm25: _MatchingBM25 | None = None
        self._bm25_filtered: _MatchingBM25 | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild BM25 from the docstore. Call at start-up and after each ingest."""
        size = self._store.docstore_count()
        if size == 0:
            self._bm25 = self._bm25_filtered = None
            return
        docstore = self._store.docstore
        top_k = self._settings.top_k_retrieve
        self._bm25 = _MatchingBM25(
            BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=min(top_k, size))
        )
        # BM25 has no metadata filter: fetch twice as many, then drop other documents after fusion
        self._bm25_filtered = _MatchingBM25(
            BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=min(2 * top_k, size))
        )

    def _fused(self, query: str, doc_ids: list[str] | None) -> list[NodeWithScore]:
        top_k = self._settings.top_k_retrieve
        filters = None
        if doc_ids is not None:
            if not doc_ids:
                return []
            filters = MetadataFilters(
                filters=[MetadataFilter(key="doc_id", value=doc_ids, operator=FilterOperator.IN)]
            )
        bm25 = self._bm25 if doc_ids is None else self._bm25_filtered
        if bm25 is None:
            return []
        fusion = QueryFusionRetriever(
            [self._store.vector_retriever(top_k, filters), bm25],
            similarity_top_k=top_k if doc_ids is None else 2 * top_k,
            num_queries=1,  # more queries make LlamaIndex call an LLM
            mode="reciprocal_rerank",
            use_async=False,
        )
        nodes = fusion.retrieve(query)
        if doc_ids is not None:
            allowed = set(doc_ids)
            nodes = [n for n in nodes if n.node.metadata["doc_id"] in allowed]
        return nodes[:top_k]

    def candidates(self, query: str, doc_ids: list[str] | None = None) -> list[Chunk]:
        """The fused list, up to TOP_K_RETRIEVE, before any rerank."""
        return [_to_chunk(n) for n in self._fused(query, doc_ids)]

    def retrieve(self, query: str, doc_ids: list[str] | None = None) -> list[Chunk]:
        """What the model will see: up to TOP_K_CONTEXT chunks, best first."""
        nodes = self._fused(query, doc_ids)
        limit = self._settings.top_k_context
        if self._settings.rerank_enabled and nodes:
            try:
                reranker = self._reranker or make_reranker(self._settings)
                self._reranker = reranker
                return [_to_chunk(n) for n in reranker.postprocess_nodes(nodes, QueryBundle(query))[:limit]]
            except Exception:  # NF-3: a reranker failure must not stop the answer
                logger.warning("rerank failed; continuing with the fused order", exc_info=True)
        return [_to_chunk(n) for n in nodes[:limit]]

    def embed_query(self, query: str) -> list[float]:
        return self._store.embed_query(query)

    def top_cosine(self, query_embedding: list[float], doc_ids: list[str] | None = None) -> float:
        return self._store.top_cosine(query_embedding, doc_ids)

    def passes_gate(self, query_embedding: list[float], doc_ids: list[str] | None = None) -> bool:
        """Relevance gate (F-RET-4). MIN_TOP_SIMILARITY is a placeholder until M4 calibrates it."""
        return self.top_cosine(query_embedding, doc_ids) >= self._settings.min_top_similarity
