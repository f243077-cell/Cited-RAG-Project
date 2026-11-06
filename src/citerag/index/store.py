"""Persistent index: Chroma vectors plus the LlamaIndex docstore that BM25 needs. Only the backend opens this."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from citerag.config import Settings

COLLECTION = "docs"


def make_embedder(settings: Settings) -> BaseEmbedding:
    """Imported here because it loads PyTorch. One space turns off LlamaIndex's automatic bge query prefix,
    which an empty string does not, so question and passage embeddings are symmetric (ARCHITECTURE section 7)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        query_instruction=" ",
        text_instruction=" ",
        embed_batch_size=32,
    )


def open_store(settings: Settings, index_dir: Path) -> IndexStore:
    return IndexStore(index_dir, make_embedder(settings))


class IndexStore:
    def __init__(self, index_dir: Path, embed_model: BaseEmbedding) -> None:
        LlamaSettings.llm = None  # LlamaIndex must never reach for an LLM; the gateway is the only path
        self.embed_model = embed_model
        self.embedding_dim = len(embed_model.get_text_embedding("dimension probe"))
        self._store_dir = index_dir / "store"
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(index_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )
        vector_store = ChromaVectorStore(chroma_collection=self._collection)
        if (self._store_dir / "docstore.json").exists():
            self._context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=str(self._store_dir)
            )
            self._index = load_index_from_storage(
                self._context, embed_model=embed_model, store_nodes_override=True
            )
        else:
            self._context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex(
                [], storage_context=self._context, embed_model=embed_model, store_nodes_override=True
            )

    def add_nodes(self, nodes: list[TextNode]) -> None:
        """Embeds first and writes after, so an embedding failure changes nothing."""
        self._index.insert_nodes(nodes)

    def delete_doc(self, doc_id: str) -> None:
        self._index.delete_ref_doc(doc_id, delete_from_docstore=True)

    def persist(self) -> None:
        self._context.persist(persist_dir=str(self._store_dir))

    def close(self) -> None:
        self._client.close()

    def count(self) -> int:
        return self._collection.count()

    def has_chunks(self, doc_ids: list[str] | None = None) -> bool:
        if doc_ids is None:
            return self.count() > 0
        if not doc_ids:
            return False
        return bool(self._collection.get(where={"doc_id": {"$in": doc_ids}}, limit=1, include=[])["ids"])

    def docstore_count(self) -> int:
        return len(self._context.docstore.docs)

    def chunk_ids(self, doc_id: str) -> list[str]:
        return sorted(self._collection.get(where={"doc_id": doc_id}, include=[])["ids"])

    def chunk_metadata(self, doc_id: str) -> list[dict]:
        found = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return [dict(meta) for meta in found["metadatas"]]

    @property
    def docstore(self) -> BaseDocumentStore:
        return self._context.docstore

    def vector_retriever(self, top_k: int, filters: MetadataFilters | None = None) -> BaseRetriever:
        return self._index.as_retriever(similarity_top_k=top_k, filters=filters)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_model.get_query_embedding(query)

    def top_cosine(self, query_embedding: list[float], doc_ids: list[str] | None = None) -> float:
        """True cosine of the best hit: LlamaIndex's score is exp(-distance), not cosine (F-RET-4).
        0.0 when nothing matches, so an empty index or filter never passes the gate."""
        if doc_ids is not None and not doc_ids:
            return 0.0
        where = None if doc_ids is None else {"doc_id": {"$in": doc_ids}}
        found = self._collection.query(
            query_embeddings=[query_embedding], n_results=1, where=where, include=["distances"]
        )
        distances = found["distances"][0]
        return 1.0 - distances[0] if distances else 0.0
