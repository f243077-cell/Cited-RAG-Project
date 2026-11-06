"""Persistent index: Chroma vectors plus the LlamaIndex docstore that BM25 needs. Only the backend opens this."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
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

    def docstore_count(self) -> int:
        return len(self._context.docstore.docs)

    def chunk_ids(self, doc_id: str) -> list[str]:
        return sorted(self._collection.get(where={"doc_id": doc_id}, include=[])["ids"])

    def chunk_metadata(self, doc_id: str) -> list[dict]:
        found = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return [dict(meta) for meta in found["metadatas"]]

    def top_cosine(self, query_embedding: list[float]) -> float:
        """True cosine of the best hit: LlamaIndex's score is exp(-distance), not cosine (F-RET-4)."""
        found = self._collection.query(query_embeddings=[query_embedding], n_results=1, include=["distances"])
        return 1.0 - found["distances"][0][0]
