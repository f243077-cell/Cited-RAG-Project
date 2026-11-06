"""Step 15 (LOCAL): cosine collection, persisted docstore, reload, BM25, hybrid fusion, doc_ids filter, true cosine gate."""

import math
import tempfile
from pathlib import Path

from common import result

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", query_instruction=" ", text_instruction=" "
)  # " " turns the automatic bge query prefix off (spike s14b)
TEXTS = [
    ("d1", "Items can be returned within thirty days of delivery."),
    ("d1", "Shipping takes three to five working days."),
    ("d2", "The warranty covers manufacturing defects for two years."),
    ("d3", "Zebrafish are a common model organism in genetics."),
]


def make_node(i: int, doc_id: str, text: str) -> TextNode:
    """SOURCE relation: LlamaIndex overwrites metadata doc_id with ref_doc_id. Excluded keys: metadata stays out of the embedding."""
    node = TextNode(
        id_=f"{doc_id}-{i:04d}", text=text, metadata={"doc_id": doc_id},
        excluded_embed_metadata_keys=["doc_id"], excluded_llm_metadata_keys=["doc_id"],
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


nodes = [make_node(i, d, t) for i, (d, t) in enumerate(TEXTS)]

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    chroma_dir, store_dir = Path(tmp, "chroma"), Path(tmp, "store")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_or_create_collection("docs", configuration={"hnsw": {"space": "cosine"}})
    print("space:", col.configuration.get("hnsw"))
    sc = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=col))
    VectorStoreIndex(nodes, storage_context=sc, store_nodes_override=True)
    sc.persist(persist_dir=str(store_dir))
    print("docstore nodes after build:", len(sc.docstore.docs))

    # reload as a new process would
    client2 = chromadb.PersistentClient(path=str(chroma_dir))
    col2 = client2.get_or_create_collection("docs")
    vs2 = ChromaVectorStore(chroma_collection=col2)
    sc2 = StorageContext.from_defaults(vector_store=vs2, persist_dir=str(store_dir))
    index = load_index_from_storage(sc2)
    ok_reload = len(sc2.docstore.docs) == len(TEXTS)

    bm25 = BM25Retriever.from_defaults(docstore=sc2.docstore, similarity_top_k=2)
    bm25_top = bm25.retrieve("warranty defects")[0].node.node_id
    print("bm25 after reload:", bm25_top)

    vector = index.as_retriever(similarity_top_k=2)
    fusion = QueryFusionRetriever(
        [vector, bm25], similarity_top_k=2, num_queries=1, mode="reciprocal_rerank", use_async=False
    )
    hybrid = fusion.retrieve("how long do I have to send an item back")
    print("hybrid:", [(n.node.node_id, round(n.score, 4)) for n in hybrid])

    in_filter = MetadataFilters(filters=[MetadataFilter(key="doc_id", value=["d2", "d3"], operator=FilterOperator.IN)])
    try:
        got_in = {n.node.metadata["doc_id"] for n in index.as_retriever(similarity_top_k=4, filters=in_filter).retrieve("refund")}
        print("FilterOperator.IN:", got_in)
    except Exception as exc:  # noqa: BLE001
        got_in = None
        print("FilterOperator.IN failed:", type(exc).__name__, str(exc)[:200])
    or_filter = MetadataFilters(
        filters=[MetadataFilter(key="doc_id", value="d2"), MetadataFilter(key="doc_id", value="d3")],
        condition=FilterCondition.OR,
    )
    got_or = {n.node.metadata["doc_id"] for n in index.as_retriever(similarity_top_k=4, filters=or_filter).retrieve("refund")}
    print("OR of equality filters:", got_or)

    # gate: direct Chroma query, 1 - distance, against the LlamaIndex score, for an indexed sentence and a weak query
    embed = Settings.embed_model
    for query in (TEXTS[0][1], "How do I bake sourdough bread?"):
        q_emb = embed.get_query_embedding(query)
        direct = col2.query(query_embeddings=[q_emb], n_results=1)
        dist = direct["distances"][0][0]
        li_score = index.as_retriever(similarity_top_k=1).retrieve(query)[0].score
        print(f"{query[:30]!r}: cosine={1 - dist:.4f} li_score={li_score:.4f} exp(-d)={math.exp(-dist):.4f}")
    q_emb = embed.get_query_embedding(TEXTS[0][1])
    self_cos = 1 - col2.query(query_embeddings=[q_emb], n_results=1)["distances"][0][0]
    del client, client2, col, col2, vs2, sc, sc2, index, vector, bm25, fusion

result(ok_reload and bm25_top == "d2-0002" and got_in == {"d2", "d3"} and got_or == {"d2", "d3"} and self_cos > 0.99, f"self cosine {self_cos:.4f}")
