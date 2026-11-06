"""Step 15b (LOCAL): why filters return nothing and self cosine is low: metadata in embeddings, doc_id collision."""
import tempfile
from common import result
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5", query_instruction=" ", text_instruction=" ")
TEXTS = [("d1", "Items can be returned within thirty days of delivery."), ("d2", "The warranty covers manufacturing defects for two years."), ("d3", "Zebrafish are a common model organism in genetics.")]
KEYS = ["doc_id", "doc_name", "page", "chunk_index", "kind"]

def node(i, d, t):
    n = TextNode(id_=f"{d}-{i:04d}", text=t, metadata={"doc_id": d, "doc_name": f"{d}.pdf", "page": 1, "chunk_index": i, "kind": "text"}, excluded_embed_metadata_keys=KEYS, excluded_llm_metadata_keys=KEYS)
    n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=d)
    return n

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    for label, mk in (("bare TextNode", lambda i, d, t: TextNode(id_=f"{d}-{i:04d}", text=t, metadata={"doc_id": d})), ("SOURCE relation + excluded keys", node)):
        col = chromadb.PersistentClient(path=f"{tmp}/{label[:4]}").get_or_create_collection("docs", configuration={"hnsw": {"space": "cosine"}})
        sc = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=col))
        nodes = [mk(i, d, t) for i, (d, t) in enumerate(TEXTS)]
        index = VectorStoreIndex(nodes, storage_context=sc, store_nodes_override=True)
        print("==", label); print("chroma metadata:", col.get(limit=1, include=["metadatas"])["metadatas"][0])
        emb = Settings.embed_model.get_query_embedding(TEXTS[0][1])
        print("self cosine:", round(1 - col.query(query_embeddings=[emb], n_results=1)["distances"][0][0], 4))
        for name, f in (
            ("EQ", MetadataFilters(filters=[MetadataFilter(key="doc_id", value="d2")])),
            ("IN", MetadataFilters(filters=[MetadataFilter(key="doc_id", value=["d2", "d3"], operator=FilterOperator.IN)])),
            ("OR", MetadataFilters(filters=[MetadataFilter(key="doc_id", value="d2"), MetadataFilter(key="doc_id", value="d3")], condition=FilterCondition.OR)),
        ):
            got = sorted({n.node.metadata.get("doc_id") for n in index.as_retriever(similarity_top_k=3, filters=f).retrieve("refund")}, key=str)
            print(f"  filter {name}: {got}")
        n0 = index.as_retriever(similarity_top_k=1).retrieve(TEXTS[0][1])[0].node
        print("  retrieved node metadata:", n0.metadata, "ref_doc_id:", n0.ref_doc_id)
result(True, "see lines above")
