"""Turn loaded pages into LlamaIndex nodes with stable ids (ARCHITECTURE section 6, node setup)."""

from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from citerag.ingest.loaders import LoadedPage

_METADATA_KEYS = ["doc_id", "doc_name", "page", "chunk_index", "kind", "image_path"]


def image_node(
    doc_id: str, doc_name: str, sha256: str, text: str, page: int | None, image_path: str
) -> TextNode:
    """One chunk per image, so a caption is retrieved and cited like any passage."""
    metadata: dict[str, str | int] = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "kind": "image",
        "image_path": image_path,
    }
    if page is not None:
        metadata["page"] = page
    node = TextNode(
        id_=f"{doc_id}-img-{sha256[:8]}",
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=_METADATA_KEYS,
        excluded_llm_metadata_keys=_METADATA_KEYS,
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def chunk_pages(
    pages: list[LoadedPage], doc_id: str, doc_name: str, chunk_size: int, chunk_overlap: int
) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: list[TextNode] = []
    for page in pages:
        for text in splitter.split_text(page.text):
            index = len(nodes)
            metadata: dict[str, str | int] = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_index": index,
                "kind": "text",
            }
            if page.page is not None:  # Chroma metadata has no null
                metadata["page"] = page.page
            node = TextNode(
                id_=f"{doc_id}-{index:04d}",
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=_METADATA_KEYS,  # otherwise metadata is embedded with the text
                excluded_llm_metadata_keys=_METADATA_KEYS,
            )
            # LlamaIndex overwrites metadata doc_id with ref_doc_id in the vector store, so the two must match
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
            nodes.append(node)
    return nodes
