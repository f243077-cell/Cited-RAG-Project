from citerag.ingest.chunking import chunk_pages
from citerag.ingest.loaders import LoadedPage

KEYS = {"doc_id", "doc_name", "chunk_index", "kind"}


def _nodes(pages, size=64, overlap=8):
    return chunk_pages(pages, "abc123", "a.pdf", chunk_size=size, chunk_overlap=overlap)


def test_stable_ids_and_metadata():
    nodes = _nodes([LoadedPage(1, "Refunds take five days."), LoadedPage(2, "Shipping takes three days.")])
    assert [n.id_ for n in nodes] == ["abc123-0000", "abc123-0001"]
    assert nodes[1].metadata == {
        "doc_id": "abc123",
        "doc_name": "a.pdf",
        "chunk_index": 1,
        "kind": "text",
        "page": 2,
    }


def test_text_files_have_no_page_key():
    assert "page" not in _nodes([LoadedPage(None, "Plain text.")])[0].metadata


def test_source_relationship_keeps_doc_id_in_the_vector_store():
    assert _nodes([LoadedPage(1, "Some text.")])[0].ref_doc_id == "abc123"


def test_metadata_stays_out_of_the_embedded_text():
    node = _nodes([LoadedPage(1, "Some text.")])[0]
    assert KEYS | {"page"} <= set(node.excluded_embed_metadata_keys)
    assert KEYS | {"page"} <= set(node.excluded_llm_metadata_keys)
    assert node.get_content(metadata_mode="embed") == "Some text."


def test_long_page_splits_and_every_chunk_keeps_its_page():
    text = " ".join(f"Sentence number {i} about refunds." for i in range(200))
    nodes = _nodes([LoadedPage(7, text)])
    assert len(nodes) > 3
    assert {n.metadata["page"] for n in nodes} == {7}
    assert [n.metadata["chunk_index"] for n in nodes] == list(range(len(nodes)))


def test_ids_do_not_depend_on_anything_but_the_input():
    pages = [LoadedPage(1, "Same text every time.")]
    assert [n.id_ for n in _nodes(pages)] == [n.id_ for n in _nodes(pages)]
