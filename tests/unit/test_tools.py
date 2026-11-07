"""The read-only tools on a stub retriever: validation, caps, the gate, the filter and page lookup."""

from datetime import UTC, datetime

import pytest

from citerag.errors import ValidationFailedError
from citerag.ingest.manifest import DocEntry, Manifest
from citerag.rag.context import Chunk
from citerag.tools.contracts import SearchArgs, ToolHit
from citerag.tools.library import NO_RELEVANT, ToolLibrary, cap_hits, validate_args

NOW = datetime(2026, 11, 7, tzinfo=UTC)


def manifest(*docs):
    entries = {
        doc_id: DocEntry(name=name, sha256="x", pages=pages, chunks=1, size_bytes=9, ingested_at=NOW)
        for doc_id, name, pages in docs
    }
    return Manifest(
        embedding_model="m",
        embedding_dim=4,
        chunk_size=512,
        chunk_overlap=64,
        caption_model="v",
        created_at=NOW,
        docs=entries,
    )


MANIFEST = manifest(("d1", "policy.pdf", 4), ("d2", "notes.md", None))


class Retriever:
    def __init__(self, cosine=0.9, chunks=None):
        self.cosine = cosine
        self.chunks = chunks or [
            Chunk(f"d1-000{n}", "d1", "policy.pdf", n + 1, f"text {n}") for n in range(8)
        ]
        self.filters = []

    def embed_query(self, query):
        return [1.0]

    def top_cosine(self, query_embedding, doc_ids=None):
        self.filters.append(doc_ids)
        return self.cosine

    def candidates(self, query, doc_ids=None):
        self.filters.append(doc_ids)
        return [c for c in self.chunks if doc_ids is None or c.doc_id in doc_ids]

    def page_chunks(self, doc_id, page):
        return [c for c in self.chunks if c.doc_id == doc_id and c.page == page]


def library(settings, retriever=None, scope=None, known=MANIFEST):
    return ToolLibrary(settings, retriever or Retriever(), lambda: known, scope)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("search_documents", {"query": ""}),
        ("search_documents", {"query": "x" * 501}),
        ("search_documents", {"query": "q", "k": 0}),
        ("search_documents", {"query": "q", "k": 11}),
        ("search_documents", {"query": "q", "path": "/etc/passwd"}),
        ("read_page", {"doc": "policy.pdf", "page": 0}),
        ("read_page", {"doc": "", "page": 1}),
        ("list_documents", {"extra": 1}),
    ],
)
def test_bad_arguments_are_rejected_without_echoing_them(name, args):
    with pytest.raises(ValidationFailedError) as caught:
        validate_args(name, args)
    assert "/etc/passwd" not in caught.value.message and name in caught.value.message


def test_a_bad_argument_is_a_result_not_an_exception(settings):
    result = library(settings).run("search_documents", {"query": "q", "k": 99})
    assert not result.ok and "k" in result.error and result.hits == []


def test_search_returns_k_hits_with_the_gate_value(settings):
    result = library(settings).run("search_documents", {"query": "refunds", "k": 3})
    assert result.ok and [h.chunk_id for h in result.hits] == ["d1-0000", "d1-0001", "d1-0002"]
    assert result.top_similarity == 0.9 and not result.truncated


def test_below_the_gate_there_is_no_passage_and_no_llm_call(settings):
    result = library(settings, Retriever(cosine=0.1)).run("search_documents", {"query": "q"})
    assert result.ok and result.hits == [] and result.message == NO_RELEVANT and result.top_similarity == 0.1


def test_results_are_cut_to_the_character_budget(settings):
    long = [Chunk(f"d1-{n}", "d1", "policy.pdf", 1, "x" * 4000) for n in range(3)]
    result = library(settings, Retriever(chunks=long)).run("search_documents", {"query": "q", "k": 3})
    assert result.truncated and sum(len(h.text) for h in result.hits) == settings.agent_tool_result_chars
    assert len(result.hits) == 2


def test_cap_hits_keeps_order_and_reports_cutting():
    hits = [ToolHit(chunk_id=str(n), doc_id="d", doc_name="a", text="abcd") for n in range(3)]
    kept, cut = cap_hits(hits, 6)
    assert [h.text for h in kept] == ["abcd", "ab"] and cut
    assert cap_hits(hits, 12) == (hits, False)


def test_the_request_filter_limits_every_search(settings):
    retriever = Retriever()
    result = library(settings, retriever, scope=["d2"]).run("search_documents", {"query": "q"})
    assert retriever.filters[-1] == ["d2"] and result.hits == []


def test_a_search_cannot_widen_the_request_filter(settings):
    retriever = Retriever()
    library(settings, retriever, scope=["d2"]).run(
        "search_documents", {"query": "q", "doc_ids": ["d1", "d2"]}
    )
    assert retriever.filters[-1] == ["d2"]


def test_a_filter_that_leaves_nothing_searches_nothing(settings):
    retriever = Retriever()
    result = library(settings, retriever, scope=["d2"]).run(
        "search_documents", {"query": "q", "doc_ids": ["d1"]}
    )
    assert result.ok and result.message == "no documents in scope" and retriever.filters == []


def test_unknown_doc_ids_are_named(settings):
    result = library(settings).run("search_documents", {"query": "q", "doc_ids": ["nope"]})
    assert not result.ok and result.error == "unknown doc_ids: nope"


def test_an_empty_library_says_so(settings):
    for name, args in (("list_documents", {}), ("search_documents", {"query": "q"})):
        result = library(settings, known=None).run(name, args)
        assert result.ok and result.message == "no documents yet"


def test_read_page_by_id_or_exact_name(settings):
    for doc in ("d1", "policy.pdf"):
        result = library(settings).run("read_page", {"doc": doc, "page": 2})
        assert result.ok and [h.chunk_id for h in result.hits] == ["d1-0001"]


def test_read_page_never_treats_doc_as_a_path(settings):
    for doc in ("../.env", "data/uploads/d1.pdf", "POLICY.PDF"):
        result = library(settings).run("read_page", {"doc": doc, "page": 1})
        assert not result.ok and result.error == f"unknown document: {doc}"


def test_read_page_past_the_last_page_or_outside_the_filter(settings):
    assert library(settings).run("read_page", {"doc": "d1", "page": 5}).error == "policy.pdf has 4 pages"
    assert not library(settings, scope=["d2"]).run("read_page", {"doc": "d1", "page": 1}).ok


def test_a_page_with_no_stored_text_is_not_an_error(settings):
    result = library(settings).run("read_page", {"doc": "notes.md", "page": 3})
    assert result.ok and result.hits == [] and result.message == "no text stored for page 3"


def test_list_documents_respects_the_filter(settings):
    assert [d.doc_name for d in library(settings).run("list_documents", {}).documents] == [
        "policy.pdf",
        "notes.md",
    ]
    assert [d.doc_id for d in library(settings, scope=["d2"]).run("list_documents", {}).documents] == ["d2"]


def test_search_args_default_k_is_five():
    assert SearchArgs(query="q").k == 5
