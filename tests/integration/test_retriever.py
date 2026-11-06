"""Hybrid retrieval over real Chroma, BM25 and fusion, with HashEmbedding instead of the model download."""

import logging

import pytest

from citerag.index.retriever import HybridRetriever
from citerag.index.store import IndexStore
from citerag.ingest.pipeline import Ingestor
from citerag.rag.context import Chunk
from tests.helpers import HashEmbedding, hash_vector
from tests.pdf_fixture import make_pdf

POLICY = make_pdf(
    ["Items can be returned within thirty days of delivery.", "Shipping takes three working days."]
)
WARRANTY = b"The warranty covers manufacturing defects for two years."
FISH = b"Zebrafish are a common model organism in genetics research."


@pytest.fixture
def world(settings):
    store = IndexStore(settings.index_dir, HashEmbedding())
    ingestor = Ingestor(settings, store, settings.index_dir)
    ids = {
        "policy": ingestor.ingest_bytes("policy.pdf", POLICY).doc_id,
        "warranty": ingestor.ingest_bytes("warranty.md", WARRANTY).doc_id,
        "fish": ingestor.ingest_bytes("fish.txt", FISH).doc_id,
    }
    yield settings, store, ids
    store.close()


def _retriever(settings, store, **kw):
    return HybridRetriever(store, settings, **kw)


def test_returns_plain_chunks_with_document_fields(world):
    settings, store, ids = world
    top = _retriever(settings, store).retrieve("how many days to return items")[0]
    assert type(top) is Chunk
    assert (top.doc_id, top.doc_name, top.page, top.kind) == (ids["policy"], "policy.pdf", 1, "text")
    assert top.chunk_id == f"{ids['policy']}-0000" and "thirty days" in top.text


def test_text_files_have_no_page(world):
    settings, store, _ = world
    assert _retriever(settings, store).retrieve("warranty manufacturing defects")[0].page is None


def test_fusion_merges_the_same_chunk_from_both_retrievers(world):
    settings, store, _ = world
    chunks = _retriever(settings, store).candidates("warranty manufacturing defects")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert chunks[0].doc_name == "warranty.md"
    assert chunks[0].score == pytest.approx(2 / 61, abs=0.002)  # rank 1 in both lists: 2 x 1/(60 + 1)


def test_bm25_finds_a_rare_keyword_that_vectors_alone_may_miss(world):
    settings, store, _ = world
    assert _retriever(settings, store).retrieve("zebrafish")[0].doc_name == "fish.txt"


def test_doc_ids_filter_with_one_and_several_documents(world):
    settings, store, ids = world
    retriever = _retriever(settings, store)
    one = retriever.candidates("warranty and shipping", doc_ids=[ids["policy"]])
    assert one and {c.doc_id for c in one} == {ids["policy"]}
    two = retriever.candidates("warranty and shipping", doc_ids=[ids["policy"], ids["fish"]])
    assert {c.doc_id for c in two} <= {ids["policy"], ids["fish"]} and ids["warranty"] not in {
        c.doc_id for c in two
    }


def test_empty_or_unknown_filter_returns_nothing(world):
    settings, store, _ = world
    retriever = _retriever(settings, store)
    assert retriever.candidates("warranty", doc_ids=[]) == []
    assert retriever.candidates("warranty", doc_ids=["unknown"]) == []


def test_retrieve_cuts_to_top_k_context_and_candidates_to_top_k_retrieve(world):
    settings, store, _ = world
    retriever = _retriever(settings.model_copy(update={"top_k_context": 2, "top_k_retrieve": 3}), store)
    assert len(retriever.retrieve("days")) == 2
    assert len(retriever.candidates("days")) == 3


def test_empty_index_returns_nothing(settings):
    store = IndexStore(settings.index_dir, HashEmbedding())
    retriever = _retriever(settings, store)
    assert retriever.retrieve("anything") == [] and retriever.top_cosine([1.0] + [0.0] * 127) == 0.0
    store.close()


def test_refresh_picks_up_new_and_replaced_documents_for_bm25(world):
    settings, store, _ = world
    retriever = _retriever(settings, store)
    ingestor = Ingestor(settings, store, settings.index_dir)
    ingestor.ingest_bytes("extra.md", b"Quokkas live on Rottnest Island.")
    retriever.refresh()
    assert retriever.retrieve("quokkas")[0].doc_name == "extra.md"
    ingestor.ingest_bytes("extra.md", b"Wombats dig long burrows.")
    retriever.refresh()
    assert all("Quokkas" not in c.text for c in retriever.candidates("quokkas"))


def test_gate_uses_true_cosine(world):
    settings, store, ids = world
    retriever = _retriever(settings, store)
    sentence = "The warranty covers manufacturing defects for two years."
    assert retriever.top_cosine(hash_vector(sentence)) == pytest.approx(1.0, abs=0.01)
    assert retriever.passes_gate(hash_vector(sentence))
    unrelated = hash_vector("completely different words about nothing here")
    assert retriever.top_cosine(unrelated) < 0.3 and not retriever.passes_gate(unrelated)


def test_gate_respects_the_document_filter(world):
    settings, store, ids = world
    retriever = _retriever(settings, store)
    sentence = hash_vector("The warranty covers manufacturing defects for two years.")
    assert retriever.top_cosine(sentence, doc_ids=[ids["fish"]]) < 0.9
    assert retriever.top_cosine(sentence, doc_ids=[ids["warranty"]]) == pytest.approx(1.0, abs=0.01)
    assert retriever.top_cosine(sentence, doc_ids=[]) == 0.0


class ReverseReranker:
    """Stands in for the cross-encoder: reverses the list and honours top_n."""

    def __init__(self):
        self.calls = 0

    def postprocess_nodes(self, nodes, query_bundle):
        self.calls += 1
        return list(reversed(nodes))[:2]


class BrokenReranker:
    def postprocess_nodes(self, nodes, query_bundle):
        raise RuntimeError("model missing")


def test_rerank_switch(world):
    settings, store, _ = world
    fake = ReverseReranker()
    off = _retriever(settings, store, reranker=fake)
    off.retrieve("days")
    assert fake.calls == 0  # RERANK_ENABLED defaults to false
    cfg = settings.model_copy(update={"rerank_enabled": True})
    plain = _retriever(settings, store).candidates("warranty manufacturing defects")
    reranked = _retriever(cfg, store, reranker=fake).retrieve("warranty manufacturing defects")
    assert fake.calls == 1 and len(reranked) == 2
    assert reranked[0].chunk_id == plain[-1].chunk_id


def test_rerank_failure_falls_back_to_the_fused_order(world, caplog):
    settings, store, _ = world
    cfg = settings.model_copy(update={"rerank_enabled": True, "top_k_context": 2})
    plain = _retriever(settings.model_copy(update={"top_k_context": 2}), store).retrieve("days")
    with caplog.at_level(logging.WARNING):
        out = _retriever(cfg, store, reranker=BrokenReranker()).retrieve("days")
    assert [c.chunk_id for c in out] == [c.chunk_id for c in plain]
    assert "rerank failed" in caplog.text


def test_no_framework_objects_leave_the_retriever(world):
    settings, store, _ = world
    retriever = _retriever(settings, store)
    for chunk in retriever.retrieve("days") + retriever.candidates("days"):
        assert isinstance(chunk, Chunk) and isinstance(chunk.score, float)
