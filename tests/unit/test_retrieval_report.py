import pytest

from citerag.eval.retrieval import first_gold_rank, score_retrieval
from citerag.rag.context import Chunk
from citerag.schemas import GoldenItem, GoldenSource


def chunk(doc, page):
    return Chunk(chunk_id=f"{doc}-{page}", doc_id=doc, doc_name=doc, page=page, text="t")


def item(item_id, doc="a.pdf", page=2, kind="answerable"):
    sources = [GoldenSource(doc=doc, page=page)] if kind != "unanswerable" else []
    return GoldenItem(id=item_id, question=item_id, type=kind, gold_sources=sources)


RANKED = {
    "first": [chunk("a.pdf", 2), chunk("b.pdf", 1)],  # gold at rank 1
    "third": [chunk("b.pdf", 1), chunk("b.pdf", 2), chunk("a.pdf", 2)],  # gold at rank 3
    "miss": [chunk("b.pdf", 1)],  # never found
    "none": [chunk("b.pdf", 1)],
}
COSINES = {"first": 0.9, "third": 0.8, "miss": 0.5, "none": 0.4}


def _score():
    items = [item("first"), item("third"), item("miss"), item("none", kind="unanswerable")]
    return score_retrieval(items, RANKED.__getitem__, COSINES.__getitem__, ks=(1, 3, 5), mrr_k=5)


def test_only_answerable_kinds_are_scored():
    report = _score()
    assert report.n_scored == 3 and report.n_items == 4


def test_hit_counts_and_rates():
    report = _score()
    assert report.hits == {1: 1, 3: 2, 5: 2}
    assert report.hit_rate[3] == pytest.approx(2 / 3)


def test_mrr_uses_the_first_relevant_rank():
    assert _score().mrr == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_per_item_rows_show_rank_and_top_sources():
    rows = {r.id: r for r in _score().items}
    assert rows["third"].first_hit_rank == 3 and rows["miss"].first_hit_rank is None
    assert rows["first"].top_sources[0] == "a.pdf p2"
    assert rows["none"].scored is False


def test_gate_cosines_are_reported_for_every_item_type():
    report = _score()
    assert report.answerable_cosines == [0.9, 0.8, 0.5]
    assert report.unanswerable_cosines == [0.4]


def test_first_gold_rank_counts_from_one():
    gold = [GoldenSource(doc="a.pdf", page=2)]
    assert first_gold_rank(RANKED["third"], gold) == 3
    assert first_gold_rank(RANKED["miss"], gold) is None
    assert first_gold_rank([], gold) is None


def test_empty_scored_set_does_not_divide_by_zero():
    report = score_retrieval([item("none", kind="unanswerable")], RANKED.__getitem__, COSINES.__getitem__)
    assert report.n_scored == 0 and report.mrr == 0.0 and report.hit_rate[1] == 0.0
