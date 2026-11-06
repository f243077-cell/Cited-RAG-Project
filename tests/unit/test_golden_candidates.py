"""The draft set must keep the shape PRD F-EVA-1 asks for, and its fixtures must exist."""

from collections import Counter
from pathlib import Path

from citerag.eval.golden import load_golden

ROOT = Path(__file__).resolve().parents[2]
ITEMS = load_golden(ROOT / "eval" / "golden" / "candidates.jsonl")
POISONED = ROOT / "eval" / "fixtures" / "poisoned"


def _of(kind):
    return [item for item in ITEMS if item.type == kind]


def test_counts_meet_f_eva_1():
    counts = Counter(item.type for item in ITEMS)
    assert len(ITEMS) >= 40
    assert counts["answerable"] >= 20 and counts["multi_chunk"] >= 5 and counts["unanswerable"] >= 5
    assert counts["injection"] >= 5 and counts["image"] >= 5


def test_candidates_stay_unverified():
    assert not [item.id for item in ITEMS if item.verified]  # a human moves items to golden.jsonl


def test_answerable_kinds_have_answers_and_sources():
    for item in ITEMS:
        if item.type != "unanswerable":
            assert item.gold_answer and item.gold_sources, item.id


def test_unanswerable_items_have_no_sources():
    assert not [item.id for item in _of("unanswerable") if item.gold_sources or item.gold_answer]


def test_multi_chunk_items_need_more_than_one_source():
    assert all(len(item.gold_sources) >= 2 for item in _of("multi_chunk"))


def test_every_injection_has_a_unique_canary_planted_in_its_fixture():
    items = _of("injection")
    assert len({item.canary for item in items}) == len(items)
    for item in items:
        path = POISONED / item.gold_sources[0].doc
        assert path.exists(), item.id
        if path.suffix != ".png":  # the PNG canary is text inside the picture
            assert item.canary in path.read_text(encoding="utf-8"), item.id


def test_one_injection_is_hidden_in_an_image():
    assert [item.id for item in _of("injection") if item.gold_sources[0].doc.endswith(".png")]


def test_image_items_point_at_existing_files():
    for item in _of("image"):
        assert (ROOT / item.image_path).stat().st_size < 5 * 1024 * 1024, item.id  # F-MM-1 limit
