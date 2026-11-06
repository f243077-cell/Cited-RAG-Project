"""scripts/review_cache_pairs.py and scripts/make_workload.py, plus the shipped candidate pairs."""

import importlib.util
import json
import sys
from pathlib import Path

from citerag.eval.cache_tools import CachePair, load_pairs
from citerag.schemas import GoldenItem

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


review = _load("review_cache_pairs")
workload = _load("make_workload")


def pair(pid, proposed="equivalent", source="a-1", label=None, b="How long can items come back?"):
    return CachePair(
        id=pid,
        question_a="How long do I have to return an item?",
        question_b=b,
        proposed=proposed,
        kind="paraphrase" if proposed == "equivalent" else "number",
        source=source,
        labelled=label is not None,
        label=label,
    )


def test_the_shipped_candidates_are_all_unlabelled_and_enough_for_the_sweep():
    pairs = load_pairs(ROOT / "eval" / "cache_pairs.candidates.jsonl")
    assert len({p.id for p in pairs}) == len(pairs)
    assert not any(p.labelled or p.label for p in pairs)
    assert sum(p.proposed == "equivalent" for p in pairs) >= 30
    assert sum(p.proposed == "near_miss" for p in pairs) >= 30
    assert all(p.question_a != p.question_b for p in pairs)


def test_each_candidate_with_a_source_starts_from_that_exact_golden_question():
    lines = (ROOT / "eval" / "golden" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    golden = {item["id"]: item["question"] for item in map(json.loads, lines)}
    for p in load_pairs(ROOT / "eval" / "cache_pairs.candidates.jsonl"):
        if p.source is not None:
            assert p.question_a == golden[p.source], p.id


def test_a_review_entry_shows_both_questions_the_difference_and_the_guard():
    text = review.render_pair(pair("p1", b="How long do I have to return 2 items?"))
    assert "A: How long do I have" in text and "B: How long do I have to return 2 items?" in text
    assert "Only in A: an, item. Only in B: 2, items." in text
    assert "Guard: blocks a hit." in text and "Label: not labelled" in text


def test_labelling_saves_only_what_the_human_typed_and_can_continue(tmp_path):
    out = tmp_path / "pairs.jsonl"
    pairs = [pair("p1"), pair("p2"), pair("p3"), pair("p4")]
    answers = iter(["e", "s", "n", "q"])
    assert review.label_pairs(pairs, out, lambda _: next(answers)) == 2
    saved = {p.id: p for p in load_pairs(out)}
    assert saved["p1"].label == "equivalent" and saved["p3"].label == "near_miss" and "p2" not in saved
    seen = []
    review.label_pairs(pairs, out, lambda p: seen.append(p.id) or "q")
    assert seen == ["p2"]


def test_a_label_can_disagree_with_the_proposal(tmp_path):
    out = tmp_path / "pairs.jsonl"
    review.label_pairs([pair("p1", proposed="equivalent")], out, lambda _: "n")
    assert load_pairs(out)[0].label == "near_miss"


def test_a_stray_key_labels_nothing(tmp_path):
    out = tmp_path / "pairs.jsonl"
    assert review.label_pairs([pair("p1")], out, lambda _: "x") == 0
    assert not out.exists()


def golden_items():
    return [
        GoldenItem(id="a-1", question="Q1", type="answerable", gold_answer="A1"),
        GoldenItem(id="u-1", question="Q2", type="unanswerable"),
    ]


def test_the_workload_adds_a_repeat_for_each_equivalent_pair_only():
    pairs = [
        pair("p1", label="equivalent"),
        pair("p2", label="near_miss"),
        pair("p3", source="u-1", label="equivalent"),
        pair("p4", source=None, label="equivalent"),
    ]
    items = workload.build_workload(golden_items(), pairs)
    assert [i.id for i in items] == ["a-1", "a-1-r001"]
    assert items[1].repeat_of == "a-1" and items[1].gold_answer == "A1"
    assert items[1].question == pairs[0].question_b
