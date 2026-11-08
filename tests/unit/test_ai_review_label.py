"""An AI review is allowed only under its own label: golden items, cache pairs, run summaries and reports all say so."""

import importlib.util
import sys
from pathlib import Path

from citerag.eval.cache_tools import CachePair, SweepResult, sweep_pairs, threshold_warning
from citerag.eval.modes import ModeColumn, render
from citerag.schemas import GoldenItem, RunSummary

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


accept_golden = _load("accept_golden")
review_pairs = _load("review_cache_pairs")


def item(iid):
    return GoldenItem(id=iid, question="q?", type="unanswerable")


def test_accept_records_who_reviewed_the_item():
    accepted = accept_golden.accept([item("a"), item("b")], [], ["a"], reviewed_by="claude")
    assert [(i.id, i.verified, i.reviewed_by) for i in accepted] == [("a", True, "claude")]


def test_accept_defaults_to_a_human_reviewer():
    assert accept_golden.accept([item("a")], [], ["a"])[0].reviewed_by == "human"


def test_hand_labelling_marks_the_pair_as_labelled_by_a_human(tmp_path):
    pair = CachePair(id="p", question_a="a", question_b="b", proposed="equivalent", kind="paraphrase")
    review_pairs.label_pairs([pair], tmp_path / "labelled.jsonl", lambda _: "e")
    saved = review_pairs.load_pairs(tmp_path / "labelled.jsonl")
    assert saved[0].reviewed_by == "human"


def test_the_sweep_counts_ai_labelled_pairs():
    vectors = {"a": [1.0, 0.0], "a2": [1.0, 0.01], "b": [0.0, 1.0], "b2": [0.2, 1.0]}
    pairs = [
        CachePair(id="1", question_a="a", question_b="a2", proposed="equivalent", kind="paraphrase",
                  labelled=True, label="equivalent", reviewed_by="claude"),
        CachePair(id="2", question_a="b", question_b="b2", proposed="near_miss", kind="number",
                  labelled=True, label="near_miss", reviewed_by="human"),
    ]  # fmt: skip
    assert sweep_pairs(pairs, vectors.__getitem__).ai_labelled == 1


def test_a_threshold_from_ai_labelled_pairs_still_warns():
    sweep = SweepResult(
        n_equivalent=30,
        n_near_miss=30,
        enough_pairs=True,
        max_false_hit_rate=0.02,
        recommended=0.97,
        rows=[],
        rows_without_guard=[],
        worst_near_misses=[],
        ai_labelled=60,
    )
    warning = threshold_warning(True, 0.97, sweep)
    assert warning is not None and "labelled by Claude, not a human" in warning
    assert threshold_warning(True, 0.97, sweep.model_copy(update={"ai_labelled": 0})) is None
    assert threshold_warning(False, 0.97, sweep) is None


def test_the_mode_report_says_when_items_were_ai_reviewed():
    summary = RunSummary(
        run_id="r", status="complete", golden="g", n_items=2, unverified_items=0, ai_reviewed_items=2
    )
    text = render([ModeColumn("e8_agent", summary, [])])
    assert "(provisional)" in text.splitlines()[0]
    assert "AI-reviewed (Claude), not human-verified" in text
