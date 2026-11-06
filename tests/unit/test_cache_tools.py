"""Threshold sweep on labelled pairs: pure logic, with a fake embedder."""

import json

import numpy as np
import pytest

from citerag.errors import ValidationFailedError
from citerag.eval.cache_tools import (
    CachePair,
    SweepRow,
    load_pairs,
    load_sweep,
    save_pairs,
    save_sweep,
    score_pair,
    sweep_pairs,
)

VECTORS = {
    "a": [1.0, 0.0],
    "a same": [1.0, 0.01],
    "b": [0.0, 1.0],
    "b near": [0.2, 1.0],
    "return in 30 days": [1.0, 0.0],
    "return in 60 days": [1.0, 0.0],
}


def embed(text):
    return VECTORS[text]


def pair(a, b, label=None, pid="p", kind="paraphrase"):
    return CachePair(
        id=pid,
        question_a=a,
        question_b=b,
        proposed=label or "equivalent",
        kind=kind,
        labelled=label is not None,
        label=label,
    )


def test_a_pair_cannot_be_labelled_without_a_label_or_the_other_way():
    base = {"id": "p", "question_a": "a", "question_b": "b", "proposed": "equivalent", "kind": "paraphrase"}
    with pytest.raises(ValueError):
        CachePair(**base, labelled=True)
    with pytest.raises(ValueError):
        CachePair(**base, labelled=False, label="equivalent")


def test_candidates_that_nobody_labelled_stop_the_sweep_with_a_clear_message():
    with pytest.raises(ValidationFailedError, match="No labelled pairs"):
        sweep_pairs([pair("a", "a same"), pair("b", "b near")], embed)


def test_only_labelled_pairs_count():
    result = sweep_pairs([pair("a", "a same", "equivalent"), pair("b", "a", None)], embed)
    assert (result.n_equivalent, result.n_near_miss) == (1, 0)


def test_a_small_sample_is_flagged_and_nothing_is_recommended_without_both_classes():
    result = sweep_pairs([pair("a", "a same", "equivalent")], embed)
    assert not result.enough_pairs and result.recommended is None


def test_thirty_of_each_is_enough():
    pairs = [pair("a", "a same", "equivalent", f"e{n}") for n in range(30)]
    pairs += [pair("a", "b", "near_miss", f"n{n}") for n in range(30)]
    result = sweep_pairs(pairs, embed)
    assert result.enough_pairs and result.recommended is not None


def test_the_recommendation_comes_from_the_guarded_rows():
    """Same vectors, different number: only the guard separates them, so the sweep must apply it like the cache."""
    pairs = [
        pair("a", "a same", "equivalent", "e"),
        pair("return in 30 days", "return in 60 days", "near_miss", "n", "number"),
    ]
    result = sweep_pairs(pairs, embed)
    assert result.recommended == 0.99
    at_top = next(row for row in result.rows if row.tau == 0.9)
    no_guard = next(row for row in result.rows_without_guard if row.tau == 0.9)
    assert at_top.false_hit_rate == 0.0 and no_guard.false_hit_rate == 1.0


def test_worst_near_misses_come_first_so_the_reviewer_reads_them_first():
    pairs = [pair("a", "b", "near_miss", "far"), pair("b", "b near", "near_miss", "close")]
    pairs.append(pair("a", "a same", "equivalent", "e"))
    assert [p.id for p in sweep_pairs(pairs, embed).worst_near_misses] == ["close", "far"]


def test_similarity_is_cosine_of_unit_vectors():
    score = score_pair(pair("a", "b", "near_miss"), embed)
    assert abs(score.similarity) < 1e-6 and score.guard_ok
    assert score_pair(pair("a", "a same", "equivalent"), lambda _: np.ones(3)).similarity == pytest.approx(
        1.0
    )


def test_pairs_and_sweeps_round_trip_through_files(tmp_path):
    path = tmp_path / "pairs.jsonl"
    save_pairs(path, [pair("a", "b", "near_miss")])
    assert load_pairs(path)[0].label == "near_miss"
    result = sweep_pairs([pair("a", "a same", "equivalent")], embed)
    save_sweep(result, tmp_path / "s" / "sweep.json")
    assert load_sweep(tmp_path / "s" / "sweep.json") == result
    assert load_sweep(tmp_path / "none.json") is None


def test_a_missing_file_says_where_the_candidates_are(tmp_path):
    with pytest.raises(ValidationFailedError, match="cache_pairs.candidates.jsonl"):
        load_pairs(tmp_path / "cache_pairs.jsonl")


def test_a_broken_line_names_the_file(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"id": "p"}) + "\n", encoding="utf-8")
    with pytest.raises(ValidationFailedError, match="pairs.jsonl"):
        load_pairs(path)


def test_sweep_rows_cover_the_whole_range():
    rows = sweep_pairs([pair("a", "a same", "equivalent")], embed).rows
    assert isinstance(rows[0], SweepRow) and rows[0].tau == 0.70 and rows[-1].tau == 1.0
