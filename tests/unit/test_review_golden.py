"""Sentence matching in scripts/review_golden.py, on small synthetic documents (no index, no network)."""

import importlib.util
import sys
from pathlib import Path

from citerag.schemas import GoldenItem, GoldenSource

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "review_golden.py"
_SPEC = importlib.util.spec_from_file_location("review_golden", _PATH)
review = importlib.util.module_from_spec(_SPEC)
sys.modules["review_golden"] = review  # dataclasses look their module up here
_SPEC.loader.exec_module(review)

DOCS = {
    "doc.pdf": {
        1: "Risk matters to everyone. The four functions are Govern, Map, Measure and Manage. Thanks for reading.",
        2: "Nothing relevant lives on this page at all.",
    },
    "notes.md": {None: "Intro text here. Items can be returned within thirty days. More closing text."},
    "pic.png": {},
}


def item(answer="Govern, Map, Measure and Manage.", sources=None, kind="answerable", **kw):
    sources = [GoldenSource(doc="doc.pdf", page=1)] if sources is None else sources
    return GoldenItem(
        id="a-1",
        question="What are the functions?",
        type=kind,
        gold_answer=answer,
        gold_sources=sources,
        **kw,
    )


def test_the_marked_sentence_is_the_one_holding_the_answer():
    text = review.render_item(item(), DOCS)
    assert ">>> The four functions are Govern, Map, Measure and Manage. <<<" in text
    assert "shares 4 of 4 answer terms" in text


def test_word_endings_still_match():
    assert review.key_terms("categories") == review.key_terms("category")
    assert (
        review.best_sentence(
            "Managing risk is hard. Other words here.", review.key_terms("manage risk")
        ).found
        == 2
    )


def test_a_page_without_the_answer_shows_a_low_count():
    text = review.render_item(item(sources=[GoldenSource(doc="doc.pdf", page=2)]), DOCS)
    assert "shares 0 of 4 answer terms" in text


def test_missing_page_and_missing_file_are_called_out():
    assert "PAGE NOT FOUND" in review.render_item(item(sources=[GoldenSource(doc="doc.pdf", page=9)]), DOCS)
    assert "FILE NOT FOUND" in review.render_item(item(sources=[GoldenSource(doc="gone.pdf", page=1)]), DOCS)


def test_whole_document_gold_shows_the_match_in_context_not_the_file():
    source = GoldenSource(doc="notes.md")
    text = review.render_item(item("Returned within thirty days.", [source]), DOCS)
    assert ">>> Items can be returned within thirty days. <<<" in text and "whole document" in text


def test_parts_are_shown_for_multi_chunk_sources():
    source = GoldenSource(doc="doc.pdf", page=1, part=1)
    assert "doc.pdf p1, part 1" in review.render_item(item(sources=[source]), DOCS)


def test_an_image_source_has_no_text_and_does_not_fail():
    text = review.render_item(item(sources=[GoldenSource(doc="pic.png")]), DOCS)
    assert "no text, an image file" in text


def test_unanswerable_items_show_the_closest_corpus_text_to_rule_out():
    unanswerable = GoldenItem(id="u-1", question="What are the four functions?", type="unanswerable")
    text = review.render_item(unanswerable, DOCS)
    assert "does NOT answer" in text and ">>> The four functions are" in text


def test_injection_items_show_their_canary():
    injected = GoldenItem(id="i-1", question="Summarise.", type="injection", canary="CANARY-1")
    assert "CANARY-1" in review.render_item(injected, DOCS)


def test_every_candidate_renders_against_small_docs():
    from citerag.eval.golden import load_golden

    root = _PATH.parents[1]
    for candidate in load_golden(root / "eval" / "golden" / "candidates.jsonl"):
        assert candidate.id in review.render_item(candidate, DOCS)


def unanswerable(question):
    return GoldenItem(id="u-1", question=question, type="unanswerable")


TERM_DOCS = {
    "a.pdf": {
        1: "The annual fee is listed here.",
        2: "Managing verifica-\ntion takes time.",
        3: "Framework text.",
    },
    "b.md": {None: "Hours of support are shown."},
    **{f"f{n}.pdf": {1: "The framework again."} for n in range(5)},
    "c.pdf": {4: "The framework once more."},
}


def test_a_distinctive_term_lists_each_page_it_appears_on():
    assert "  annual: a.pdf p1" in review.render_item(unanswerable("What is the annual cost?"), TERM_DOCS)


def test_a_missing_term_says_so():
    text = review.render_item(unanswerable("What is the recertification rule?"), TERM_DOCS)
    assert "  recertification: not found in the corpus" in text


def test_word_forms_never_produce_a_false_not_found():
    text = review.render_item(unanswerable("Fees, hour, manage and verification?"), TERM_DOCS)
    assert "  fees: a.pdf p1" in text and "  hour: b.md" in text
    assert "  manage: a.pdf p2" in text and "  verification: a.pdf p2" in text  # across a line-break hyphen


def test_identifiers_match_whole():
    docs = {"a.pdf": {1: "See SP 800-61r3 here."}}
    text = review.render_item(unanswerable("Does 800-61 or 800-61r3 say so?"), docs)
    assert "  800-61r3: a.pdf p1" in text and "  800-61: not found in the corpus" in text


def test_common_corpus_words_and_stopwords_are_left_out_of_the_term_list():
    text = review.render_item(unanswerable("What does the framework say about recertification?"), TERM_DOCS)
    terms = text.split("Distinctive question terms:")[1]
    assert "  framework:" not in terms and "  what:" not in terms
    assert "Left out as common in the corpus: framework" in terms


def test_a_corpus_with_no_text_does_not_crash():
    text = review.render_item(unanswerable("What is the annual cost?"), {"pic.png": {}})
    assert "No text in the corpus to compare." in text
    assert "  annual: not found in the corpus" in text


def test_an_empty_corpus_does_not_crash():
    assert "not found in the corpus" in review.render_item(unanswerable("What is the annual cost?"), {})


def test_nothing_matching_shows_no_arbitrary_sentence():
    text = review.render_item(
        unanswerable("Tungsten boiling point?"), {"a.pdf": {1: "Unrelated words only here."}}
    )
    assert "No text in the corpus shares a question term." in text
    assert ">>>" not in text and "  tungsten: not found in the corpus" in text


def test_a_question_of_only_stopwords_does_not_crash():
    text = review.render_item(unanswerable("What is it?"), {"a.pdf": {1: "Some text."}})
    assert "Distinctive question terms: none" in text


def test_the_closest_text_line_stays_for_a_real_match():
    text = review.render_item(unanswerable("What is the annual fee?"), TERM_DOCS)
    assert "confirm it does NOT answer" in text and ">>> The annual fee is listed here. <<<" in text


def test_sentence_matching_cannot_tell_revisions_apart_but_the_term_list_can():
    assert review.key_terms("800-61r2") == review.key_terms("800-61r3")  # the limit the docstring states
    docs = {"a.pdf": {1: "See SP 800-61r3 here."}}
    assert "  800-61r2: not found in the corpus" in review.term_lines("Does 800-61r2 apply?", docs)
