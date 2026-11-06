"""Page matching in scripts/audit_gold_pages.py, on small synthetic documents (no index, no network)."""

import importlib.util
import sys
from pathlib import Path

from citerag.schemas import GoldenItem, GoldenSource

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_gold_pages.py"
_SPEC = importlib.util.spec_from_file_location("audit_gold_pages", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
sys.modules["audit_gold_pages"] = audit  # dataclasses look their module up here
_SPEC.loader.exec_module(audit)


def pages(*texts):
    return {"doc.pdf": {i: audit.normalise(t) for i, t in enumerate(texts, start=1)}}


def item(answer, *gold_pages):
    sources = [GoldenSource(doc="doc.pdf", page=p) for p in gold_pages]
    return GoldenItem(id="a-1", question="q", type="answerable", gold_answer=answer, gold_sources=sources)


def listed(rows):
    return [r.page for r in rows]


def test_key_terms_drop_stopwords_and_keep_identifiers():
    assert audit.key_terms("The SP 800-61r2 (Revision 2) and RV.3 guide.") == [
        "sp",
        "800-61r2",
        "revision",
        "2",
        "rv.3",
        "guide",
    ]


def test_matching_ignores_case_and_word_endings():
    assert audit.pattern("categories").search("one category or two categorized")
    assert audit.pattern("supersedes").search("this superseding text")


def test_identifiers_match_whole():
    rx = audit.pattern("rv.3")
    assert rx.search("practice rv.3 here") and not rx.search("task rv.3.1 here")


def test_a_line_break_hyphen_still_matches():
    assert audit.score({"verification": 1.0}, audit.normalise("verifica-\ntion and more"))[0] == 1.0


def test_identifier_alone_lists_a_page_that_shares_few_terms():
    """The a-007 case: page 3 says only 'Supersedes NIST SP 800-61r2'."""
    stored = pages(
        "This publication supersedes SP 800-61r2 Revision 2 Computer Security Incident Handling Guide.",
        "Supersedes NIST SP 800-61r2 (August 2012).",
        "Unrelated text about the guide.",
    )
    answer = "SP 800-61r2 (Revision 2), the Computer Security Incident Handling Guide."
    assert listed(audit.find_rows([item(answer, 1)], stored, 0.6)) == [2]


def test_terms_found_on_a_page_are_listed_and_gold_pages_are_not():
    stored = pages("Govern, Map, Measure and Manage.", "Govern, Map, Measure and Manage.", "Nothing here.")
    assert listed(audit.find_rows([item("Govern, Map, Measure and Manage.", 1)], stored, 0.6)) == [2]


def test_span_separates_a_stated_list_from_scattered_mentions():
    filler = "words " * 200
    stored = pages(
        "Govern, Map, Manage.", f"Govern, Map, Manage. {filler}", f"Govern {filler} Map {filler} Manage"
    )
    rows = audit.find_rows([item("Govern, Map, Manage.", 1)], stored, 0.6)
    assert listed(rows) == [2, 3] and rows[0].span < 30 < rows[1].span
