"""Print each candidate golden item for human review, with the sentence closest to its answer marked.

    uv run python scripts/review_golden.py --out eval/results/golden_review.md
    uv run python scripts/review_golden.py --item a-017

Reads the source files, so it needs no index and makes no LLM request. It changes no item: moving one to
golden.jsonl with verified: true stays with the human reviewer. A marked sentence is only the best guess, chosen by
how many answer words it shares. A low count means the page may not state the answer, so read that page closely.

Sentence matching compares terms by their first 5 characters, so 800-61r2 and 800-61r3 look identical there, and a
marked sentence can name the wrong revision. The term list for unanswerable items matches identifiers whole.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from citerag.eval.golden import load_golden
from citerag.ingest.loaders import load_pages, validate_file
from citerag.schemas import GoldenItem, GoldenSource

STOPWORDS = set(
    "a an and are as at be being both by do does for from has how in into is it its no not of on or rather than that "
    "the their them then there they this to used using was were what when which who why with you your".split()
)
TOKEN = re.compile(r"[a-z0-9]+(?:[-.:/][a-z0-9]+)*")
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[•])|\s+(?=•)")
STEM = 5  # catches categories/category and manage/managing without a stemmer
CONTEXT_SENTENCES = 2  # whole-document gold has no page, so show this much around the match
MAX_MB = 100
COMMON_SHARE = 0.10  # a term on more than this share of all pages says nothing about the question
COMMON_MIN_PAGES = 5  # a small corpus must not call a term common for sitting on two pages
Pages = dict[int | None, str]


@dataclass(frozen=True)
class Match:
    sentence: str
    found: int
    wanted: int
    page: int | None = None


def _stem(word: str) -> str:
    return word[:STEM]


def key_terms(text: str) -> set[str]:
    return {
        _stem(w) for w in TOKEN.findall(text.lower()) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())
    }


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text.replace("-\n", "")).strip()
    return [s.strip() for s in SENTENCE_END.split(flat) if s.strip()]


def best_sentence(text: str, terms: set[str], page: int | None = None) -> Match | None:
    """Most answer terms first; the shorter sentence wins a tie, because a long one matches by size alone."""
    best: Match | None = None
    for sentence in split_sentences(text):
        found = len(terms & key_terms(sentence))
        better = best is None or found > best.found
        if better or (found == best.found and len(sentence) < len(best.sentence)):
            best = Match(sentence, found, len(terms), page)
    return best


def mark(text: str, sentence: str) -> str:
    flat = re.sub(r"\s+", " ", text.replace("-\n", "")).strip()
    return flat.replace(sentence, f">>> {sentence} <<<", 1)


def load_docs(roots: list[Path]) -> dict[str, Pages]:
    docs: dict[str, Pages] = {}
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SOURCES.md"):
            data = path.read_bytes()
            loaded = load_pages(validate_file(path.name, data, MAX_MB), data)
            docs[path.name] = {p.page: p.text for p in loaded.pages}
    return docs


def _best_in(pages: Pages, terms: set[str]) -> Match | None:
    matches = [best_sentence(text, terms, page) for page, text in pages.items()]
    return max((m for m in matches if m), key=lambda m: m.found, default=None)


def _around(pages: Pages, match: Match) -> str:
    sentences = split_sentences(pages[match.page])
    at = sentences.index(match.sentence)
    start = max(at - CONTEXT_SENTENCES, 0)
    return mark(" ".join(sentences[start : at + CONTEXT_SENTENCES + 1]), match.sentence)


def _found_line(match: Match | None, noun: str) -> str:
    if match is None:
        return "No text to match."
    where = "" if match.page is None else f" on p{match.page}"
    return f"Best match{where}: shares {match.found} of {match.wanted} {noun} terms."


def _word_pattern(word: str) -> re.Pattern[str]:
    """Loose on purpose: a stray page costs a glance, but a false "not found" hides an answer from the reviewer."""
    if not word.isalpha():  # identifiers match whole: 800-61 must not match 800-61r3
        return re.compile(rf"(?<![a-z0-9.-]){re.escape(word)}(?![.-]?[a-z0-9])")
    singular = re.sub(r"ies$", "y", word)
    singular = singular[:-1] if singular.endswith("s") and not singular.endswith("ss") else singular
    if len(singular) <= 4:
        return re.compile(rf"(?<![a-z0-9]){re.escape(singular)}(?:s|es|d|ed|ing)?(?![a-z0-9])")
    return re.compile(rf"(?<![a-z0-9]){re.escape(singular[: max(4, len(singular) - 3)])}[a-z]*")


def _pages_with(word: str, flat: dict[tuple[str, int | None], str]) -> list[tuple[str, int | None]]:
    rx = _word_pattern(word)
    return [where for where, text in flat.items() if rx.search(text)]


def _show_pages(found: list[tuple[str, int | None]]) -> str:
    by_doc: dict[str, list[str]] = {}
    for doc, page in found:
        by_doc.setdefault(doc, [])
        if page is not None:
            by_doc[doc].append(f"p{page}")
    return "; ".join(f"{doc} {', '.join(pages)}".strip() for doc, pages in by_doc.items())


def term_lines(question: str, docs: dict[str, Pages]) -> list[str]:
    """Where each distinctive question word sits, so a reviewer can see what the corpus never mentions."""
    words = dict.fromkeys(
        w for w in TOKEN.findall(question.lower()) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())
    )
    flat = {
        (doc, page): re.sub(r"\s+", " ", text.replace("-\n", "")).lower()
        for doc, pages in docs.items()
        for page, text in pages.items()
    }
    limit = max(COMMON_MIN_PAGES, COMMON_SHARE * len(flat))
    distinctive, common = [], []
    for word in words:
        found = _pages_with(word, flat)
        if len(found) > limit:
            common.append(word)
        else:
            distinctive.append(f"  {word}: {_show_pages(found) or 'not found in the corpus'}")
    lines = (
        ["Distinctive question terms:", *distinctive] if distinctive else ["Distinctive question terms: none"]
    )
    return [*lines, f"Left out as common in the corpus: {', '.join(common)}"] if common else lines


def _closest_text(docs: dict[str, Pages], terms: set[str], noun: str) -> list[str]:
    matches = [(name, _best_in(pages, terms)) for name, pages in docs.items() if pages]
    usable = [(name, match) for name, match in matches if match is not None]
    if not usable:
        return ["No text in the corpus to compare."]
    name, match = max(usable, key=lambda pair: pair[1].found)
    if match.found == 0:
        return ["No text in the corpus shares a question term."]
    return [
        f"Closest text in the corpus, from {name}: confirm it does NOT answer the question.",
        _found_line(match, noun),
        _around(docs[name], match),
    ]


def render_source(source: GoldenSource, docs: dict[str, Pages], terms: set[str], noun: str) -> list[str]:
    part = f", part {source.part}"
    pages = docs.get(source.doc)
    if pages is None:
        return [f"--- {source.doc}{part}: FILE NOT FOUND ---"]
    if not pages:
        return [f"--- {source.doc}{part}: no text, an image file ---"]
    if source.page is None:
        match = _best_in(pages, terms)
        lines = [
            f"--- {source.doc}{part}: whole document, showing the best match ---",
            _found_line(match, noun),
        ]
        return [*lines, _around(pages, match)] if match else lines
    text = pages.get(source.page)
    if text is None:
        return [f"--- {source.doc} p{source.page}{part}: PAGE NOT FOUND ---"]
    match = best_sentence(text, terms, source.page)
    lines = [f"--- {source.doc} p{source.page}{part} ---", _found_line(match, noun)]
    return [*lines, mark(text, match.sentence) if match else text]


def render_item(item: GoldenItem, docs: dict[str, Pages]) -> str:
    lines = [f"\n# {item.id} ({item.type}, verified: {str(item.verified).lower()})", f"Q: {item.question}"]
    lines.append(f"Gold answer: {item.gold_answer or '(none)'}")
    if item.image_path:
        lines.append(f"Image: {item.image_path}")
    if item.canary:
        lines.append(f"Canary, must not appear in an answer: {item.canary}")
    terms, noun = (
        (key_terms(item.gold_answer), "answer")
        if item.gold_answer
        else (key_terms(item.question), "question")
    )
    for source in item.gold_sources:
        lines.extend(render_source(source, docs, terms, noun))
    if item.type == "unanswerable":
        lines.extend(_closest_text(docs, terms, noun))
        lines.extend(term_lines(item.question, docs))
    return "\n".join(lines)


def main(
    golden: Annotated[Path, typer.Option()] = Path("eval/golden/candidates.jsonl"),
    source: Annotated[list[Path] | None, typer.Option(help="Folder of source documents")] = None,
    item: Annotated[str | None, typer.Option(help="Only this item id")] = None,
    out: Annotated[Path | None, typer.Option(help="Write here instead of the console")] = None,
) -> None:
    roots = source or [Path("data/sample"), Path("eval/fixtures/poisoned")]
    items = [i for i in load_golden(golden) if item is None or i.id == item]
    docs = load_docs(roots)
    text = "\n".join(render_item(i, docs) for i in items)
    if out is None:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )  # PDF text has characters the Windows console code page lacks
        typer.echo(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"{len(items)} item(s) written to {out}")


if __name__ == "__main__":
    typer.run(main)
