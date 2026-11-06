"""Print each candidate golden item for human review, with the sentence closest to its answer marked.

    uv run python scripts/review_golden.py --out eval/results/golden_review.md
    uv run python scripts/review_golden.py --item a-017

Reads the source files, so it needs no index and makes no LLM request. It changes no item: moving one to
golden.jsonl with verified: true stays with the human reviewer. A marked sentence is only the best guess, chosen by
how many answer words it shares. A low count means the page may not state the answer, so read that page closely.
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
        matches = [(name, _best_in(pages, terms)) for name, pages in docs.items() if pages]
        name, match = max(((n, m) for n, m in matches if m), key=lambda pair: pair[1].found)
        lines.append(f"Closest text in the corpus, from {name}: confirm it does NOT answer the question.")
        lines.extend([_found_line(match, noun), _around(docs[name], match)])
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
