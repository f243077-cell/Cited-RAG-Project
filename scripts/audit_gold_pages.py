"""List pages in a gold document that share the gold answer's key terms but are not in gold_sources.

    uv run python scripts/audit_gold_pages.py --index-dir data/index_eval

Reads the stored docstore, so it needs no embedding model and no LLM request. Terms that sit on many pages of a
document count for little. A page is listed when it holds an identifier from the answer (such as 800-61r2), or
when its score reaches the best gold page or --min-score, whichever is lower. Span is the shortest stretch of
text holding every term found: a small span suggests the page states the answer, a large one that it only
mentions the terms. This is a review aid, not a verdict.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from citerag.eval.golden import load_golden
from citerag.schemas import GoldenItem

SCORED_TYPES = {"answerable", "multi_chunk"}
STOPWORDS = set(
    "a an and are as at be being both by for from has in into is it its no not of on or rather than that the "
    "their them then there they this to used using was were which with".split()
)
TOKEN = re.compile(r"[a-z0-9]+(?:[-.:/][a-z0-9]+)*")
SHOWN = 110  # characters of page text in the table


@dataclass
class Row:
    item_id: str
    doc: str
    page: int
    score: float
    best_gold: float
    span: int
    found: list[str]
    missing: list[str]
    text: str


def normalise(text: str) -> tuple[str, str]:
    """Two spellings of a page, because a word split at a line end may or may not keep its hyphen."""
    flat = re.sub(r"\s+", " ", text).lower()
    return flat.replace("- ", "-"), flat.replace("- ", "")


def key_terms(answer: str) -> list[str]:
    words = TOKEN.findall(answer.lower())
    return list(dict.fromkeys(w for w in words if w not in STOPWORDS and (len(w) > 1 or w.isdigit())))


def is_identifier(term: str) -> bool:
    return len(term) >= 4 and any(c.isdigit() for c in term) and any(c.isalpha() for c in term)


def pattern(term: str) -> re.Pattern[str]:
    if not term.isalpha():  # identifiers match whole: RV.3 must not match RV.3.1
        return re.compile(rf"(?<![a-z0-9.-]){re.escape(term)}(?![.-]?[a-z0-9])")
    stem = re.sub(r"(ing|es|ed|s)$", "", term) if len(term) > 5 else term
    return re.compile(rf"(?<![a-z0-9]){re.escape(stem)}[a-z]*")


def load_pages(index_dir: Path) -> dict[str, dict[int, tuple[str, str]]]:
    store = json.loads((index_dir / "store" / "docstore.json").read_text(encoding="utf-8"))
    joined: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for entry in store["docstore/data"].values():
        data = entry["__data__"]
        page = data["metadata"].get("page")
        if page is not None:
            joined[data["metadata"]["doc_name"]][page].append(data["text"])
    return {
        doc: {page: normalise(" ".join(parts)) for page, parts in pages.items()}
        for doc, pages in joined.items()
    }


def weights(terms: list[str], pages: dict[int, tuple[str, str]]) -> dict[str, float]:
    """Rare terms in this document count more: log(1 + pages / pages holding the term)."""
    result = {}
    for term in terms:
        rx = pattern(term)
        held = sum(any(rx.search(s) for s in spellings) for spellings in pages.values())
        result[term] = math.log(1 + len(pages) / held) if held else 0.0
    return result


def score(terms: dict[str, float], spellings: tuple[str, str]) -> tuple[float, list[str]]:
    total = sum(terms.values())
    found = [t for t in terms if any(pattern(t).search(s) for s in spellings)]
    return (sum(terms[t] for t in found) / total if total else 0.0), found


def tightest_span(text: str, terms: list[str]) -> tuple[int, int]:
    """Start and end of the shortest stretch of text holding every term that occurs in it."""
    hits = sorted((m.start(), m.end(), t) for t in terms for m in pattern(t).finditer(text))
    wanted = {t for _, _, t in hits}
    best = (0, len(text))
    latest: dict[str, tuple[int, int]] = {}
    for start, end, term in hits:
        latest[term] = (start, end)
        if len(latest) == len(wanted):
            window = (min(s for s, _ in latest.values()), max(e for _, e in latest.values()))
            if window[1] - window[0] < best[1] - best[0]:
                best = window
    return best


def find_rows(items: list[GoldenItem], stored: dict, min_score: float) -> list[Row]:
    rows = []
    for item in items:
        terms = key_terms(item.gold_answer or "")
        gold: dict[str, set[int]] = defaultdict(set)
        for source in item.gold_sources:
            if source.page is not None:
                gold[source.doc].add(source.page)
        for doc, gold_pages in gold.items():
            pages = stored[doc]
            weighted = weights(terms, pages)
            best = max(score(weighted, pages[p])[0] for p in gold_pages if p in pages)
            for page, spellings in pages.items():
                value, found = score(weighted, spellings)
                strong = len(found) >= min(2, len(terms)) and value >= min(best, min_score)
                if page in gold_pages or not (strong or any(is_identifier(t) for t in found)):
                    continue
                text = spellings[0]
                start, end = tightest_span(text, found)
                missing = [t for t in terms if t not in found]
                rows.append(
                    Row(item.id, doc, page, value, best, end - start, found, missing, text[start:end][:SHOWN])
                )
    return sorted(rows, key=lambda r: (r.item_id, r.span))


def main(
    golden: Annotated[Path, typer.Option()] = Path("eval/golden/candidates.jsonl"),
    index_dir: Annotated[Path, typer.Option()] = Path("data/index_eval"),
    min_score: Annotated[float, typer.Option(help="Upper limit for the listing threshold")] = 0.6,
) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # PDF text has characters the Windows console code page lacks
    items = [item for item in load_golden(golden) if item.type in SCORED_TYPES]
    rows = find_rows(items, load_pages(index_dir), min_score)
    typer.echo("| item | doc | page | span | score | best gold | terms found | terms missing | text |")
    typer.echo("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cells = (r.item_id, r.doc, r.page, r.span, f"{r.score:.2f}", f"{r.best_gold:.2f}")
        cells += (" ".join(r.found), " ".join(r.missing), r.text.replace("|", "/"))
        typer.echo("| " + " | ".join(map(str, cells)) + " |")
    whole = [f"{i.id} {s.doc}" for i in items for s in i.gold_sources if s.page is None]
    typer.echo(f"\n{len(rows)} page(s) listed. Whole-document gold, no pages to check: {', '.join(whole)}")


if __name__ == "__main__":
    typer.run(main)
