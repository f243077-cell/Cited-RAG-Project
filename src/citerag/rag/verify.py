"""Citation verification: never trust a quote the LLM claims to have copied."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from citerag.schemas import Citation, LLMCitation


@dataclass(frozen=True)
class Passage:
    marker: int  # the [n] shown to the LLM
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int | None
    text: str
    kind: str = "text"  # "image" for VLM captions
    image_path: str | None = None


def _norm(s: str) -> str:
    s = re.sub(r"[*_`#>\[\]]", "", s.lower())  # drop markdown noise the LLM may add or remove
    return re.sub(r"\s+", " ", s).strip()


def verify_citations(
    llm_citations: list[LLMCitation], passages: dict[int, Passage], min_score: float = 90.0
) -> tuple[list[Citation], list[str]]:
    """Returns (citations with `valid` set, warnings). Unknown markers are dropped with a warning."""
    out: list[Citation] = []
    warnings: list[str] = []
    for c in llm_citations:
        p = passages.get(c.marker)
        if p is None:
            warnings.append(f"citation [{c.marker}] refers to a passage that was not provided")
            continue
        score = float(fuzz.partial_ratio(_norm(c.quote), _norm(p.text)))
        out.append(
            Citation(
                marker=c.marker,
                chunk_id=p.chunk_id,
                doc_id=p.doc_id,
                doc_name=p.doc_name,
                page=p.page,
                kind=p.kind,
                quote=c.quote,
                valid=score >= min_score,
                match_score=round(score, 1),
                image_path=p.image_path,
            )
        )
    return out, warnings


def markers_in_text(answer: str) -> set[int]:
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer)}


def needs_retry(answerable: bool, answer: str, citations: list[Citation]) -> bool:
    """True when an 'answerable' answer has no verified citation, or uses [n] markers nobody backs up."""
    if not answerable:
        return False
    valid = {c.marker for c in citations if c.valid}
    return not valid or not markers_in_text(answer) <= valid
