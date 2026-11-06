"""Context engineering: pick passages within a token budget and render them as untrusted data."""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

from rapidfuzz import fuzz

from citerag.rag.verify import Passage


@dataclass(frozen=True)
class Chunk:
    """App-level chunk. LlamaIndex nodes are converted to this at the retrieval boundary."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page: int | None
    text: str
    score: float = 0.0
    kind: str = "text"  # "image" for VLM captions
    image_path: str | None = None


def estimate_tokens(text: str) -> int:
    """Rough estimate (characters / 4). Real usage comes back from the API and is logged."""
    return max(1, len(text) // 4)


def build_context(
    chunks: list[Chunk],
    budget_tokens: int,
    max_passages: int = 8,
    dup_ratio: float = 95.0,
    tag_overhead: int = 25,
) -> tuple[list[Passage], list[str]]:
    """`chunks` must be ordered best-first. Keeps rank order, skips near-duplicates, stops at the budget.
    Lowest-ranked chunks are dropped first and a chunk is never cut in half. The best chunk is always kept."""
    kept: list[Chunk] = []
    warnings: list[str] = []
    used = 0
    for c in chunks:
        if len(kept) >= max_passages:
            break
        if any(k.doc_id == c.doc_id and fuzz.partial_ratio(c.text, k.text) >= dup_ratio for k in kept):
            continue
        cost = estimate_tokens(c.text) + tag_overhead
        if kept and used + cost > budget_tokens:
            break
        if not kept and cost > budget_tokens:
            warnings.append("best passage alone exceeds the context budget; kept it anyway")
        kept.append(c)
        used += cost
    passages = [
        Passage(
            marker=i,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            doc_name=c.doc_name,
            page=c.page,
            text=c.text,
            kind=c.kind,
            image_path=c.image_path,
        )
        for i, c in enumerate(kept, start=1)
    ]
    return passages, warnings


def render_passages(passages: list[Passage]) -> str:
    """XML block for the prompt. Escaping stops a document from closing the tag and injecting structure."""
    parts = ["<documents>"]
    for p in passages:
        attrs = f"id={quoteattr(str(p.marker))} source={quoteattr(p.doc_name)} kind={quoteattr(p.kind)}"
        if p.page is not None:
            attrs += f" page={quoteattr(str(p.page))}"
        parts.append(f"<passage {attrs}>\n{escape(p.text)}\n</passage>")
    parts.append("</documents>")
    return "\n".join(parts)
