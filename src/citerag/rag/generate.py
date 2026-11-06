"""Answer generation: passages stay inside the escaped <documents> block, rules stay in the system prompt."""

from __future__ import annotations

from xml.sax.saxutils import escape

from citerag.llm.gateway import LLMGateway, StructuredResult
from citerag.prompts import load_prompt
from citerag.rag.context import render_passages
from citerag.rag.verify import Passage
from citerag.schemas import Citation, LLMCitedAnswer


def render_answer_input(passages: list[Passage], question: str, hint: str | None = None) -> str:
    parts = [render_passages(passages), f"<question>{escape(question)}</question>"]
    if hint:
        parts.append(f"<correction>{escape(hint)}</correction>")
    return "\n".join(parts)


def citation_hint(citations: list[Citation]) -> str:
    """Names markers only, so no model or document text comes back into the instructions."""
    bad = sorted({c.marker for c in citations if not c.valid})
    problem = (
        f"citations {bad} did not match their passages" if bad else "your answer lacked backed citations"
    )
    return (
        f"Your previous reply failed verification: {problem}. Answer again. Every [n] in the answer needs an entry "
        "in `citations` with a quote copied word for word from passage n."
    )


async def generate(
    llm: LLMGateway, passages: list[Passage], question: str, request_id: str, hint: str | None = None
) -> StructuredResult[LLMCitedAnswer]:
    return await llm.structured_call(
        role="strong",
        schema=LLMCitedAnswer,
        system=load_prompt("answer").text,
        user=render_answer_input(passages, question, hint),
        call_name="generate",
        request_id=request_id,
    )
