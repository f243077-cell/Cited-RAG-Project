"""Small query calls on the fast model: condense a follow-up, rewrite a weak search."""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from citerag.llm.gateway import LLMGateway, StructuredResult
from citerag.prompts import load_prompt
from citerag.schemas import ImageAnalysis, Message, StandaloneQuery


def render_history(history: list[Message], message: str, image: ImageAnalysis | None = None) -> str:
    """Past answers and image text can carry document or attacker text, so both are escaped."""
    turns = [f"<turn role={quoteattr(m.role)}>{escape(m.content)}</turn>" for m in history]
    lines = ["<history>", *turns, "</history>"]
    if image is not None:
        seen = f"{image.description}\nVisible text: {image.visible_text}"
        lines.append(f"<image>{escape(seen)}</image>")
    return "\n".join([*lines, f"<message>{escape(message)}</message>"])


async def condense(
    llm: LLMGateway,
    history: list[Message],
    message: str,
    request_id: str,
    image: ImageAnalysis | None = None,
) -> StructuredResult[StandaloneQuery]:
    return await llm.structured_call(
        role="fast",
        schema=StandaloneQuery,
        system=load_prompt("condense").text,
        user=render_history(history, message, image),
        call_name="condense",
        request_id=request_id,
    )


async def rewrite(llm: LLMGateway, query: str, request_id: str) -> StructuredResult[StandaloneQuery]:
    return await llm.structured_call(
        role="fast",
        schema=StandaloneQuery,
        system=load_prompt("rewrite").text,
        user=f"<question>{escape(query)}</question>",
        call_name="rewrite",
        request_id=request_id,
    )
