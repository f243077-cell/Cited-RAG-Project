"""Judge call (F-EVA-3): a structured LLM verdict, given the same passages the generator saw."""

from __future__ import annotations

from xml.sax.saxutils import escape

from citerag.llm.gateway import LLMGateway, StructuredResult
from citerag.prompts import load_prompt
from citerag.schemas import ItemRecord, JudgeVerdict


def render_judge_input(record: ItemRecord) -> str:
    answer = record.answer.answer if record.answer else ""
    return "\n".join(
        [
            f"<question>{escape(record.question)}</question>",
            f"<reference>{escape(record.gold_answer or 'none given')}</reference>",
            record.context,  # already escaped by render_passages
            f"<answer>{escape(answer)}</answer>",
        ]
    )


async def judge_answer(
    llm: LLMGateway, record: ItemRecord, request_id: str
) -> StructuredResult[JudgeVerdict]:
    return await llm.structured_call(
        role="judge",
        schema=JudgeVerdict,
        system=load_prompt("judge").text,
        user=render_judge_input(record),
        call_name="judge",
        request_id=request_id,
    )
