"""Gateway-owned parsing of structured output from free OpenRouter models.

Why not LangChain's with_structured_output? Spike finding: on invalid JSON it RAISES a pydantic ValidationError and
the raw message (with its token usage) is lost. Free models often wrap JSON in code fences, add prose around it or
emit <think> blocks, so the gateway calls the model itself and parses with this tolerant extractor.
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> str:
    """Return the most likely JSON object inside `text`: drop <think> blocks, unwrap a code fence, then cut from the
    first '{' to the last '}'. Returns the cleaned text unchanged if there is no object (the caller then fails
    validation and the gateway retries)."""
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        t = t[i : j + 1]
    return t


def parse_model(schema: type[T], text: str) -> T:
    """Validate model output against a Pydantic schema. Raises pydantic.ValidationError on failure."""
    return schema.model_validate_json(extract_json(text))


def schema_response_format(schema: type[BaseModel], strict: bool = False) -> dict:
    """OpenAI-style response_format for chat completions (verified accepted by langchain-openai's bind())."""
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": strict},
    }
