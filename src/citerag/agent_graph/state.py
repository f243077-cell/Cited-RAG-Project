"""Graph state. Plain Pydantic: nothing in it is a framework object, so it can leave the package as is."""

from __future__ import annotations

from pydantic import BaseModel, Field

from citerag.schemas import AgentStep, ToolCall
from citerag.tools.contracts import ToolHit


class AgentState(BaseModel):
    question: str
    request_id: str
    documents: list[str] = Field(default_factory=list)  # library names, listed by code for the act prompt
    sub_questions: list[str] = Field(default_factory=list)
    hits: dict[str, ToolHit] = Field(default_factory=dict)  # by chunk_id, in the order found
    pending: list[ToolCall] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    searched: list[str] = Field(default_factory=list)  # queries already run, shown to the act step
    steps_used: int = 0  # LLM calls of plan, act and reflect
    done: bool = False
    stop_reason: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    shadow_usd: float = 0.0
