"""Pydantic outputs of the CrewAI crews. Every crew task must return one of these (output_pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from citerag.schemas import LLMCitedAnswer


# ---- Deep mode crew -----------------------------------------------------------------
class EvidenceItem(BaseModel):
    marker: int = Field(ge=1)
    fact: str
    quote: str = Field(max_length=300, description="Exact words from the passage")


class EvidenceTable(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    revised: LLMCitedAnswer = Field(
        description="The answer with unsupported claims removed; same citation rules"
    )


# ---- Evaluation crews ---------------------------------------------------------------
class QACandidate(BaseModel):
    question: str
    gold_answer: str
    doc: str
    page: int | None = None
    critic_ok: bool = Field(
        description="Critic confirms: answerable from that passage alone, unambiguous, not trivial"
    )
    critic_note: str = ""


class QACandidateList(BaseModel):
    candidates: list[QACandidate] = Field(default_factory=list)


FailureCause = Literal[
    "retrieval_miss",
    "chunking",
    "prompt",
    "hallucination",
    "citation_error",
    "refusal_error",
    "injection",
    "other",
]


class FailureCluster(BaseModel):
    cause: FailureCause
    item_ids: list[str]
    explanation: str
    suggested_fix: str


class FailureAnalysis(BaseModel):
    clusters: list[FailureCluster] = Field(default_factory=list)
