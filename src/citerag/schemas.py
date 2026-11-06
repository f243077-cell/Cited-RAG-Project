"""Pydantic contracts shared by API, pipeline, cache, eval and UI."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Confidence = Literal["high", "medium", "low"]
ImageType = Literal["screenshot", "chart", "diagram", "photo", "document_scan", "other"]


# ---- 1. What the LLM may produce: only things code cannot derive ------------------
class LLMCitation(BaseModel):
    marker: int = Field(ge=1, description="Number [n] of the context passage that supports the claim")
    quote: str = Field(min_length=3, max_length=300, description="Exact words copied from that passage")


class LLMCitedAnswer(BaseModel):
    answer: str = Field(description="Answer with inline [n] markers. If not answerable, say so briefly.")
    answerable: bool = Field(description="False when the passages do not contain the answer")
    citations: list[LLMCitation] = Field(default_factory=list)
    confidence: Confidence


# ---- 2. What the API returns: enriched and verified by code, never by the LLM -----
class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int | None = None
    kind: Literal["text", "image"] = "text"
    quote: str
    valid: bool  # quote found in the cited passage (see rag/verify.py)
    match_score: float  # 0-100 fuzzy match score
    image_path: str | None = None


class CitedAnswer(BaseModel):
    answer: str
    answerable: bool
    confidence: Confidence
    citations: list[Citation] = Field(default_factory=list)


class CacheInfo(BaseModel):
    enabled: bool
    hit: bool = False
    similarity: float | None = None
    matched_query: str | None = None
    saved_usd: float | None = None  # shadow cost avoided by this hit
    saved_requests: int | None = None  # LLM requests avoided by this hit


class LatencyMs(BaseModel):
    condense: int = 0
    embed: int = 0
    cache: int = 0
    retrieve: int = 0
    rerank: int = 0
    generate: int = 0
    verify: int = 0
    total: int = 0


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_requests: int = 0  # HTTP attempts to the LLM provider for this request (the free-tier quota unit)
    est_cost_usd: float = (
        0.0  # SHADOW cost: what the tokens would cost on the paid reference model (see pricing.yaml)
    )
    latency_ms: LatencyMs = Field(default_factory=LatencyMs)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)
    image_b64: str | None = Field(default=None, max_length=7_000_000)  # about 5 MB decoded
    image_media_type: Literal["image/png", "image/jpeg", "image/webp"] | None = None
    mode: Literal["standard", "deep"] = "standard"
    use_cache: bool = True
    doc_ids: list[str] | None = None  # restrict retrieval to these documents

    @model_validator(mode="after")
    def _image_fields_travel_together(self) -> ChatRequest:
        if (self.image_b64 is None) != (self.image_media_type is None):
            raise ValueError("image_b64 and image_media_type must be sent together")
        return self


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    standalone_query: str
    answer: CitedAnswer
    cache: CacheInfo
    usage: UsageInfo
    warnings: list[str] = Field(default_factory=list)


# ---- 2b. Documents and errors ---------------------------------------------------
class IngestReport(BaseModel):
    doc_id: str
    doc_name: str
    status: Literal["ingested", "unchanged", "replaced"]
    pages: int | None = None
    chunks: int = 0
    images_found: int = 0
    images_captioned: int = 0
    images_skipped: int = 0
    caption_requests: int = 0  # LLM requests spent on captions (the free-tier quota unit)
    caption_input_tokens: int = 0
    caption_output_tokens: int = 0
    caption_cost_usd: float = 0.0
    duration_ms: int = 0


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    pages: int | None = None
    chunks: int
    images: int = 0
    size_bytes: int
    ingested_at: datetime


ErrorCode = Literal[
    "invalid_file_type",
    "file_too_large",
    "budget_exceeded",
    "llm_unavailable",
    "structured_output_failed",
    "index_mismatch",
    "not_found",
    "unauthorized",
    "validation_error",
    "internal_error",
]


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---- 3. Small structured calls --------------------------------------------------
class StandaloneQuery(BaseModel):
    query: str = Field(description="The user's latest question rewritten so it needs no chat history")


class ImageAnalysis(BaseModel):  # user attaches an image to a question
    description: str
    visible_text: str = ""
    image_type: ImageType
    search_query: str = Field(description="Short query to find related passages in the documents")


class ImageCaption(BaseModel):  # image found inside a document at ingestion time
    caption: str = Field(max_length=600)
    visible_text: str = ""
    image_type: ImageType
    key_facts: list[str] = Field(default_factory=list, max_length=8)


# ---- 4. Evaluation ---------------------------------------------------------------
class JudgeVerdict(BaseModel):
    faithful: bool = Field(description="Every claim in the answer is supported by the given context")
    unsupported_claims: list[str] = Field(default_factory=list)
    correctness: Literal["correct", "partial", "incorrect"]
    reasoning: str = Field(max_length=500)


class GoldenSource(BaseModel):
    doc: str
    page: int | None = None  # None = any page of that document


class GoldenItem(BaseModel):
    id: str
    question: str
    type: Literal["answerable", "multi_chunk", "unanswerable", "injection", "image"]
    gold_answer: str | None = None
    gold_sources: list[GoldenSource] = Field(default_factory=list)
    image_path: str | None = None
    canary: str | None = None  # injection items: this string must NOT appear in the answer
    verified: bool = False  # a human checked question, answer and sources
