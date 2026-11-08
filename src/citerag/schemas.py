"""Pydantic contracts shared by API, pipeline, cache, eval and UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    mode: Literal["standard", "deep", "agent"] = "standard"
    use_cache: bool = True
    doc_ids: list[str] | None = None  # restrict retrieval to these documents

    @model_validator(mode="after")
    def _image_fields_travel_together(self) -> ChatRequest:
        if (self.image_b64 is None) != (self.image_media_type is None):
            raise ValueError("image_b64 and image_media_type must be sent together")
        return self


class AgentStep(BaseModel):
    node: Literal["plan", "act", "tools", "reflect", "synthesize", "supervise"]
    tool: str | None = None
    args: dict[str, Any] | None = None  # validated arguments only; a query is model text, shown as plain text
    hits: list[str] = Field(default_factory=list)  # "doc p3" references, never passage text
    note: str = ""  # written by code, not by the model
    llm_requests: int = 0
    latency_ms: int = 0


class AgentTrace(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    stop_reason: str = ""
    llm_requests: int = 0
    method: Literal["tools", "json"] = "tools"


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    standalone_query: str
    answer: CitedAnswer
    cache: CacheInfo
    usage: UsageInfo
    warnings: list[str] = Field(default_factory=list)
    trace: AgentTrace | None = None  # Agent mode only (Phase 3)


class Message(BaseModel):
    session_id: str
    ts: float  # Unix seconds
    role: Literal["user", "assistant"]
    content: str
    request_id: str | None = None


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
    # Sources with the same `part` are alternatives (any one satisfies the part). A multi_chunk item is fully
    # covered only when every distinct part is retrieved. Default 0 keeps single-part items unchanged.
    part: int = 0


class GoldenItem(BaseModel):
    id: str
    question: str
    type: Literal["answerable", "multi_chunk", "unanswerable", "injection", "image"]
    gold_answer: str | None = None
    gold_sources: list[GoldenSource] = Field(default_factory=list)
    image_path: str | None = None
    canary: str | None = None  # injection items: this string must NOT appear in the answer
    verified: bool = False  # a reviewer checked question, answer and sources
    reviewed_by: Literal["human", "claude"] | None = None  # "claude": AI-reviewed, and every report says so


class ItemRecord(BaseModel):
    """One golden item in a run, kept so a run can resume and verdicts can be added later."""

    id: str
    type: str
    question: str
    gold_answer: str | None = None
    gold_sources: list[GoldenSource] = Field(default_factory=list)
    canary: str | None = None
    status: Literal["answered", "error"]
    error_code: str | None = None
    answer: CitedAnswer | None = None
    context: str = ""  # the rendered passages: the judge sees what the generator saw (F-EVA-3)
    retrieved: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: UsageInfo | None = None
    verdict: JudgeVerdict | None = None
    judge_error: str | None = None
    judge_requests: int = 0
    trace: AgentTrace | None = None  # Agent mode: tool calls are checked for canaries too (E8)


class Rate(BaseModel):
    n: int
    of: int
    rate: float | None = None  # None when nothing was measured


class AnswerSummary(BaseModel):
    faithfulness: Rate
    correctness: Rate
    correct_or_partial: Rate
    citation_validity: Rate
    citation_precision: Rate
    correct_refusal: Rate
    false_refusal: Rate
    injection_leaks: int
    injection_items: int
    pending_judge: int
    errors: int
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    mean_llm_requests: float | None = None
    mean_shadow_usd: float | None = None


class RetrievalScores(BaseModel):
    n_scored: int
    ks: list[int]
    hits: dict[int, int]
    hit_rate: dict[int, float]
    mrr_k: int
    mrr: float
    n_multi_chunk: int = 0
    full_coverage: dict[int, int] = Field(
        default_factory=dict
    )  # multi_chunk items with every part in the top k


class RunSummary(BaseModel):
    run_id: str
    status: Literal["offline", "complete", "paused", "stopped"]
    reason: str | None = None
    experiment: str | None = None
    golden: str
    n_items: int
    n_done: int = 0
    unverified_items: int
    ai_reviewed_items: int = 0  # verified by Claude, not a human: the numbers stay provisional
    requests_used: int = 0
    before_rerank: RetrievalScores | None = None
    after_rerank: RetrievalScores | None = None
    answerable_cosines: list[float] = Field(default_factory=list)
    unanswerable_cosines: list[float] = Field(default_factory=list)
    answers: AnswerSummary | None = None

    def headline(self) -> dict[str, float | None]:
        """Shares from 0 to 1, for compare tables and charts."""
        found: dict[str, float | None] = {}
        for prefix, scores in (("", self.before_rerank), ("rerank ", self.after_rerank)):
            if scores is not None and scores.n_scored:  # no scored item is no measurement, not 0%
                found |= {f"{prefix}hit@{k}": scores.hit_rate[k] for k in scores.ks}
                found[f"{prefix}MRR@{scores.mrr_k}"] = scores.mrr
                for (
                    k,
                    n,
                ) in scores.full_coverage.items():  # the headline for multi_chunk: hit@k needs one part only
                    found[f"{prefix}multi_chunk full_coverage@{k}"] = n / scores.n_multi_chunk
        if self.answers is not None:
            for name in AnswerSummary.model_fields:
                value = getattr(self.answers, name)
                if isinstance(value, Rate):
                    found[name] = value.rate
        return found


class RunDetail(BaseModel):
    summary: RunSummary
    items: list[ItemRecord]


# ---- 5. Agent mode (Phase 3): what the planner, the act step and reflection may produce ----------
AgentToolName = Literal["list_documents", "search_documents", "read_page"]


class Plan(BaseModel):
    sub_questions: list[str] = Field(
        min_length=1, max_length=3, description="1 to 3 questions that together answer it"
    )


class ToolCall(BaseModel):
    name: AgentToolName
    args: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    calls: list[ToolCall] = Field(default_factory=list, max_length=3)
    done: bool = Field(default=False, description="True when no more searching is needed")


class Reflection(BaseModel):
    enough: list[bool] = Field(description="One per sub-question: does the evidence answer it?")
    next_queries: list[str] = Field(
        default_factory=list, max_length=3, description="What to search next, if anything"
    )


class Route(BaseModel):  # the M12 supervisor's handoff
    action: Literal["research", "answer"]
    task: str = Field(default="", max_length=500, description="For research: one short, specific task")
