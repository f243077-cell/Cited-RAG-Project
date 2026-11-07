"""Query pipeline (ARCHITECTURE section 8): plain async functions, no hidden agent loops.

At most four LLM calls per request: condense, rewrite, generate, regenerate (F-RET-5). Deep mode replaces generate
with a crew of at least three requests (section 12).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np

from citerag.analytics.logs import LogStore, RequestRow
from citerag.cache.base import CacheBackend
from citerag.cache.semantic_cache import CacheHit
from citerag.config import Settings
from citerag.errors import (
    BudgetExceededError,
    CiteRAGError,
    LLMUnavailableError,
    StructuredOutputError,
    ValidationFailedError,
)
from citerag.llm.gateway import LLMGateway, StructuredResult
from citerag.multimodal.vision import analyze_image, validate_image
from citerag.prompts import load_prompt
from citerag.rag.condense import condense, rewrite
from citerag.rag.context import Chunk, build_context
from citerag.rag.generate import citation_hint, generate
from citerag.rag.verify import Passage, needs_retry, verify_citations
from citerag.schemas import (
    AgentStep,
    AgentTrace,
    CacheInfo,
    ChatRequest,
    ChatResponse,
    Citation,
    CitedAnswer,
    LatencyMs,
    LLMCitedAnswer,
    UsageInfo,
)
from citerag.tools.contracts import ToolResult

R = TypeVar("R")
NOT_FOUND = "I could not find this in the uploaded documents."


@dataclass(frozen=True)
class DeepAnswer:
    result: StructuredResult[LLMCitedAnswer]
    removed_claims: int  # a count, not the text: model text must not reach the warnings unchecked


class DeepRunner(Protocol):
    async def run(self, question: str, passages: list[Passage], request_id: str) -> DeepAnswer: ...


@dataclass(frozen=True)
class AgentFindings:
    chunks: list[Chunk]  # every tool hit, in the order found
    trace: AgentTrace
    requests: int
    input_tokens: int = 0
    output_tokens: int = 0
    shadow_usd: float = 0.0


class ToolRunnerLike(Protocol):
    def run(self, name: str, args: dict[str, Any]) -> ToolResult: ...


class AgentRunner(Protocol):
    async def run(
        self, question: str, run_tool: Callable[[str, dict[str, Any]], Awaitable[ToolResult]], request_id: str
    ) -> AgentFindings: ...


class Retriever(Protocol):
    def has_documents(self, doc_ids: list[str] | None = None) -> bool: ...

    def retrieve(self, query: str, doc_ids: list[str] | None = None) -> list[Chunk]: ...

    def embed_query(self, query: str) -> list[float]: ...

    def top_cosine(self, query_embedding: list[float], doc_ids: list[str] | None = None) -> float: ...


@dataclass
class Deps:
    cfg: Settings
    llm: LLMGateway
    retriever: Retriever
    cache: CacheBackend
    log: LogStore
    namespace: Callable[[], str]
    index_lock: threading.Lock = field(default_factory=threading.Lock)  # uploads and searches take turns
    observe: Callable[[list[Passage], list[Chunk]], None] | None = None  # eval: what the generator saw
    deep: DeepRunner | None = None  # Deep mode, built only where a crew may run
    agent: AgentRunner | None = None  # Agent mode (Phase 3)
    tool_library: Callable[[list[str] | None], ToolRunnerLike] | None = None  # tools scoped to a doc filter


class _Trace:
    """What one request used, for the response and the request log."""

    def __init__(self) -> None:
        self.request_id = uuid.uuid4().hex
        self.ts = time.time()
        self.started = time.perf_counter()
        self.latency: dict[str, int] = {}
        self.usage = UsageInfo()
        self.models: dict[str, str] = {}
        self.retries = 0
        self.standalone: str | None = None
        self.top_similarity: float | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        began = time.perf_counter()
        try:
            yield
        finally:
            spent = int((time.perf_counter() - began) * 1000)
            self.latency[name] = self.latency.get(name, 0) + spent

    def add(self, name: str, result: StructuredResult) -> StructuredResult:
        self.usage.input_tokens += result.input_tokens
        self.usage.output_tokens += result.output_tokens
        self.usage.llm_requests += result.requests
        self.usage.est_cost_usd += result.shadow_usd
        self.models[name] = result.model
        return result

    def latency_ms(self) -> LatencyMs:
        total = int((time.perf_counter() - self.started) * 1000)
        return LatencyMs(**self.latency, total=total)


async def _indexed(d: Deps, call: Callable[..., R], *args: object) -> R:
    """Index work blocks, so it leaves the event loop; the lock keeps it clear of an upload."""

    def locked() -> R:
        with d.index_lock:
            return call(*args)

    return await asyncio.to_thread(locked)


def _reject_unavailable(req: ChatRequest, d: Deps) -> None:
    if req.mode == "deep" and d.deep is None:
        raise ValidationFailedError("Deep mode is not available here")
    if req.mode == "agent":
        if d.agent is None or d.tool_library is None or not d.cfg.agent_enabled:
            raise ValidationFailedError("Agent mode is not available here")
        if req.image_b64 is not None:
            raise ValidationFailedError(
                "Agent mode does not take images yet: use standard mode for image questions"
            )


def assemble(llm_answer: LLMCitedAnswer, citations: list[Citation]) -> tuple[CitedAnswer, list[str]]:
    """F-ANS-3 to F-ANS-5: a refusal carries no citations; any failed check lowers confidence to low."""
    if not llm_answer.answerable:
        return (
            CitedAnswer(answer=llm_answer.answer, answerable=False, confidence=llm_answer.confidence),
            [],
        )
    unverified = needs_retry(True, llm_answer.answer, citations)
    doubtful = unverified or any(not c.valid for c in citations)
    answer = CitedAnswer(
        answer=llm_answer.answer,
        answerable=True,
        confidence="low" if doubtful else llm_answer.confidence,
        citations=citations,
    )
    return answer, ["Some claims could not be verified against the documents."] if doubtful else []


def _storable(answer: CitedAnswer, problems: list[str]) -> bool:
    """Rule 10: only fully verified answers. A dropped citation or an unbacked [n] leaves every listed citation
    valid, so the problems found during verification count too."""
    verified = bool(answer.citations) and all(c.valid for c in answer.citations)
    return answer.answerable and verified and not problems


def _refusal(req: ChatRequest, t: _Trace, text: str, cache: CacheInfo | None = None) -> ChatResponse:
    t.usage.latency_ms = t.latency_ms()
    return ChatResponse(
        request_id=t.request_id,
        session_id=req.session_id,
        standalone_query=t.standalone or req.message,
        answer=CitedAnswer(answer=text, answerable=False, confidence="high"),
        cache=cache or CacheInfo(enabled=False),
        usage=t.usage,
    )


def _from_cache(req: ChatRequest, t: _Trace, hit: CacheHit) -> ChatResponse:
    stored = ChatResponse.model_validate_json(hit.response_json)
    t.usage.latency_ms = t.latency_ms()
    return stored.model_copy(
        update={
            "request_id": t.request_id,
            "session_id": req.session_id,
            "standalone_query": t.standalone,
            "usage": t.usage,
            "cache": CacheInfo(
                enabled=True,
                hit=True,
                similarity=hit.similarity,
                matched_query=hit.query,
                saved_usd=max(hit.cost_usd - t.usage.est_cost_usd, 0.0),
                saved_requests=max(stored.usage.llm_requests - t.usage.llm_requests, 0),
            ),
        }
    )


async def _answer(req: ChatRequest, d: Deps, t: _Trace) -> ChatResponse:
    if not await _indexed(d, d.retriever.has_documents, req.doc_ids):  # before any LLM call (F-ANS-6)
        none = "Upload documents first." if req.doc_ids is None else "The selected documents have no text."
        return _refusal(req, t, none)

    t.standalone = req.message
    history = d.log.last_turns(req.session_id, d.cfg.history_turns)
    picture = (req.image_b64, req.image_media_type) if req.image_b64 else None
    analysis = None
    if picture:
        with t.stage("condense"):  # LatencyMs has no vision stage; it counts as preparing the question
            seen = t.add("vision", await analyze_image(d.llm, *picture, t.request_id))
        analysis = seen.parsed
    if history or analysis:
        with t.stage("condense"):
            call = condense(d.llm, history, req.message, t.request_id, analysis)
            result = t.add("condense", await call)
        t.standalone = result.parsed.query.strip() or req.message
    question = t.standalone
    if req.mode == "agent":
        return await _answer_agent(req, d, t, question)
    search_query = f"{question} {analysis.search_query}" if analysis else question  # F-MM-1

    with t.stage("embed"):
        embedding = await _indexed(d, d.retriever.embed_query, search_query)

    use_cache = (
        req.use_cache
        and d.cfg.cache_enabled
        and d.cache.stats().enabled
        and not picture  # F-MM-6
        and req.doc_ids is None  # a stored answer may cite documents outside the filter
        and req.mode == "standard"  # F-CAC-4: Deep mode neither reads nor writes
    )
    cache_info = CacheInfo(enabled=use_cache)
    if use_cache:
        vector, namespace = np.asarray(embedding), d.namespace()
        with t.stage("cache"):
            hit = d.cache.lookup(question, vector, namespace)
            if hit is None:  # the dashboard compares this with the threshold (chart 6)
                cache_info.similarity = d.cache.best_similarity(vector, namespace)
        if hit is not None:
            return _from_cache(req, t, hit)

    with t.stage("retrieve"):
        t.top_similarity = await _indexed(d, d.retriever.top_cosine, embedding, req.doc_ids)
    if t.top_similarity < d.cfg.min_top_similarity:  # corrective step (a), once
        t.retries += 1
        with t.stage("condense"):
            result = t.add("rewrite", await rewrite(d.llm, question, t.request_id))
        search_query = result.parsed.query.strip() or question
        with t.stage("retrieve"):
            embedding = await _indexed(d, d.retriever.embed_query, search_query)
            t.top_similarity = await _indexed(d, d.retriever.top_cosine, embedding, req.doc_ids)
        if t.top_similarity < d.cfg.min_top_similarity:
            return _refusal(req, t, NOT_FOUND, cache_info)

    with t.stage("retrieve"):
        chunks = await _indexed(d, d.retriever.retrieve, search_query, req.doc_ids)
    passages, warnings = build_context(chunks, d.cfg.context_token_budget)
    if not passages:
        return _refusal(req, t, NOT_FOUND, cache_info)
    if d.observe is not None:
        d.observe(passages, chunks)
    answer, problems = await _verified_answer(req, d, t, question, passages, picture, warnings)
    t.usage.latency_ms = t.latency_ms()
    response = ChatResponse(
        request_id=t.request_id,
        session_id=req.session_id,
        standalone_query=question,
        answer=answer,
        cache=cache_info,
        usage=t.usage,
        warnings=[*warnings, *problems],
    )
    if use_cache and _storable(answer, problems):
        d.cache.insert(
            question,
            np.asarray(embedding),
            response.model_dump_json(),
            d.namespace(),
            cost_usd=t.usage.est_cost_usd,
        )
    return response


async def _verified_answer(
    req: ChatRequest,
    d: Deps,
    t: _Trace,
    question: str,
    passages: list[Passage],
    picture: tuple[str, str] | None,
    warnings: list[str],
) -> tuple[CitedAnswer, list[str]]:
    """Generate, verify, and retry once on failed citations. Returns the answer and the verification problems;
    context warnings are appended to `warnings` in place."""
    by_marker = {p.marker: p for p in passages}
    with t.stage("generate"):
        if req.mode == "deep":
            deep = await d.deep.run(question, passages, t.request_id)
            first = t.add("crew", deep.result)
            if deep.removed_claims:
                warnings.append(f"The Fact Checker removed {deep.removed_claims} unsupported claim(s).")
        else:
            first = t.add("generate", await generate(d.llm, passages, question, t.request_id, image=picture))
    llm_answer = first.parsed
    with t.stage("verify"):
        citations, citation_warnings = verify_citations(llm_answer.citations, by_marker)

    if needs_retry(llm_answer.answerable, llm_answer.answer, citations):  # corrective step (b), once
        t.retries += 1
        hint = citation_hint(citations)
        try:
            with t.stage("generate"):
                call = generate(d.llm, passages, question, t.request_id, hint, picture)
                second = t.add("generate", await call)
        except (BudgetExceededError, LLMUnavailableError, StructuredOutputError) as exc:
            warnings.append(f"The citation retry failed ({exc.code}).")
        else:
            llm_answer = second.parsed
            with t.stage("verify"):
                citations, citation_warnings = verify_citations(llm_answer.citations, by_marker)

    answer, doubts = assemble(llm_answer, citations)
    return answer, [*citation_warnings, *doubts]


async def _answer_agent(req: ChatRequest, d: Deps, t: _Trace, question: str) -> ChatResponse:
    """Agent mode: the graph gathers evidence, then the standard generate-and-verify path answers (F-GRF-2).
    The loop's time goes under `retrieve`, synthesis under `generate`; the trace has the per-step split."""
    d.llm.check_room("strong", d.cfg.agent_max_steps + 2)  # F-GRF-5: room for the worst case before any call
    library = d.tool_library(req.doc_ids)

    async def run_tool(name: str, args: dict[str, Any]) -> ToolResult:
        return await _indexed(d, library.run, name, args)

    try:
        with t.stage("retrieve"):
            findings = await asyncio.wait_for(
                d.agent.run(question, run_tool, t.request_id), d.cfg.agent_timeout_s
            )
    except TimeoutError as exc:
        raise LLMUnavailableError(f"the agent ran past its {d.cfg.agent_timeout_s:.0f} s limit") from exc
    t.usage.llm_requests += findings.requests
    t.usage.input_tokens += findings.input_tokens
    t.usage.output_tokens += findings.output_tokens
    t.usage.est_cost_usd += findings.shadow_usd
    trace = findings.trace
    passages, warnings = build_context(
        findings.chunks, d.cfg.context_token_budget, max_passages=d.cfg.agent_context_max_passages
    )
    if not passages:  # F-GRF-8: nothing above the gate, so nothing to synthesize
        refused = _refusal(req, t, NOT_FOUND, CacheInfo(enabled=False))
        return refused.model_copy(update={"trace": trace.model_copy(update={"stop_reason": "no_evidence"})})
    if d.observe is not None:
        used = {p.chunk_id for p in passages}
        d.observe(passages, [c for c in findings.chunks if c.chunk_id in used])
    before, began = t.usage.llm_requests, time.perf_counter()
    answer, problems = await _verified_answer(req, d, t, question, passages, None, warnings)
    spent = t.usage.llm_requests - before
    synthesis = AgentStep(
        node="synthesize",
        hits=[f"{p.doc_name} p{p.page}" if p.page is not None else p.doc_name for p in passages],
        note=f"{len(passages)} passage(s), {len(answer.citations)} citation(s)",
        llm_requests=spent,
        latency_ms=int((time.perf_counter() - began) * 1000),
    )
    trace = trace.model_copy(
        update={"steps": [*trace.steps, synthesis], "llm_requests": trace.llm_requests + spent}
    )
    t.usage.latency_ms = t.latency_ms()
    return ChatResponse(
        request_id=t.request_id,
        session_id=req.session_id,
        standalone_query=question,
        answer=answer,
        cache=CacheInfo(enabled=False),  # ADR-18: Agent mode neither reads nor writes the cache
        usage=t.usage,
        warnings=[*warnings, *problems],
        trace=trace,
    )


def _request_row(
    req: ChatRequest, t: _Trace, response: ChatResponse | None, error_code: str | None
) -> RequestRow:
    latency = t.latency_ms() if response is None else response.usage.latency_ms
    row = RequestRow(
        request_id=t.request_id,
        ts=t.ts,
        session_id=req.session_id,
        mode=req.mode,
        message=req.message,
        standalone_query=t.standalone,
        has_image=req.image_b64 is not None,
        top_similarity=t.top_similarity,
        n_retries=t.retries,
        input_tokens=t.usage.input_tokens,
        output_tokens=t.usage.output_tokens,
        llm_requests=t.usage.llm_requests,
        est_cost_usd=t.usage.est_cost_usd,
        lat_total=latency.total,
        lat_condense=latency.condense,
        lat_embed=latency.embed,
        lat_cache=latency.cache,
        lat_retrieve=latency.retrieve,
        lat_rerank=latency.rerank,
        lat_generate=latency.generate,
        lat_verify=latency.verify,
        prompt_version=load_prompt(
            "deep_writer" if req.mode == "deep" else "answer"
        ).version,  # agent: answer
        models_json=json.dumps(t.models),
        error_code=error_code,
    )
    if response is None:
        return row
    citations = response.answer.citations
    return row.model_copy(
        update={
            "cache_enabled": response.cache.enabled,
            "cache_hit": response.cache.hit,
            "cache_similarity": response.cache.similarity,
            "answerable": response.answer.answerable,
            "n_citations": len(citations),
            "n_valid_citations": sum(c.valid for c in citations),
            "confidence": response.answer.confidence,
            "saved_usd": response.cache.saved_usd,
            "saved_requests": response.cache.saved_requests,
        }
    )


async def answer_question(req: ChatRequest, d: Deps) -> ChatResponse:
    _reject_unavailable(req, d)
    if req.image_b64 is not None:
        validate_image(req.image_b64, req.image_media_type)
    t = _Trace()
    try:
        response = await _answer(req, d, t)
    except CiteRAGError as exc:
        exc.request_id = t.request_id
        t.usage.llm_requests = d.log.llm_calls_for(t.request_id)  # calls that failed return no result to add
        d.log.write_request(_request_row(req, t, None, exc.code))
        raise
    d.log.add_turn(req.session_id, req.message, response.answer.answer, t.request_id)
    d.log.write_request(_request_row(req, t, response, None))
    return response
