"""Query pipeline (ARCHITECTURE section 8): plain async functions, no hidden agent loops.

At most four LLM calls per request: condense, rewrite, generate, regenerate (F-RET-5).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

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
from citerag.prompts import load_prompt
from citerag.rag.condense import condense, rewrite
from citerag.rag.context import Chunk, build_context
from citerag.rag.generate import citation_hint, generate
from citerag.rag.verify import needs_retry, verify_citations
from citerag.schemas import (
    CacheInfo,
    ChatRequest,
    ChatResponse,
    Citation,
    CitedAnswer,
    LatencyMs,
    LLMCitedAnswer,
    UsageInfo,
)

R = TypeVar("R")
NOT_FOUND = "I could not find this in the uploaded documents."


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


def _reject_unbuilt(req: ChatRequest) -> None:
    if req.image_b64 is not None:
        raise ValidationFailedError("Image questions arrive in milestone M5")
    if req.mode != "standard":
        raise ValidationFailedError("Deep mode arrives in milestone M7")


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
                saved_usd=hit.cost_usd,
                saved_requests=stored.usage.llm_requests,
            ),
        }
    )


async def _answer(req: ChatRequest, d: Deps, t: _Trace) -> ChatResponse:
    if not await _indexed(d, d.retriever.has_documents, req.doc_ids):  # before any LLM call (F-ANS-6)
        none = "Upload documents first." if req.doc_ids is None else "The selected documents have no text."
        return _refusal(req, t, none)

    t.standalone = req.message
    history = d.log.last_turns(req.session_id, d.cfg.history_turns)
    if history:
        with t.stage("condense"):
            result = t.add("condense", await condense(d.llm, history, req.message, t.request_id))
        t.standalone = result.parsed.query.strip() or req.message
    question = t.standalone

    with t.stage("embed"):
        embedding = await _indexed(d, d.retriever.embed_query, question)

    use_cache = req.use_cache and d.cfg.cache_enabled and d.cache.stats().enabled
    cache_info = CacheInfo(enabled=use_cache)
    if use_cache:
        with t.stage("cache"):
            hit = d.cache.lookup(question, np.asarray(embedding), d.namespace())
        if hit is not None:
            return _from_cache(req, t, hit)

    with t.stage("retrieve"):
        t.top_similarity = await _indexed(d, d.retriever.top_cosine, embedding, req.doc_ids)
    search_query = question
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
    by_marker = {p.marker: p for p in passages}

    with t.stage("generate"):
        first = t.add("generate", await generate(d.llm, passages, question, t.request_id))
    llm_answer = first.parsed
    with t.stage("verify"):
        citations, citation_warnings = verify_citations(llm_answer.citations, by_marker)

    if needs_retry(llm_answer.answerable, llm_answer.answer, citations):  # corrective step (b), once
        t.retries += 1
        hint = citation_hint(citations)
        try:
            with t.stage("generate"):
                second = t.add("generate", await generate(d.llm, passages, question, t.request_id, hint))
        except (BudgetExceededError, LLMUnavailableError, StructuredOutputError) as exc:
            warnings.append(f"The citation retry failed ({exc.code}).")
        else:
            llm_answer = second.parsed
            with t.stage("verify"):
                citations, citation_warnings = verify_citations(llm_answer.citations, by_marker)

    answer, doubts = assemble(llm_answer, citations)
    t.usage.latency_ms = t.latency_ms()
    response = ChatResponse(
        request_id=t.request_id,
        session_id=req.session_id,
        standalone_query=question,
        answer=answer,
        cache=cache_info,
        usage=t.usage,
        warnings=[*warnings, *citation_warnings, *doubts],
    )
    if use_cache and _storable(answer, [*citation_warnings, *doubts]):
        d.cache.insert(
            question,
            np.asarray(embedding),
            response.model_dump_json(),
            d.namespace(),
            cost_usd=t.usage.est_cost_usd,
        )
    return response


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
        prompt_version=load_prompt("answer").version,
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
    _reject_unbuilt(req)
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
