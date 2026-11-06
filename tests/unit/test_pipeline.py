"""The answer pipeline with FakeGateway and a stub retriever: no index, no network."""

import asyncio
import sqlite3

import pytest

from citerag.analytics.logs import LogStore
from citerag.cache.base import CacheStats, NoOpCache
from citerag.cache.semantic_cache import CacheHit
from citerag.errors import BudgetExceededError, StructuredOutputError, ValidationFailedError
from citerag.llm.fake import FakeGateway
from citerag.rag.context import Chunk
from citerag.rag.pipeline import Deps, answer_question
from citerag.schemas import (
    CacheInfo,
    ChatRequest,
    ChatResponse,
    CitedAnswer,
    LLMCitation,
    LLMCitedAnswer,
    StandaloneQuery,
    UsageInfo,
)

TEXT = "Items can be returned within thirty days of delivery."
CHUNK = Chunk("doc-0000", "doc", "policy.pdf", 3, TEXT)


class StubRetriever:
    def __init__(self, cosines=(0.8,), documents=True):
        self.cosines, self.documents = list(cosines), documents
        self.queries: list[str] = []
        self.doc_ids: list = []

    def has_documents(self, doc_ids=None):
        self.doc_ids.append(doc_ids)
        return self.documents

    def embed_query(self, query):
        return [1.0, 0.0]

    def top_cosine(self, query_embedding, doc_ids=None):
        return self.cosines.pop(0)

    def retrieve(self, query, doc_ids=None):
        self.queries.append(query)
        return [CHUNK]


class StubCache(NoOpCache):
    def __init__(self, hit=None):
        self.hit, self.inserted, self.looked_up = hit, [], []

    def stats(self):
        return CacheStats(backend="stub", enabled=True)

    def lookup(self, q_text, q_emb, namespace, mode="standard", now=None):
        self.looked_up.append(q_text)
        return self.hit

    def insert(self, q_text, q_emb, response_json, namespace, mode="standard", cost_usd=0.0, now=None):
        self.inserted.append(q_text)

    def best_similarity(self, q_emb, namespace, mode="standard", now=None):
        return 0.61


def good(quote=TEXT, answerable=True, marker=1):
    return LLMCitedAnswer(
        answer="Thirty days [1].",
        answerable=answerable,
        citations=[LLMCitation(marker=marker, quote=quote)],
        confidence="high",
    )


@pytest.fixture
def world(settings):
    llm = FakeGateway()

    def make(retriever=None, cache=None):
        return Deps(
            cfg=settings,
            llm=llm,
            retriever=retriever or StubRetriever(),
            cache=cache or NoOpCache(),
            log=LogStore(settings.logs_db_path),
            namespace=lambda: "ns",
        )

    return llm, make


def ask(deps, message="How long do I have to return an item?", **kw):
    return asyncio.run(answer_question(ChatRequest(session_id="s", message=message, **kw), deps))


def request_rows(deps, settings):
    db = sqlite3.connect(settings.logs_db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM requests")]
    db.close()
    return rows


def test_happy_path_returns_a_verified_citation_from_code_not_the_model(world, settings):
    llm, make = world
    llm.queue("generate", good())
    response = ask(make())
    citation = response.answer.citations[0]
    assert response.answer.answerable and response.answer.confidence == "high"
    assert citation.valid and (citation.doc_name, citation.page) == ("policy.pdf", 3)
    assert [c.call_name for c in llm.calls] == ["generate"]  # no history, so no condense
    assert response.warnings == [] and response.usage.llm_requests == 1


def test_passages_stay_in_the_user_message_and_rules_in_the_system_prompt(world):
    llm, make = world
    llm.queue("generate", good())
    ask(make(), message="a <b> question")
    call = llm.calls[0]
    assert TEXT in call.user and TEXT not in call.system and call.role == "strong"
    assert "<question>a &lt;b&gt; question</question>" in call.user


def test_empty_index_refuses_with_no_llm_call_even_with_history(world):
    llm, make = world
    deps = make(StubRetriever(documents=False))
    deps.log.add_turn("s", "earlier", "answer", "r0")
    response = ask(deps)
    assert not response.answer.answerable and response.answer.answer == "Upload documents first."
    assert llm.calls == [] and response.answer.citations == []


def test_follow_up_is_condensed_and_search_uses_the_standalone_question(world):
    llm, make = world
    retriever = StubRetriever()
    deps = make(retriever)
    deps.log.add_turn("s", "What is the refund policy?", "Thirty days [1].", "r0")
    llm.queue("condense", StandaloneQuery(query="What is the shipping policy?"))
    llm.queue("generate", good())
    response = ask(deps, message="And what about shipping?")
    assert retriever.queries == ["What is the shipping policy?"]
    assert response.standalone_query == "What is the shipping policy?"
    assert [c.role for c in llm.calls] == ["fast", "strong"]
    assert "<message>And what about shipping?</message>" in llm.calls[0].user
    assert "What is the refund policy?" in llm.calls[0].user


def test_history_is_stored_for_the_next_turn(world):
    llm, make = world
    deps = make()
    llm.queue("generate", good())
    ask(deps)
    assert [m.role for m in deps.log.messages("s")] == ["user", "assistant"]


def test_weak_retrieval_rewrites_once_then_searches_with_the_rewrite(world, settings):
    llm, make = world
    retriever = StubRetriever(cosines=[0.1, 0.5])
    llm.queue("rewrite", StandaloneQuery(query="return window days"))
    llm.queue("generate", good())
    response = ask(make(retriever))
    assert retriever.queries == ["return window days"]
    assert response.standalone_query == "How long do I have to return an item?"
    assert "How long do I have" in llm.calls[-1].user  # the model still answers the real question


def test_weak_retrieval_refuses_after_one_rewrite(world):
    llm, make = world
    retriever = StubRetriever(cosines=[0.1, 0.2])
    llm.queue("rewrite", StandaloneQuery(query="still nothing"))
    response = ask(make(retriever))
    assert not response.answer.answerable
    assert [c.call_name for c in llm.calls] == ["rewrite"] and retriever.queries == []


def test_model_refusal_returns_no_citations(world):
    llm, make = world
    llm.queue("generate", good(answerable=False))
    response = ask(make())
    assert not response.answer.answerable and response.answer.citations == []
    assert len(llm.calls) == 1  # a refusal is not a citation failure


def test_failed_citation_regenerates_once_with_a_hint(world, settings):
    llm, make = world
    deps = make()
    llm.queue("generate", good(quote="words the passage never says at all"))
    llm.queue("generate", good())
    response = ask(deps)
    assert response.answer.confidence == "high" and response.answer.citations[0].valid
    assert "<correction>" in llm.calls[1].user and "[1]" in llm.calls[1].user
    assert "never says" not in llm.calls[1].user  # model text does not come back as instructions
    assert request_rows(deps, settings)[0]["n_retries"] == 1


def test_still_failing_returns_low_confidence_with_a_warning_and_flags_the_citation(world):
    llm, make = world
    llm.queue("generate", good(quote="invented quote number one here"))
    llm.queue("generate", good(quote="invented quote number two here"))
    response = ask(make())
    assert response.answer.confidence == "low" and response.warnings
    assert [c.valid for c in response.answer.citations] == [False]  # shown flagged, never dropped


def test_marker_without_a_valid_citation_triggers_the_retry(world):
    llm, make = world
    unbacked = good()
    unbacked.answer = "Thirty days [1] and more [2]."
    llm.queue("generate", unbacked)
    llm.queue("generate", good())
    assert ask(make()).answer.confidence == "high" and len(llm.calls) == 2


def test_a_failed_retry_keeps_the_first_answer_with_warnings(world):
    llm, make = world
    llm.queue("generate", good(quote="invented quote number one here"))
    llm.queue("generate", StructuredOutputError("no valid answer"))
    response = ask(make())
    assert response.answer.confidence == "low"
    assert any("structured_output_failed" in w for w in response.warnings)


def test_budget_error_is_logged_with_its_code_and_stores_no_turn(world, settings):
    llm, make = world
    deps = make()
    llm.queue("generate", BudgetExceededError("limit"))
    with pytest.raises(BudgetExceededError) as caught:
        ask(deps)
    row = request_rows(deps, settings)[0]
    assert row["error_code"] == "budget_exceeded" and row["request_id"] == caught.value.request_id
    assert deps.log.messages("s") == []


def test_request_row_holds_the_prompt_version_and_counts(world, settings):
    llm, make = world
    deps = make()
    llm.queue("generate", good())
    response = ask(deps)
    row = request_rows(deps, settings)[0]
    assert row["prompt_version"] == "1" and row["n_citations"] == row["n_valid_citations"] == 1
    assert row["request_id"] == response.request_id and row["answerable"] == 1
    assert row["standalone_query"] == response.standalone_query and row["top_similarity"] == 0.8


def test_doc_filter_reaches_the_retriever(world):
    llm, make = world
    retriever = StubRetriever()
    llm.queue("generate", good())
    ask(make(retriever), doc_ids=["doc"])
    assert retriever.doc_ids == [["doc"]]


def test_deep_mode_is_rejected_not_ignored(world):
    llm, make = world
    with pytest.raises(ValidationFailedError):
        ask(make(), mode="deep")
    assert llm.calls == []


def test_cache_hit_makes_no_answer_call(world):
    llm, make = world
    llm.queue("generate", good())
    first = ask(make())
    hit = CacheHit(1, 0.97, "stored question", first.model_dump_json(), 0.0)
    llm.calls.clear()
    deps = make(cache=StubCache(hit))
    deps.log.delete_session("s")
    response = ask(deps)
    assert llm.calls == []
    assert response.cache.hit and response.cache.similarity == 0.97
    assert response.cache.saved_requests == 1 and response.usage.llm_requests == 0
    assert response.request_id != first.request_id


def test_only_fully_verified_answers_are_stored(world):
    llm, make = world
    cache = StubCache()
    llm.queue("generate", good())
    llm.queue("generate", good(quote="invented quote number one here"))
    llm.queue("generate", good(quote="invented quote number two here"))
    ask(make(cache=cache))
    assert cache.inserted == ["How long do I have to return an item?"]
    deps = make(cache=cache)
    deps.log.delete_session("s")
    ask(deps)
    assert len(cache.inserted) == 1


def test_response_round_trips_through_json(world):
    llm, make = world
    llm.queue("generate", good())
    response = ask(make())
    assert ChatResponse.model_validate_json(response.model_dump_json()) == response


def test_an_unbacked_marker_is_never_stored_even_when_listed_citations_are_valid(world):
    llm, make = world
    cache = StubCache()
    for _ in range(2):
        unbacked = good()
        unbacked.answer = "Thirty days [1] and a fee [2]."
        llm.queue("generate", unbacked)
    response = ask(make(cache=cache))
    assert all(c.valid for c in response.answer.citations) and response.answer.confidence == "low"
    assert response.warnings and cache.inserted == []


def test_a_citation_of_a_missing_passage_is_never_stored(world):
    llm, make = world
    cache = StubCache()
    answer = good()
    answer.citations.append(LLMCitation(marker=9, quote="a passage that was never provided"))
    llm.queue("generate", answer)
    response = ask(make(cache=cache))
    assert any("[9]" in w for w in response.warnings) and cache.inserted == []


def test_an_answer_kept_after_a_failed_retry_is_never_stored(world):
    llm, make = world
    cache = StubCache()
    llm.queue("generate", good(quote="invented quote number one here"))
    llm.queue("generate", StructuredOutputError("no valid answer"))
    ask(make(cache=cache))
    assert cache.inserted == []


def test_a_miss_records_the_best_similarity_it_saw(world, settings):
    llm, make = world
    llm.queue("generate", good())
    deps = make(cache=StubCache())
    response = ask(deps)
    assert response.cache.enabled and not response.cache.hit and response.cache.similarity == 0.61
    assert request_rows(deps, settings)[0]["cache_similarity"] == 0.61


def test_a_filtered_request_neither_reads_nor_writes_the_cache(world):
    """A cached answer may cite documents outside the filter, and an answer from a filter is not a general one."""
    llm, make = world
    cache = StubCache(hit=object())  # a lookup would crash on this hit
    llm.queue("generate", good())
    response = ask(make(cache=cache), doc_ids=["d1"])
    assert cache.looked_up == [] and cache.inserted == [] and not response.cache.enabled


def test_use_cache_false_bypasses_read_and_write(world):
    llm, make = world
    cache = StubCache(hit=object())
    llm.queue("generate", good())
    ask(make(cache=cache), use_cache=False)
    assert cache.looked_up == [] and cache.inserted == []


def test_a_hit_counts_savings_as_stored_minus_this_request(world):
    """A follow-up pays for condense, so the hit saves one request less than the stored answer cost."""
    llm, make = world
    stored = ChatResponse(
        request_id="old",
        session_id="other",
        standalone_query="stored question",
        answer=CitedAnswer(answer="Thirty days [1].", answerable=True, confidence="high"),
        cache=CacheInfo(enabled=True),
        usage=UsageInfo(llm_requests=2, est_cost_usd=0.5),
    )
    deps = make(cache=StubCache(CacheHit(1, 0.97, "stored question", stored.model_dump_json(), 0.5)))
    deps.log.add_turn("s", "earlier", "answer", "r0")
    llm.queue("condense", StandaloneQuery(query="How long do I have to return an item?"))
    response = ask(deps, message="and for that?")
    assert response.cache.hit and response.usage.llm_requests == 1
    assert response.cache.saved_requests == 1


def test_a_hit_is_still_served_when_the_daily_guard_is_full(world):
    """F-OPS-2: a first-turn hit needs no LLM call, so the guard has nothing to refuse."""
    llm, make = world
    llm.queue("generate", good())
    first = ask(make())
    llm.queue("generate", BudgetExceededError("Daily LLM request limit reached"))
    deps = make(cache=StubCache(CacheHit(1, 0.97, "q", first.model_dump_json(), 0.0)))
    deps.log.delete_session("s")
    assert ask(deps).cache.hit
