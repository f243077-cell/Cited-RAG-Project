"""bench-cache end to end: real cache, real pipeline, FakeGateway and HashEmbedding. No network."""

import asyncio
import json

import numpy as np
import pytest

from citerag.analytics.logs import LogStore
from citerag.cache.base import NoOpCache
from citerag.cache.store import PersistentCache
from citerag.errors import BudgetExceededError, LLMUnavailableError
from citerag.eval.bench import (
    BenchRecord,
    Spend,
    WorkloadItem,
    build_summary,
    estimate_requests,
    load_workload,
    render_report,
    repeat_share,
    run_pass,
)
from citerag.llm.fake import FakeGateway
from citerag.rag.context import Chunk
from citerag.rag.pipeline import Deps
from citerag.schemas import JudgeVerdict, LLMCitation, LLMCitedAnswer

TEXT = "Items can be returned within thirty days of delivery."
CHUNK = Chunk("doc-0000", "doc", "policy.pdf", 3, TEXT)
ITEMS = [
    WorkloadItem(id="w1", question="How long can I return items?", gold_answer="Thirty days."),
    WorkloadItem(
        id="w2", question="How long can I return items??", gold_answer="Thirty days.", repeat_of="w1"
    ),
]


class Retriever:
    def has_documents(self, doc_ids=None):
        return True

    def embed_query(self, query):
        return list(np.ones(8))  # every question is the same to this embedder: only the threshold is in play

    def top_cosine(self, query_embedding, doc_ids=None):
        return 0.9

    def retrieve(self, query, doc_ids=None):
        return [CHUNK]


def answer():
    citation = LLMCitation(marker=1, quote=TEXT)
    return LLMCitedAnswer(answer="Thirty days [1].", answerable=True, citations=[citation], confidence="high")


def verdict(correctness="correct"):
    return JudgeVerdict(faithful=True, correctness=correctness, reasoning="ok")


@pytest.fixture
def bench(settings, tmp_path):
    llm = FakeGateway()
    log = LogStore(settings.logs_db_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def deps(cache):
        return Deps(settings, llm, Retriever(), cache, log, lambda: "bench")

    return llm, deps, run_dir


def cache_in(run_dir):
    return PersistentCache(str(run_dir / "cache.sqlite"), 8, threshold=0.92)


def run(*args, **kw):
    return asyncio.run(run_pass(*args, **kw))


def replay(llm, deps, run_dir, spend):
    for _ in ITEMS:  # the off pass answers and judges everything
        llm.queue("generate", answer())
        llm.queue("judge", verdict())
    llm.queue("generate", answer())  # the on pass answers once
    llm.queue("judge", verdict())
    off = run("off", ITEMS, deps(NoOpCache()), run_dir, spend, False)
    on = run("on", ITEMS, deps(cache_in(run_dir)), run_dir, spend, False)
    return off, on


def test_a_full_replay_compares_cache_off_with_cache_on(bench):
    llm, deps, run_dir = bench
    spend = Spend(None)
    assert replay(llm, deps, run_dir, spend) == (("complete", None), ("complete", None))
    summary = build_summary("r", "complete", None, run_dir / "w.jsonl", 0.92, ITEMS, run_dir, spend.used)
    off, on = summary.passes["off"], summary.passes["on"]
    assert (off.hits, on.hits) == (0, 1) and on.hit_rate.rate == 0.5
    assert (off.llm_requests, on.llm_requests) == (2, 1) and summary.request_reduction == 0.5
    assert off.correct.n == on.correct.n == 2 and summary.correctness_change_points == 0
    assert summary.false_hits == [] and spend.used == 5  # the hit reused a verdict: 2 + 2 + 1, not 6


def test_a_hit_with_another_gold_answer_is_judged_and_a_wrong_one_is_reported(bench):
    llm, deps, run_dir = bench
    items = [ITEMS[0], ITEMS[1].model_copy(update={"gold_answer": "Sixty days."})]
    for call, item in (("generate", answer()), ("judge", verdict()), ("judge", verdict("incorrect"))):
        llm.queue(call, item)
    spend = Spend(None)
    run("on", items, deps(cache_in(run_dir)), run_dir, spend, False)
    summary = build_summary("r", "complete", None, run_dir / "w", 0.92, items, run_dir, spend.used)
    assert summary.false_hits == ["w2"] and spend.used == 3


def test_a_hit_is_judged_on_the_passages_of_the_answer_it_served(bench):
    llm, deps, run_dir = bench
    items = [ITEMS[0], ITEMS[1].model_copy(update={"gold_answer": "Sixty days."})]
    for call, item in (("generate", answer()), ("judge", verdict()), ("judge", verdict())):
        llm.queue(call, item)
    run("on", items, deps(cache_in(run_dir)), run_dir, Spend(None), False)
    judged = [c for c in llm.calls if c.call_name == "judge"]
    assert TEXT in judged[1].user


def test_the_cap_pauses_and_a_resume_costs_nothing_twice(bench):
    llm, deps, run_dir = bench
    llm.queue("generate", answer())
    spend = Spend(3)
    status, reason = run("off", ITEMS, deps(NoOpCache()), run_dir, spend, True)
    assert status == "paused" and "--max-requests 3" in reason and spend.used == 1
    assert len(llm.calls) == 1
    llm.queue("generate", answer())
    assert run("off", ITEMS, deps(NoOpCache()), run_dir, Spend(None), True)[0] == "complete"
    assert len(llm.calls) == 2


def test_quota_pauses_and_a_hanging_provider_stops(bench):
    llm, deps, run_dir = bench
    llm.queue("generate", BudgetExceededError("limit"))
    assert run("off", ITEMS, deps(NoOpCache()), run_dir, Spend(None), True) == ("paused", "limit")
    llm.queue("generate", LLMUnavailableError("down"))
    assert run("off", ITEMS, deps(NoOpCache()), run_dir, Spend(None), True) == ("stopped", "down")


def test_the_bench_cache_stays_in_the_run_folder(bench, settings):
    llm, deps, run_dir = bench
    llm.queue("generate", answer())
    run("on", ITEMS, deps(cache_in(run_dir)), run_dir, Spend(None), True)
    assert (run_dir / "cache.sqlite").exists() and not settings.cache_db_path.exists()


def test_a_refusal_counts_against_correctness(bench):
    llm, deps, run_dir = bench
    refusal = LLMCitedAnswer(answer="Not in the documents.", answerable=False, confidence="high")
    llm.queue("generate", refusal)
    run("off", ITEMS[:1], deps(NoOpCache()), run_dir, Spend(None), False)
    summary = build_summary("r", "complete", None, run_dir / "w", 0.92, ITEMS[:1], run_dir, 1)
    assert summary.passes["off"].correct.n == 0 and summary.passes["off"].correct.of == 1


def test_an_unfinished_run_reports_no_reduction(bench):
    llm, deps, run_dir = bench
    llm.queue("generate", answer())
    run("off", ITEMS[:1], deps(NoOpCache()), run_dir, Spend(None), True)
    summary = build_summary("r", "paused", "cap", run_dir / "w", 0.92, ITEMS, run_dir, 1)
    assert summary.request_reduction is None and summary.correctness_change_points is None


def test_the_report_labels_itself_and_shows_the_targets(bench):
    llm, deps, run_dir = bench
    spend = Spend(None)
    replay(llm, deps, run_dir, spend)
    summary = build_summary("r", "complete", None, run_dir / "w", 0.92, ITEMS, run_dir, spend.used)
    report = render_report(summary, "chart.png")
    assert "(provisional)" in report and "Same model as answerer" in report
    assert (
        "| Hit rate | at least 30% | 1 of 2 (50%) |" in report and "![Cache off vs on](chart.png)" in report
    )


def test_the_estimate_needs_no_llm_and_counts_repeats_once(tmp_path):
    assert estimate_requests(ITEMS) == (6, 16)
    assert estimate_requests(ITEMS, skip_judge=True) == (3, 12)
    assert repeat_share(ITEMS) == 0.5


def test_a_workload_file_round_trips_and_a_missing_one_says_how_to_build_it(tmp_path):
    path = tmp_path / "w.jsonl"
    path.write_text("".join(i.model_dump_json() + "\n" for i in ITEMS), encoding="utf-8")
    assert load_workload(path) == ITEMS
    with pytest.raises(Exception, match="make_workload"):
        load_workload(tmp_path / "none.jsonl")


def test_records_survive_a_save_with_their_cache_fields():
    record = BenchRecord(
        id="w", type="answerable", question="q", status="answered", cache_hit=True, similarity=0.9
    )
    assert BenchRecord.model_validate(json.loads(record.model_dump_json())).similarity == 0.9
