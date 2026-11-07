"""One `citerag eval run`: open the index, score retrieval, then answer and judge unless offline."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from citerag.agent_graph.factory import build_agent
from citerag.agents.deep_mode import DeepCrew
from citerag.agents.runtime import CrewRuntime
from citerag.analytics.logs import LogStore
from citerag.cache.base import NoOpCache
from citerag.cache.store import PersistentCache
from citerag.config import Settings
from citerag.eval.bench import BenchSummary, Spend, WorkloadItem, build_summary, run_both
from citerag.eval.experiments import Experiment
from citerag.eval.report import RESULTS_DIR
from citerag.eval.runner import RunResult, run_config, run_items, score_modes, start_or_resume, to_scores
from citerag.eval.scoring import summarize
from citerag.index.retriever import HybridRetriever
from citerag.index.store import open_store
from citerag.ingest.manifest import load_manifest
from citerag.llm.gateway import OpenRouterGateway
from citerag.rag.context import Chunk
from citerag.rag.pipeline import Deps
from citerag.schemas import GoldenItem, RunSummary
from citerag.tools.library import ToolLibrary


def _ranker(retriever: HybridRetriever, mode: str, top_k: int):
    def vector(query: str) -> list[Chunk]:
        return retriever.vector_ranking(query)[:top_k]

    def bm25(query: str) -> list[Chunk]:
        return retriever.bm25_ranking(query)[:top_k]

    return {"hybrid": retriever.candidates, "vector": vector, "bm25": bm25}[mode]


def execute_run(
    settings: Settings,
    golden: Path,
    items: list[GoldenItem],
    index_dir: Path,
    experiment: Experiment | None,
    offline: bool,
    rerank: bool,
    max_requests: int | None,
    resume: str | None,
    skip_judge: bool,
) -> RunSummary:
    name = experiment.name if experiment else "golden"
    run_id = resume or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{name}"
    run_dir = RESULTS_DIR / run_id
    start_or_resume(run_dir, run_config(settings, golden, name if experiment else None))
    ranking = experiment.retrieval if experiment else "hybrid"
    score_rerank = rerank or (experiment.rerank if experiment else False)

    result: RunResult | None = None
    store = open_store(settings, index_dir)
    try:
        retriever = HybridRetriever(store, settings)

        def cosine(question: str) -> float:
            return retriever.top_cosine(retriever.embed_query(question))

        reranked = None
        if score_rerank:
            on = settings.model_copy(update={"rerank_enabled": True})
            reranked = HybridRetriever(store, on).retrieve
        rank = _ranker(retriever, ranking, settings.top_k_retrieve)
        before, after = score_modes(items, rank, cosine, reranked, settings.top_k_context)
        if not offline:
            log = LogStore(settings.logs_db_path)
            llm = OpenRouterGateway(settings, log)
            mode = experiment.mode if experiment else "standard"
            deps = Deps(
                cfg=settings,
                llm=llm,
                retriever=retriever,
                cache=NoOpCache(),
                log=log,
                namespace=lambda: "eval",
                deep=DeepCrew(CrewRuntime(settings, llm, log)) if mode == "deep" else None,
                agent=build_agent(settings, llm) if mode == "agent" else None,
                tool_library=lambda scope: ToolLibrary(
                    settings, retriever, lambda: load_manifest(index_dir), scope
                ),
            )
            result = asyncio.run(run_items(items, deps, run_dir, max_requests, skip_judge, mode))
    finally:
        store.close()

    summary = RunSummary(
        run_id=run_id,
        status="offline" if result is None else result.status,
        reason=None if result is None else result.reason,
        experiment=name if experiment else None,
        golden=str(golden),
        n_items=len(items),
        n_done=0 if result is None else len(result.records),
        unverified_items=sum(not item.verified for item in items),
        requests_used=0 if result is None else result.requests,
        before_rerank=to_scores(before),
        after_rerank=to_scores(after) if after else None,
        answerable_cosines=before.answerable_cosines,
        unanswerable_cosines=before.unanswerable_cosines,
        answers=summarize(result.records) if result and result.records else None,
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


BENCH_NAMESPACE = "bench"


def execute_bench(
    settings: Settings,
    workload: Path,
    items: list[WorkloadItem],
    index_dir: Path,
    max_requests: int | None,
    resume: str | None,
    skip_judge: bool,
) -> BenchSummary:
    """Never uses a stored response: that would hide the thing being measured (ARCHITECTURE section 9)."""
    run_id = resume or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-bench-cache"
    run_dir = RESULTS_DIR / run_id
    start_or_resume(run_dir, run_config(settings, workload, "bench-cache"))
    spend = Spend(max_requests)
    store = open_store(settings, index_dir)
    try:
        retriever = HybridRetriever(store, settings)
        log = LogStore(settings.logs_db_path)
        llm = OpenRouterGateway(settings, log)
        cache = PersistentCache(
            str(run_dir / "cache.sqlite"),
            384,
            settings.cache_threshold,
            max_entries=settings.cache_max_entries,
        )

        on = settings.model_copy(
            update={"cache_enabled": True}
        )  # the benchmark measures the cache, whatever .env says

        def deps(backend: NoOpCache | PersistentCache) -> Deps:
            return Deps(
                cfg=on,
                llm=llm,
                retriever=retriever,
                cache=backend,
                log=log,
                namespace=lambda: BENCH_NAMESPACE,
            )

        status, reason = asyncio.run(
            run_both(items, deps(NoOpCache()), deps(cache), run_dir, spend, skip_judge)
        )
    finally:
        store.close()
    summary = build_summary(
        run_id, status, reason, workload, settings.cache_threshold, items, run_dir, spend.used
    )
    (run_dir / "bench_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary
