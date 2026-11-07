"""Command line entry point."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from citerag.config import Settings, SettingsError, UISettings, get_settings
from citerag.errors import CiteRAGError, ValidationFailedError

if TYPE_CHECKING:
    from citerag.eval.retrieval import RetrievalReport

app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True, help="Measure retrieval and answers.")
app.add_typer(eval_app, name="eval")
cache_app = typer.Typer(no_args_is_help=True, help="Inspect or empty the semantic cache.")
app.add_typer(cache_app, name="cache")
logs_app = typer.Typer(no_args_is_help=True, help="Manage the local request log.")
app.add_typer(logs_app, name="logs")


def run() -> None:
    """The `citerag` entry point: a missing setting is one clear line on stderr and exit code 2, not a stack trace."""
    try:
        app()
    except SettingsError as exc:
        typer.echo(str(exc), err=True)
        raise SystemExit(2) from None


@app.callback()
def main() -> None:
    """CiteRAG: cited RAG over your own documents."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(package_version("citerag"))


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="A file or a folder")],
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
    via_api: Annotated[bool, typer.Option(help="Send the files to the running backend")] = False,
    config: Annotated[Path | None, typer.Option(help="Experiment YAML: settings and index folder")] = None,
    captions: Annotated[bool, typer.Option(help="Caption images: one LLM request per new image")] = False,
) -> None:
    """Add documents to the index. Stop the API first, or use --via-api: Chroma allows one writer process."""
    if via_api:
        _ingest_via_api(path, captions)
        return
    from citerag.eval.experiments import apply_overrides, load_experiment
    from citerag.index.store import open_store  # loads the heavy frameworks, so not at import time
    from citerag.ingest.pipeline import Ingestor

    settings = get_settings()
    try:
        experiment = load_experiment(config) if config else None
        if experiment is not None:
            settings = apply_overrides(settings, experiment.overrides)
            index_dir = index_dir or experiment.index_dir
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    target = index_dir or settings.index_dir
    store = open_store(settings, target)
    caption, close_captions = None, None
    if captions:
        from citerag.analytics.logs import LogStore
        from citerag.llm.gateway import OpenRouterGateway
        from citerag.multimodal.vision import blocking_captioner

        llm = OpenRouterGateway(settings, LogStore(settings.logs_db_path))
        caption, close_captions = blocking_captioner(llm)
    try:
        reports, failures = Ingestor(settings, store, target, captioner=caption).ingest_path(path)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    finally:
        store.close()
        if close_captions is not None:
            close_captions()
    for r in reports:
        typer.echo(
            f"{r.doc_name}: {r.status} ({r.pages or '-'} pages, {r.chunks} chunks, "
            f"{r.images_captioned} captioned, {r.images_skipped} images skipped, {r.duration_ms} ms)"
        )
    for f in failures:
        typer.echo(f"{f.name}: {f.code}: {f.message}")
    typer.echo(f"{len(reports)} file(s), {sum(r.chunks for r in reports)} chunk(s), {len(failures)} failed")
    if failures:
        raise typer.Exit(1)


def _ingest_via_api(path: Path, captions: bool = False) -> None:
    """One upload per file, so one bad file does not stop the others."""
    from citerag.ingest.loaders import SKIPPED_NAMES
    from citerag.ui.api_client import ApiClient, ApiError

    if not path.exists():
        typer.echo(f"not_found: {path} does not exist")
        raise typer.Exit(1)
    cfg = UISettings()
    client = ApiClient(
        cfg.api_base_url, cfg.citerag_api_key.get_secret_value() if cfg.citerag_api_key else None
    )
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    failed = 0
    for file in files:
        if path.is_dir() and file.name in SKIPPED_NAMES:
            continue
        try:
            for r in client.upload([(file.name, file.read_bytes())], captions):
                typer.echo(f"{r.doc_name}: {r.status} ({r.pages or '-'} pages, {r.chunks} chunks)")
        except ApiError as exc:
            failed += 1
            typer.echo(f"{file.name}: {exc.code}: {exc.message}")
    if failed:
        raise typer.Exit(1)


def _print_report(title: str, report: RetrievalReport) -> None:
    typer.echo(f"\n{title} (n={report.n_scored})")
    for k in report.ks:
        typer.echo(f"  hit@{k}: {report.hits[k]}/{report.n_scored} ({report.hit_rate[k]:.0%})")
    typer.echo(f"  MRR@{report.mrr_k}: {report.mrr:.3f}")
    for k, n in report.full_coverage.items():
        typer.echo(f"  multi_chunk full_coverage@{k}: {n}/{report.n_multi_chunk}")


def _check_index(target: Path, settings: Settings) -> None:
    from citerag.ingest.manifest import check_manifest, load_manifest

    manifest = load_manifest(target)
    if manifest is None or not manifest.docs:
        raise ValidationFailedError(f"no index at {target}: ingest documents first")
    check_manifest(manifest, settings)


@eval_app.command("retrieval")
def eval_retrieval(
    golden: Annotated[Path, typer.Option(help="JSONL of golden items")] = Path(
        "eval/golden/candidates.jsonl"
    ),
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
    rerank: Annotated[bool, typer.Option(help="Also score the list after the cross-encoder")] = False,
    require_verified: Annotated[bool, typer.Option(help="Refuse items no human has verified")] = False,
    skip_type: Annotated[list[str] | None, typer.Option(help="Leave out an item type, e.g. image")] = None,
) -> None:
    """hit@k and MRR on a golden file. Uses zero LLM requests."""
    from citerag.eval.golden import load_golden
    from citerag.eval.retrieval import score_retrieval
    from citerag.index.retriever import HybridRetriever
    from citerag.index.store import open_store

    settings = get_settings()
    target = index_dir or settings.index_dir
    try:
        items = load_golden(golden, require_verified=require_verified)
        items = [item for item in items if item.type not in (skip_type or [])]
        _check_index(target, settings)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc

    store = open_store(settings, target)
    try:
        retriever = HybridRetriever(store, settings)

        def cosine(question: str) -> float:
            return retriever.top_cosine(retriever.embed_query(question))

        before = score_retrieval(items, retriever.candidates, cosine)
        after = None
        if rerank:
            on = settings.model_copy(update={"rerank_enabled": True})
            ks = tuple(k for k in before.ks if k <= settings.top_k_context)
            after = score_retrieval(items, HybridRetriever(store, on).retrieve, cosine, ks=ks, mrr_k=ks[-1])
    finally:
        store.close()

    unverified = sum(not item.verified for item in items)
    if unverified:
        typer.echo(f"UNVERIFIED: {unverified} of {len(items)} items have not been checked by a human")
    for row in before.items:
        rank = "-" if row.first_hit_rank is None else row.first_hit_rank
        typer.echo(f"{row.id}: rank {rank}, cosine {row.top_cosine:.3f}, top {row.top_sources[:1]}")
    _print_report("Hybrid, before rerank", before)
    if after is not None:
        _print_report("After rerank", after)
    typer.echo(f"\nGate cosines, answerable: {sorted(round(c, 3) for c in before.answerable_cosines)}")
    typer.echo(f"Gate cosines, unanswerable: {sorted(round(c, 3) for c in before.unanswerable_cosines)}")

    run_dir = Path("eval/results") / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-retrieval")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "golden": str(golden),
        "unverified_items": unverified,
        "config": _retrieval_config(settings),
        "before_rerank": before.model_dump(),
        "after_rerank": after.model_dump() if after else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"Saved {run_dir / 'summary.json'}")


@eval_app.command("run")
def eval_run(
    golden: Annotated[Path, typer.Option(help="JSONL of golden items")] = Path("eval/golden/golden.jsonl"),
    config: Annotated[Path | None, typer.Option(help="Experiment YAML in configs/experiments")] = None,
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
    offline: Annotated[bool, typer.Option(help="Retrieval metrics only: zero LLM requests")] = False,
    rerank: Annotated[bool, typer.Option(help="Also score the list after the cross-encoder")] = False,
    limit: Annotated[int | None, typer.Option(help="Only the first N items")] = None,
    max_requests: Annotated[int | None, typer.Option(help="Pause before this many LLM requests")] = None,
    resume: Annotated[str | None, typer.Option(help="Continue this run id")] = None,
    skip_judge: Annotated[bool, typer.Option(help="No verdicts now; --resume adds them later")] = False,
    skip_type: Annotated[list[str] | None, typer.Option(help="Leave out an item type")] = None,
    require_verified: Annotated[bool, typer.Option(help="Refuse items no human has verified")] = False,
    estimate: Annotated[bool, typer.Option(help="Print the LLM request estimate and stop")] = False,
) -> None:
    """Score the system on a golden file. Saves after every item, so a run can resume."""
    from citerag.eval.execute import execute_run
    from citerag.eval.experiments import apply_overrides, load_experiment
    from citerag.eval.golden import load_golden

    try:
        settings, experiment = get_settings(), None
        if config is not None:
            experiment = load_experiment(config)
            settings = apply_overrides(settings, experiment.overrides)
            index_dir = index_dir or experiment.index_dir
        target = index_dir or settings.index_dir
        items = load_golden(golden, require_verified=require_verified)
        mode = experiment.mode if experiment else "standard"
        no_images = offline or mode == "agent"  # offline needs a vision call; Agent mode takes no images
        leave_out = {*(skip_type or []), *(["image"] if no_images else [])}
        items = [item for item in items if item.type not in leave_out][:limit]
        if estimate:
            from citerag.eval.modes import estimate as estimate_requests

            team = settings.agent_architecture == "supervisor"
            expected, worst = estimate_requests(items, mode, skip_judge, supervisor=team)
            typer.echo(f"{len(items)} items in {mode} mode: about {expected} LLM requests, {worst} at worst")
            return
        _check_index(target, settings)
        summary = execute_run(
            settings, golden, items, target, experiment, offline, rerank, max_requests, resume, skip_judge
        )
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc

    if summary.unverified_items:
        typer.echo(
            f"UNVERIFIED: {summary.unverified_items} of {summary.n_items} items: provisional numbers only"
        )
    typer.echo(f"Run {summary.run_id}: {summary.status}{f' ({summary.reason})' if summary.reason else ''}")
    typer.echo(f"Items done: {summary.n_done} of {summary.n_items}. LLM requests: {summary.requests_used}")
    for name, value in summary.headline().items():
        typer.echo(f"  {name}: {'not measured' if value is None else f'{value:.1%}'}")
    if summary.answers is not None:
        a = summary.answers
        typer.echo(f"  injection leaks: {a.injection_leaks} of {a.injection_items}")
        typer.echo(f"  waiting for the judge: {a.pending_judge}; errors: {a.errors}")
    if summary.status in ("paused", "stopped"):
        typer.echo(f"Continue with --resume {summary.run_id}")


@eval_app.command("compare")
def eval_compare(run_a: str, run_b: str) -> None:
    """Delta table for two runs. With about 40 items, differences under 3 points are noise."""
    from citerag.eval.report import NOISE_POINTS, compare_rows, load_summary

    try:
        a, b = load_summary(run_a), load_summary(run_b)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc

    def show(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    typer.echo(f"{'metric':<24}{'A':>9}{'B':>9}{'delta':>10}")
    for name, x, y, delta in compare_rows(a, b):
        change = "-" if delta is None else f"{delta:+.1f}"
        note = "" if delta is None or abs(delta) >= NOISE_POINTS else "  (noise)"
        typer.echo(f"{name:<24}{show(x):>9}{show(y):>9}{change:>10}{note}")


@eval_app.command("gate")
def eval_gate(run_id: str) -> None:
    """What each MIN_TOP_SIMILARITY would do on a run's cosines. Prints a proposal and changes nothing."""
    from citerag.eval.report import load_summary
    from citerag.eval.scoring import gate_sweep, propose_gate

    try:
        summary = load_summary(run_id)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    answerable, unanswerable = summary.answerable_cosines, summary.unanswerable_cosines
    rows = gate_sweep(answerable, unanswerable)
    for row in rows:
        if row.tau * 100 % 5 < 0.01:
            typer.echo(
                f"tau {row.tau:.2f}: stops {row.false_refusals} of {len(answerable)} answerable, "
                f"{row.gate_refusals} of {len(unanswerable)} unanswerable"
            )
    best = propose_gate(rows)
    if best is None:
        typer.echo("No threshold keeps every answerable item.")
    else:
        typer.echo(
            f"Proposal: {best.tau:.2f}, the highest with no false refusal. It stops "
            f"{best.gate_refusals} of {len(unanswerable)} unanswerable items. Set MIN_TOP_SIMILARITY yourself."
        )


@app.command()
def mcp(
    local: Annotated[bool, typer.Option(help="Open the index here; the API must be stopped")] = False,
    index_dir: Annotated[Path | None, typer.Option(help="With --local. Default: DATA_DIR/index")] = None,
) -> None:
    """MCP server over stdio with three read-only tools. stdout carries only the protocol."""
    from citerag.mcp_server.server import HttpBackend, LocalBackend, build_server, quiet_logging

    quiet_logging()
    settings = get_settings()
    if local or settings.mcp_mode == "local":
        backend = LocalBackend(settings, index_dir or settings.index_dir)
        backend.warm_up()  # on the main thread: see LocalBackend.warm_up
    else:
        key = settings.citerag_api_key.get_secret_value() if settings.citerag_api_key else None
        backend = HttpBackend(settings.api_base_url, key)
    build_server(backend).run("stdio")


@app.command()
def quota() -> None:
    """LLM requests in the last 24 hours against each daily guard. Reads the local log: zero LLM requests."""
    from citerag.llm.budget import requests_last_24h

    settings = get_settings()
    limits = {"openrouter": settings.max_requests_per_day, "gemini": settings.gemini_max_requests_per_day}
    for provider, limit in limits.items():
        typer.echo(f"{provider}: {requests_last_24h(settings.logs_db_path, provider=provider)} of {limit}")


@logs_app.command("purge")
def logs_purge(
    days: Annotated[int, typer.Option(min=1, help="Keep this many days. The guard needs the last one")] = 30,
) -> None:
    """Delete request, LLM-call and message rows older than this many days (ARCHITECTURE section 18)."""
    import time

    from citerag.analytics.logs import LogStore

    log = LogStore(get_settings().logs_db_path)
    requests, calls, messages = log.purge(time.time() - days * 86_400)
    typer.echo(f"Deleted {requests} requests, {calls} LLM calls, {messages} messages")


@cache_app.command("stats")
def cache_stats() -> None:
    """Entries and backend. Reads the file, so it works with the API stopped."""
    from citerag.cache.store import open_cache
    from citerag.eval.cache_tools import load_sweep, threshold_warning

    settings = get_settings()
    stats = open_cache(settings).stats()
    typer.echo(f"backend: {stats.backend}; enabled: {stats.enabled}; entries: {stats.entries}")
    warning = threshold_warning(settings.cache_enabled, settings.cache_threshold, load_sweep())
    if warning:
        typer.echo(f"WARNING: {warning}")


@cache_app.command("clear")
def cache_clear() -> None:
    """Delete every entry. A running API notices on its next lookup; POST /v1/cache/clear does the same at once."""
    from citerag.cache.store import open_cache

    cache = open_cache(get_settings())
    cache.clear()
    typer.echo(f"Cache cleared; entries: {cache.stats().entries}")


@eval_app.command("cache-sweep")
def eval_cache_sweep(
    pairs: Annotated[Path, typer.Option(help="JSONL of labelled pairs")] = Path("eval/cache_pairs.jsonl"),
    max_false_hit_rate: Annotated[float, typer.Option(help="Cap on the false-hit rate")] = 0.02,
    chart: Annotated[Path, typer.Option(help="Where the chart goes")] = Path("docs/results/e5_threshold.png"),
) -> None:
    """E5: precision, recall and false-hit rate per threshold, on labelled pairs only. Zero LLM requests.
    Prints a recommendation and changes nothing."""
    from citerag.eval.cache_tools import load_pairs, save_sweep, sweep_pairs

    settings = get_settings()
    try:
        found = load_pairs(pairs)
        if not any(pair.labelled for pair in found):
            sweep_pairs(found, _no_embedding)  # says there are no labelled pairs
        from citerag.analytics.plots import threshold_chart
        from citerag.index.store import make_embedder

        result = sweep_pairs(found, make_embedder(settings).get_query_embedding, max_false_hit_rate)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    save_sweep(result)
    chart.parent.mkdir(parents=True, exist_ok=True)
    threshold_chart(result).savefig(chart, dpi=150)
    typer.echo(f"Labelled pairs: {result.n_equivalent} equivalent, {result.n_near_miss} near miss")
    if not result.enough_pairs:
        typer.echo("PROVISIONAL: F-CAC-8 needs at least 30 of each")
    if result.recommended is None:
        typer.echo("No threshold is both safe and useful: improve the guard or the embedding first.")
    else:
        row = next(r for r in result.rows if r.tau == result.recommended)
        typer.echo(
            f"Recommended CACHE_THRESHOLD: {result.recommended:.2f} (false-hit rate {row.false_hit_rate:.1%}, "
            f"recall {row.recall:.1%}, precision {row.precision:.1%}). Nothing was changed."
        )
    for pair in result.worst_near_misses[:5]:
        guard = "guard passes" if pair.guard_ok else "guard blocks"
        typer.echo(f"  {pair.id} {pair.similarity:.3f} ({pair.kind}, {guard})")
    typer.echo(f"Chart: {chart}")


def _no_embedding(text: str) -> list[float]:
    raise AssertionError(text)  # sweep_pairs stops before embedding when nothing is labelled


@eval_app.command("bench-cache")
def eval_bench_cache(
    workload: Annotated[Path, typer.Option(help="JSONL of workload items")] = Path("eval/workload.jsonl"),
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
    max_requests: Annotated[int | None, typer.Option(help="Pause before this many LLM requests")] = None,
    resume: Annotated[str | None, typer.Option(help="Continue this run id")] = None,
    skip_judge: Annotated[bool, typer.Option(help="No verdicts now; --resume adds them later")] = False,
    estimate: Annotated[bool, typer.Option(help="Print the request estimate and stop")] = False,
    report: Annotated[Path, typer.Option(help="Written when both passes finish")] = Path(
        "docs/results/E6_cache_benchmark.md"
    ),
) -> None:
    """E6: replay a workload with the cache off, then on. Uses LLM requests; --estimate uses none."""
    from citerag.eval.bench import estimate_requests, load_workload, render_report

    settings = get_settings()
    target = index_dir or settings.index_dir
    try:
        items = load_workload(workload)
        expected, worst = estimate_requests(items, skip_judge)
        typer.echo(f"{len(items)} questions: about {expected} LLM requests, {worst} at worst")
        if estimate:
            return
        from citerag.eval.execute import execute_bench

        _check_index(target, settings)
        summary = execute_bench(settings, workload, items, target, max_requests, resume, skip_judge)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc

    off, on = summary.passes["off"], summary.passes["on"]
    typer.echo(f"Run {summary.run_id}: {summary.status}{f' ({summary.reason})' if summary.reason else ''}")
    typer.echo(f"Off: {off.n_done} of {off.n_items}, {off.llm_requests} LLM requests")
    typer.echo(f"On: {on.n_done} of {on.n_items}, {on.llm_requests} LLM requests, {on.hits} hits")
    typer.echo(f"LLM requests this call: {summary.requests_used}")
    if summary.status != "complete":
        typer.echo(f"Continue with --resume {summary.run_id}")
    elif on.n_done == on.n_items:
        from citerag.analytics.plots import bench_chart

        chart = report.with_name("e6_cache.png")
        chart.parent.mkdir(parents=True, exist_ok=True)
        bench_chart(summary).savefig(chart, dpi=150)
        report.write_text(render_report(summary, chart.name), encoding="utf-8")
        typer.echo(f"Report: {report}")


@eval_app.command("gen-candidates")
def eval_gen_candidates(
    n: Annotated[int, typer.Option(help="How many candidates to aim for")] = 60,
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
    out: Annotated[Path, typer.Option(help="Appended to")] = Path("eval/golden/candidates.jsonl"),
    seed: Annotated[int, typer.Option(help="Which chunks get sampled")] = 7,
    max_requests: Annotated[int | None, typer.Option(help="Pause before this many LLM requests")] = None,
    estimate: Annotated[bool, typer.Option(help="Print the request estimate and stop")] = False,
) -> None:
    """Question Writer plus Question Critic. Appends items with verified=false: a human must review them."""
    from citerag.agents.eval_crews import estimate_candidate_requests

    typer.echo(f"Up to {n} candidates: about {estimate_candidate_requests(n)} LLM requests")
    if estimate:
        return
    import asyncio

    from citerag.agents.eval_crews import generate_candidates
    from citerag.agents.runtime import CrewRuntime
    from citerag.analytics.logs import LogStore
    from citerag.eval.golden import load_golden
    from citerag.index.retriever import HybridRetriever
    from citerag.index.store import open_store
    from citerag.llm.gateway import OpenRouterGateway

    settings = get_settings()
    target = index_dir or settings.index_dir
    try:
        existing = load_golden(out) if out.exists() else []
        _check_index(target, settings)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    store = open_store(settings, target)
    try:
        chunks = HybridRetriever(store, settings).all_chunks()
    finally:
        store.close()
    log = LogStore(settings.logs_db_path)
    runtime = CrewRuntime(settings, OpenRouterGateway(settings, log), log)
    run = asyncio.run(generate_candidates(runtime, chunks, existing, out, n, max_requests, seed))
    typer.echo(f"{run.status}{f' ({run.reason})' if run.reason else ''}: {run.written} candidates in {out}")
    typer.echo(f"LLM requests: {run.requests}. Every item is unverified until a human reviews it.")


@eval_app.command("analyze-failures")
def eval_analyze_failures(
    run_id: str,
    out: Annotated[Path, typer.Option(help="The report")] = Path("docs/results/failure_analysis.md"),
    estimate: Annotated[bool, typer.Option(help="Count failed items and stop")] = False,
) -> None:
    """Failure Analyst: group a run's failed items by root cause. One LLM request; none if nothing failed."""
    import asyncio

    from citerag.agents.eval_crews import analyze_failures, render_failures
    from citerag.agents.runtime import CrewRuntime
    from citerag.analytics.logs import LogStore
    from citerag.eval.report import load_detail
    from citerag.llm.gateway import OpenRouterGateway

    settings = get_settings()
    try:
        records = load_detail(run_id).items
        failed = render_failures(records)[1]
        typer.echo(f"{len(failed)} failed items of {len(records)}")
        if estimate or not failed:
            return
        log = LogStore(settings.logs_db_path)
        runtime = CrewRuntime(settings, OpenRouterGateway(settings, log), log)
        asyncio.run(analyze_failures(runtime, run_id, records, out))
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    typer.echo(f"Report: {out}")


@eval_app.command("report-e8")
def eval_report_e8(
    runs: Annotated[list[str], typer.Argument(help="Run ids, in the column order you want")],
    out: Annotated[Path, typer.Option(help="The report")] = Path("docs/results/E8_modes.md"),
    title: Annotated[str, typer.Option(help="Report heading")] = "E8: standard vs Deep vs Agent",
) -> None:
    """E8 table: standard, Deep and Agent runs side by side. Reads run folders only: zero LLM requests.
    M12 uses it too, for single agent vs supervisor vs Deep."""
    from citerag.analytics.plots import grouped_bars
    from citerag.eval.modes import ModeColumn, render
    from citerag.eval.report import load_detail

    try:
        columns = []
        for run_id in runs:
            detail = load_detail(run_id)
            columns.append(ModeColumn(detail.summary.experiment or run_id, detail.summary, detail.items))
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    import pandas as pd

    shares = pd.DataFrame({c.name: c.summary.headline() for c in columns}).dropna(how="all")
    chart = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not shares.empty:
        grouped_bars(shares.iloc[:, :3], title).savefig(chart, dpi=150)
    out.write_text(render(columns, chart.name if not shares.empty else None, title), encoding="utf-8")
    typer.echo(f"Report: {out}")


@eval_app.command("parts")
def eval_parts(
    golden: Annotated[Path, typer.Option(help="JSONL of golden items")] = Path(
        "eval/golden/candidates.jsonl"
    ),
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
    k: Annotated[int, typer.Option(help="A part is missed when it is not in the top k")] = 5,
) -> None:
    """For each multi_chunk item: the hybrid rank of each gold part. Uses zero LLM requests."""
    from citerag.eval.golden import load_golden
    from citerag.eval.retrieval import part_ranks
    from citerag.index.retriever import HybridRetriever
    from citerag.index.store import open_store

    settings = get_settings()
    target = index_dir or settings.index_dir
    try:
        items = [item for item in load_golden(golden) if item.type == "multi_chunk"]
        _check_index(target, settings)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc

    store = open_store(settings, target)
    try:
        retriever = HybridRetriever(store, settings)
        found = [(item, retriever.candidates(item.question)) for item in items]
    finally:
        store.close()

    full = 0
    for item, chunks in found:
        rows = part_ranks(chunks, item.gold_sources, k)
        missed = [row.part for row in rows if row.missed]
        full += not missed
        typer.echo(f"\n{item.id}: {item.question}")
        for row in rows:
            rank = f"rank {row.first_rank}" if row.first_rank else f"not in the top {settings.top_k_retrieve}"
            state = "MISSED" if row.missed else "ok"
            typer.echo(f"  part {row.part} ({state}, {rank}): {', '.join(row.sources)}")
        typer.echo(f"  missed at k={k}: {missed or 'none'}")
    typer.echo(f"\nfull_coverage@{k}: {full}/{len(found)}")


@eval_app.command("explain")
def eval_explain(
    item_id: Annotated[str, typer.Argument(help="Golden item id, e.g. a-017")],
    golden: Annotated[Path, typer.Option(help="JSONL of golden items")] = Path(
        "eval/golden/candidates.jsonl"
    ),
    index_dir: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/index")] = None,
) -> None:
    """One item: first gold rank per retriever, plus the stored text of its gold pages. Zero LLM requests.

    Vector and BM25 rank the whole index. Hybrid is the real fused list, cut to TOP_K_RETRIEVE.
    """
    from citerag.eval.golden import load_golden
    from citerag.eval.metrics import is_hit
    from citerag.eval.retrieval import rank_line
    from citerag.index.retriever import HybridRetriever
    from citerag.index.store import open_store

    settings = get_settings()
    target = index_dir or settings.index_dir
    try:
        item = next((i for i in load_golden(golden) if i.id == item_id), None)
        if item is None:
            raise ValidationFailedError(f"{item_id} is not in {golden}")
        _check_index(target, settings)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc

    store = open_store(settings, target)
    try:
        retriever = HybridRetriever(store, settings)
        rankings = {
            "vector": retriever.vector_ranking(item.question),
            "bm25": retriever.bm25_ranking(item.question),
            "hybrid": retriever.candidates(item.question),
        }
        stored = retriever.all_chunks()
    finally:
        store.close()

    typer.echo(f"{item.id} ({item.type}): {item.question}")
    typer.echo(f"gold answer: {item.gold_answer}")
    typer.echo(f"gold sources: {[g.model_dump() for g in item.gold_sources] or 'no gold sources'}")
    for name, ranking in rankings.items():
        typer.echo(rank_line(name, ranking, item.gold_sources))
    positions = {
        name: {c.chunk_id: rank for rank, c in enumerate(ranking, start=1)}
        for name, ranking in rankings.items()
    }
    for chunk in (c for c in stored if is_hit(c.doc_name, c.page, item.gold_sources)):
        ranks = ", ".join(f"{name} {found.get(chunk.chunk_id, '-')}" for name, found in positions.items())
        typer.echo(f"\n--- {chunk.doc_name} p{chunk.page} {chunk.chunk_id[-4:]} ({ranks}) ---\n{chunk.text}")


def _retrieval_config(settings: Settings) -> dict:
    keys = (
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "top_k_retrieve",
        "top_k_context",
        "rerank_model",
    )
    return {key: getattr(settings, key) for key in keys}
