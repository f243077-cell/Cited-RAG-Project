"""Command line entry point. More commands arrive with their milestones (eval in M4, cache in M6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from citerag.config import Settings, UISettings, get_settings
from citerag.errors import CiteRAGError, ValidationFailedError

if TYPE_CHECKING:
    from citerag.eval.retrieval import RetrievalReport

app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True, help="Measure retrieval and answers.")
app.add_typer(eval_app, name="eval")


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
) -> None:
    """Add documents to the index. Stop the API first, or use --via-api: Chroma allows one writer process."""
    if via_api:
        _ingest_via_api(path)
        return
    from citerag.index.store import open_store  # loads the heavy frameworks, so not at import time
    from citerag.ingest.pipeline import Ingestor

    settings = get_settings()
    target = index_dir or settings.index_dir
    store = open_store(settings, target)
    try:
        reports, failures = Ingestor(settings, store, target).ingest_path(path)
    except CiteRAGError as exc:
        typer.echo(f"{exc.code}: {exc.message}")
        raise typer.Exit(1) from exc
    finally:
        store.close()
    for r in reports:
        typer.echo(
            f"{r.doc_name}: {r.status} ({r.pages or '-'} pages, {r.chunks} chunks, {r.duration_ms} ms)"
        )
    for f in failures:
        typer.echo(f"{f.name}: {f.code}: {f.message}")
    typer.echo(f"{len(reports)} file(s), {sum(r.chunks for r in reports)} chunk(s), {len(failures)} failed")
    if failures:
        raise typer.Exit(1)


def _ingest_via_api(path: Path) -> None:
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
            for r in client.upload([(file.name, file.read_bytes())]):
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
