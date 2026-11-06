"""Command line entry point. More commands arrive with their milestones (eval in M4, cache in M6)."""

from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import typer

from citerag.config import get_settings
from citerag.errors import CiteRAGError

app = typer.Typer(no_args_is_help=True)


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
) -> None:
    """Add documents to the index. Stop the API first: Chroma allows one writer process."""
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
