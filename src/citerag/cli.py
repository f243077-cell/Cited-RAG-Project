"""Command line entry point. Commands arrive with their milestones (ingest in M1, eval in M4, cache in M6)."""

from __future__ import annotations

from importlib.metadata import version as package_version

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """CiteRAG: cited RAG over your own documents."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(package_version("citerag"))
