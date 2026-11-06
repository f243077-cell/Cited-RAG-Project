"""Index manifest: what the index was built with and which documents it holds. Plain JSON, no framework imports."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from citerag.cache.namespace import cache_namespace, corpus_version
from citerag.config import Settings
from citerag.errors import IndexMismatchError
from citerag.prompts import load_prompt

SCHEMA_VERSION = 1


class DocEntry(BaseModel):
    name: str
    sha256: str
    pages: int | None = None
    chunks: int
    images: int = 0
    size_bytes: int
    ingested_at: datetime


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    caption_model: str
    created_at: datetime
    docs: dict[str, DocEntry] = Field(default_factory=dict)


def manifest_path(index_dir: Path) -> Path:
    return index_dir / "store" / "manifest.json"


def new_manifest(settings: Settings, embedding_dim: int) -> Manifest:
    return Manifest(
        embedding_model=settings.embedding_model,
        embedding_dim=embedding_dim,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        caption_model=settings.model_vision,
        created_at=datetime.now(UTC),
    )


def load_manifest(index_dir: Path) -> Manifest | None:
    path = manifest_path(index_dir)
    if not path.exists():
        return None
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise IndexMismatchError(f"{path} is unreadable; delete the index and ingest again") from exc


def save_manifest(index_dir: Path, manifest: Manifest) -> None:
    """Write then rename, so a crash never leaves half a manifest."""
    path = manifest_path(index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp, path)


def check_manifest(manifest: Manifest, settings: Settings, embedding_dim: int | None = None) -> None:
    """Vectors from other settings would silently return garbage (F-ING-4). The caption model is not checked:
    a new one changes future captions, not the meaning of stored vectors."""
    expected = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if embedding_dim is not None:
        expected["embedding_dim"] = embedding_dim
    different = {k: v for k, v in expected.items() if getattr(manifest, k) != v}
    if different:
        found = {k: getattr(manifest, k) for k in different}
        raise IndexMismatchError(
            f"index was built with {found} but settings say {different}; "
            "restore the settings, or delete the index and ingest again"
        )


def current_namespace(settings: Settings, manifest: Manifest) -> str:
    """Fingerprint of everything that can change an answer (ARCHITECTURE section 6)."""
    index_settings = {
        "embedding_model": manifest.embedding_model,
        "chunk_size": manifest.chunk_size,
        "chunk_overlap": manifest.chunk_overlap,
    }
    retrieval = {
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_context": settings.top_k_context,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model,
        "min_top_similarity": settings.min_top_similarity,
    }
    corpus = corpus_version(list(manifest.docs), index_settings)
    return cache_namespace(corpus, settings.model_strong, load_prompt("answer").version, retrieval)
