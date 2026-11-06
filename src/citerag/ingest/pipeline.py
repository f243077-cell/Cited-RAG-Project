"""Text ingestion (ARCHITECTURE section 7). Makes no LLM call; image captioning arrives in M5 (F-ING-6)."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from citerag.cache.base import CacheBackend, NoOpCache
from citerag.config import Settings
from citerag.errors import CiteRAGError, NotFoundError
from citerag.index.store import IndexStore
from citerag.ingest.chunking import chunk_pages
from citerag.ingest.loaders import compute_doc_id, extension, load_pages, sanitize_name, validate_file
from citerag.ingest.manifest import (
    DocEntry,
    check_manifest,
    current_namespace,
    load_manifest,
    new_manifest,
    save_manifest,
)
from citerag.schemas import ErrorCode, IngestReport

logger = logging.getLogger(__name__)
_SKIPPED_NAMES = {"SOURCES.md"}  # the licence table scripts/fetch_sample.py writes next to the corpus


@dataclass(frozen=True)
class IngestFailure:
    name: str
    code: ErrorCode
    message: str


class Ingestor:
    def __init__(
        self, settings: Settings, store: IndexStore, index_dir: Path, cache: CacheBackend | None = None
    ) -> None:
        self._settings, self._store, self._index_dir = settings, store, index_dir
        self._cache = cache or NoOpCache()
        self._uploads = settings.data_dir / "uploads"
        existing = load_manifest(index_dir)
        if existing is not None:
            check_manifest(existing, settings, store.embedding_dim)
        self._manifest = existing or new_manifest(settings, store.embedding_dim)

    def ingest_bytes(self, name: str, data: bytes) -> IngestReport:
        """Everything that can fail runs before the first write; new chunks go in before old ones go out."""
        started = time.perf_counter()
        doc_name = sanitize_name(name)
        kind = validate_file(doc_name, data, self._settings.max_upload_mb)
        doc_id = compute_doc_id(data)

        known = self._manifest.docs.get(doc_id)
        if known is not None:
            return IngestReport(
                doc_id=doc_id,
                doc_name=known.name,
                status="unchanged",
                pages=known.pages,
                chunks=known.chunks,
                duration_ms=_elapsed_ms(started),
            )

        loaded = load_pages(kind, data)
        nodes = chunk_pages(
            loaded.pages, doc_id, doc_name, self._settings.chunk_size, self._settings.chunk_overlap
        )
        previous = [old for old, entry in self._manifest.docs.items() if entry.name == doc_name]
        if nodes:
            self._store.add_nodes(nodes)
        for old_id in previous:
            self._store.delete_doc(old_id)
        self._store.persist()

        self._uploads.mkdir(parents=True, exist_ok=True)
        suffix = extension(doc_name)
        (self._uploads / f"{doc_id}{suffix}").write_bytes(data)
        for old_id in previous:
            (self._uploads / f"{old_id}{extension(self._manifest.docs[old_id].name)}").unlink(missing_ok=True)
            del self._manifest.docs[old_id]

        self._manifest.docs[doc_id] = DocEntry(
            name=doc_name,
            sha256=hashlib.sha256(data).hexdigest(),
            pages=loaded.page_count,
            chunks=len(nodes),
            size_bytes=len(data),
            ingested_at=datetime.now(UTC),
        )
        save_manifest(self._index_dir, self._manifest)
        self._cache.invalidate_except(current_namespace(self._settings, self._manifest))

        standalone_images = 1 if kind == "image" else 0  # counted and skipped until captioning exists (M5)
        return IngestReport(
            doc_id=doc_id,
            doc_name=doc_name,
            status="replaced" if previous else "ingested",
            pages=loaded.page_count,
            chunks=len(nodes),
            images_found=standalone_images,
            images_skipped=standalone_images,
            duration_ms=_elapsed_ms(started),
        )

    def ingest_path(self, path: Path) -> tuple[list[IngestReport], list[IngestFailure]]:
        """A file or a folder (recursive). One bad file does not stop the others."""
        if not path.exists():
            raise NotFoundError(f"{path} does not exist")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        reports: list[IngestReport] = []
        failures: list[IngestFailure] = []
        for file in files:
            if path.is_dir() and file.name in _SKIPPED_NAMES:
                continue
            try:
                reports.append(self.ingest_bytes(file.name, file.read_bytes()))
            except CiteRAGError as exc:
                logger.warning("skipped %s: %s", file.name, exc.message)
                failures.append(IngestFailure(file.name, exc.code, exc.message))
        return reports, failures


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
