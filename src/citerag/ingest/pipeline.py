"""Ingestion (ARCHITECTURE section 7). Image captions are the only LLM calls, and only with a captioner."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core.schema import TextNode  # noqa: E402

from citerag.cache.base import CacheBackend, NoOpCache
from citerag.config import Settings
from citerag.errors import (
    BudgetExceededError,
    CiteRAGError,
    LLMUnavailableError,
    NotFoundError,
    StructuredOutputError,
)
from citerag.index.store import IndexStore
from citerag.ingest.chunking import chunk_pages, image_node
from citerag.ingest.images import FoundImage, ImageBatch, ImageStore, caption_text, pdf_images
from citerag.ingest.loaders import (
    SKIPPED_NAMES,
    compute_doc_id,
    extension,
    load_pages,
    sanitize_name,
    validate_file,
)
from citerag.ingest.manifest import (
    DocEntry,
    Manifest,
    check_manifest,
    current_namespace,
    load_manifest,
    new_manifest,
    save_manifest,
)
from citerag.multimodal.vision import Captioner
from citerag.schemas import ErrorCode, IngestReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestFailure:
    name: str
    code: ErrorCode
    message: str


@dataclass
class _Captions:
    nodes: list[TextNode] = field(default_factory=list)
    captioned: int = 0
    skipped: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Ingestor:
    def __init__(
        self,
        settings: Settings,
        store: IndexStore,
        index_dir: Path,
        cache: CacheBackend | None = None,
        captioner: Captioner | None = None,
    ) -> None:
        self._settings, self._store, self._index_dir = settings, store, index_dir
        self._cache = cache or NoOpCache()
        self._uploads = settings.data_dir / "uploads"
        self._captioner = captioner
        self._images = ImageStore(settings.data_dir)
        existing = load_manifest(index_dir)
        if existing is not None:
            check_manifest(existing, settings, store.embedding_dim)
        self._manifest = existing or new_manifest(settings, store.embedding_dim)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def _find_images(self, kind: str, doc_name: str, data: bytes) -> ImageBatch:
        s = self._settings
        if kind == "pdf":
            return pdf_images(data, s.min_image_px, s.min_image_bytes, s.max_images_per_doc)
        if kind != "image" or len(data) < s.min_image_bytes:
            return ImageBatch(1 if kind == "image" else 0, [])
        media_type = "image/png" if extension(doc_name) == ".png" else "image/jpeg"
        sha = hashlib.sha256(data).hexdigest()
        return ImageBatch(1, [FoundImage(sha, data, media_type, None, len(data))])

    def _caption(self, doc_id: str, doc_name: str, images: list[FoundImage]) -> _Captions:
        """Runs before any write. Budget or provider trouble stops the calls but never fails the document:
        its text still goes in, the images count as skipped, and a later ingest fills them in."""
        done = _Captions()
        stopped = self._captioner is None
        for image in images:
            caption = self._images.cached_caption(image.sha256)
            if caption is None and not stopped:
                try:
                    result = self._captioner(image.data, image.media_type)
                except (BudgetExceededError, LLMUnavailableError) as exc:
                    logger.warning("stopping captions for %s: %s", doc_name, exc.code)
                    stopped = True
                except StructuredOutputError:
                    logger.warning("no usable caption for one image in %s", doc_name)
                else:
                    caption = result.parsed
                    self._images.save_caption(image.sha256, caption)
                    done.captioned += 1
                    done.requests += result.requests
                    done.input_tokens += result.input_tokens
                    done.output_tokens += result.output_tokens
                    done.cost_usd += result.shadow_usd
            if caption is None:
                done.skipped += 1
                continue
            path = self._images.save_image(image)
            text = caption_text(caption)
            done.nodes.append(image_node(doc_id, doc_name, image.sha256, text, image.page, path.as_posix()))
        return done

    def _add_missing_images(self, doc_id: str, known: DocEntry, kind: str, data: bytes) -> _Captions | None:
        """An unchanged document can still gain captions that failed or were skipped earlier."""
        indexed = set(self._store.chunk_ids(doc_id))
        batch = self._find_images(kind, known.name, data)
        missing = [i for i in batch.kept if f"{doc_id}-img-{i.sha256[:8]}" not in indexed]
        if not missing:
            return None
        done = self._caption(doc_id, known.name, missing)
        if done.nodes:
            self._store.add_nodes(done.nodes)
            self._store.persist()
            known.chunks += len(done.nodes)
            known.images += len(done.nodes)
            save_manifest(self._index_dir, self._manifest)
            self._cache.invalidate_except(current_namespace(self._settings, self._manifest))
        return done

    def ingest_bytes(self, name: str, data: bytes) -> IngestReport:
        """Everything that can fail runs before the first write; new chunks go in before old ones go out."""
        started = time.perf_counter()
        doc_name = sanitize_name(name)
        kind = validate_file(doc_name, data, self._settings.max_upload_mb)
        doc_id = compute_doc_id(data)

        known = self._manifest.docs.get(doc_id)
        if known is not None:
            done = self._add_missing_images(doc_id, known, kind, data) if self._captioner else None
            return IngestReport(
                doc_id=doc_id,
                doc_name=known.name,
                status="unchanged",
                pages=known.pages,
                chunks=known.chunks,
                images_found=known.images,
                duration_ms=_elapsed_ms(started),
                **(_caption_fields(done) if done else {}),
            )

        loaded = load_pages(kind, data)
        nodes = chunk_pages(
            loaded.pages, doc_id, doc_name, self._settings.chunk_size, self._settings.chunk_overlap
        )
        batch = self._find_images(kind, doc_name, data)
        captions = self._caption(doc_id, doc_name, batch.kept)
        nodes = [*nodes, *captions.nodes]
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
            images=len(captions.nodes),
            size_bytes=len(data),
            ingested_at=datetime.now(UTC),
        )
        save_manifest(self._index_dir, self._manifest)
        self._cache.invalidate_except(current_namespace(self._settings, self._manifest))

        return IngestReport(
            doc_id=doc_id,
            doc_name=doc_name,
            status="replaced" if previous else "ingested",
            pages=loaded.page_count,
            chunks=len(nodes),
            images_found=batch.found,
            duration_ms=_elapsed_ms(started),
            **_caption_fields(captions, batch.found),
        )

    def delete(self, doc_id: str) -> None:
        entry = self._manifest.docs.get(doc_id)
        if entry is None:
            raise NotFoundError(f"no document {doc_id}")
        self._store.delete_doc(doc_id)
        self._store.persist()
        (self._uploads / f"{doc_id}{extension(entry.name)}").unlink(missing_ok=True)
        del self._manifest.docs[doc_id]
        save_manifest(self._index_dir, self._manifest)
        self._cache.invalidate_except(current_namespace(self._settings, self._manifest))

    def ingest_path(self, path: Path) -> tuple[list[IngestReport], list[IngestFailure]]:
        """A file or a folder (recursive). One bad file does not stop the others."""
        if not path.exists():
            raise NotFoundError(f"{path} does not exist")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        reports: list[IngestReport] = []
        failures: list[IngestFailure] = []
        for file in files:
            if path.is_dir() and file.name in SKIPPED_NAMES:
                continue
            try:
                reports.append(self.ingest_bytes(file.name, file.read_bytes()))
            except CiteRAGError as exc:
                logger.warning("skipped %s: %s", file.name, exc.message)
                failures.append(IngestFailure(file.name, exc.code, exc.message))
        return reports, failures


def _caption_fields(done: _Captions, found: int | None = None) -> dict[str, int | float]:
    """Images filtered out (icons, covers, over the cap) count as skipped too, when `found` is known."""
    skipped = done.skipped if found is None else found - len(done.nodes)
    return {
        "images_captioned": done.captioned,
        "images_skipped": skipped,
        "caption_requests": done.requests,
        "caption_input_tokens": done.input_tokens,
        "caption_output_tokens": done.output_tokens,
        "caption_cost_usd": done.cost_usd,
    }


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
