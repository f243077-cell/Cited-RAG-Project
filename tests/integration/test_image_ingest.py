"""Captions through real Chroma and BM25 with a fake captioner: image chunks are searchable and citable."""

import pytest

from citerag.errors import BudgetExceededError, LLMUnavailableError, StructuredOutputError
from citerag.index.retriever import HybridRetriever
from citerag.index.store import IndexStore
from citerag.ingest.manifest import current_namespace
from citerag.ingest.pipeline import Ingestor
from citerag.llm.gateway import StructuredResult
from citerag.schemas import ImageCaption
from tests.helpers import HashEmbedding

PICTURE = b"\x89PNG\r\n\x1a\n" + b"x" * 6000
OTHER = b"\x89PNG\r\n\x1a\n" + b"y" * 6000
CAPTION = ImageCaption(
    caption="A whiteboard about the backup schedule", visible_text="Nightly at 02:00", image_type="photo"
)


class Captioner:
    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def __call__(self, data, media_type):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return StructuredResult[ImageCaption](
            parsed=CAPTION, model="m", requests=1, input_tokens=5, output_tokens=7, latency_ms=1
        )


@pytest.fixture
def store(settings):
    opened = IndexStore(settings.index_dir, HashEmbedding())
    yield opened
    opened.close()


def ingestor(settings, store, captioner=None):
    return Ingestor(settings, store, settings.index_dir, captioner=captioner)


def test_a_standalone_image_becomes_a_citable_chunk(settings, store):
    report = ingestor(settings, store, Captioner()).ingest_bytes("board.png", PICTURE)
    assert (report.chunks, report.images_found, report.images_captioned) == (1, 1, 1)
    assert report.caption_requests == 1 and report.caption_output_tokens == 7
    chunk = HybridRetriever(store, settings).retrieve("backup schedule nightly")[0]
    assert chunk.kind == "image" and chunk.doc_name == "board.png" and chunk.page is None
    assert chunk.image_path.endswith(".png") and "Nightly at 02:00" in chunk.text


def test_the_same_image_is_never_paid_for_twice(settings, store):
    captioner = Captioner()
    ingestor(settings, store, captioner).ingest_bytes("board.png", PICTURE)
    second = ingestor(settings, store, captioner).ingest_bytes("copy.png", PICTURE)
    assert captioner.calls == 1 and second.caption_requests == 0 and second.chunks == 1


def test_without_a_captioner_the_image_is_skipped_and_costs_nothing(settings, store):
    report = ingestor(settings, store).ingest_bytes("board.png", PICTURE)
    assert (report.chunks, report.images_found, report.images_skipped) == (0, 1, 1)


@pytest.mark.parametrize("error", [BudgetExceededError("limit"), LLMUnavailableError("down")])
def test_quota_or_provider_trouble_keeps_the_document(settings, store, error):
    report = ingestor(settings, store, Captioner(error)).ingest_bytes("board.png", PICTURE)
    assert report.status == "ingested" and report.images_skipped == 1 and report.chunks == 0


def test_a_later_ingest_fills_in_a_skipped_caption(settings, store):
    first = ingestor(settings, store, Captioner(LLMUnavailableError("down")))
    first.ingest_bytes("board.png", PICTURE)
    second = ingestor(settings, store, Captioner())
    before = current_namespace(settings, second.manifest)
    report = second.ingest_bytes("board.png", PICTURE)
    assert report.status == "unchanged" and report.images_captioned == 1 and report.chunks == 1
    assert current_namespace(settings, second.manifest) != before  # a new chunk can change answers
    assert second.ingest_bytes("board.png", PICTURE).caption_requests == 0


def test_a_bad_caption_skips_that_image_and_tries_the_next_document(settings, store):
    captioner = Captioner(StructuredOutputError("bad"))
    run = ingestor(settings, store, captioner)
    assert run.ingest_bytes("board.png", PICTURE).images_skipped == 1
    assert run.ingest_bytes("other.png", OTHER).images_captioned == 1


def test_delete_removes_image_chunks(settings, store):
    run = ingestor(settings, store, Captioner())
    report = run.ingest_bytes("board.png", PICTURE)
    run.delete(report.doc_id)
    assert store.count() == 0


def test_an_icon_sized_standalone_file_is_not_captioned(settings, store):
    captioner = Captioner()
    report = ingestor(settings, store, captioner).ingest_bytes("dot.png", b"\x89PNG\r\n\x1a\n" + b"x" * 10)
    assert captioner.calls == 0 and report.images_skipped == 1
