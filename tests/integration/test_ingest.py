"""Ingestion end to end: real Chroma and docstore in a temp folder, HashEmbedding instead of the model download."""

import pytest
from llama_index.core import Settings as LlamaSettings

from citerag.cache.base import NoOpCache
from citerag.errors import FileTooLargeError, IndexMismatchError, InvalidFileTypeError, NotFoundError
from citerag.index.store import IndexStore
from citerag.ingest.manifest import load_manifest
from citerag.ingest.pipeline import Ingestor
from tests.helpers import HashEmbedding, hash_vector
from tests.pdf_fixture import make_pdf

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PDF = make_pdf(["Items can be returned within thirty days.", "", "Shipping takes three working days."])


class SpyCache(NoOpCache):
    def __init__(self):
        self.namespaces = []

    def invalidate_except(self, namespace):
        self.namespaces.append(namespace)
        return 0


@pytest.fixture
def workspace(settings):
    store = IndexStore(settings.index_dir, HashEmbedding())
    yield settings, store
    store.close()


@pytest.fixture
def ingestor(workspace):
    settings, store = workspace
    return Ingestor(settings, store, settings.index_dir)


def _reopen(settings):
    return IndexStore(settings.index_dir, HashEmbedding())


def test_first_ingest_reports_counts_and_writes_everything(workspace, ingestor):
    settings, store = workspace
    report = ingestor.ingest_bytes("policy.pdf", PDF)
    assert (report.status, report.pages, report.chunks) == ("ingested", 3, 2)
    assert report.doc_name == "policy.pdf" and report.duration_ms >= 0
    assert report.caption_requests == 0
    assert store.chunk_ids(report.doc_id) == [f"{report.doc_id}-0000", f"{report.doc_id}-0001"]
    assert store.docstore_count() == 2
    entry = load_manifest(settings.index_dir).docs[report.doc_id]
    assert (entry.name, entry.pages, entry.chunks, entry.size_bytes) == ("policy.pdf", 3, 2, len(PDF))
    assert (settings.data_dir / "uploads" / f"{report.doc_id}.pdf").read_bytes() == PDF


def test_reingest_identical_bytes_is_a_no_op(workspace, ingestor):
    settings, store = workspace
    first = ingestor.ingest_bytes("policy.pdf", PDF)
    before = load_manifest(settings.index_dir)
    again = ingestor.ingest_bytes("policy.pdf", PDF)
    renamed = ingestor.ingest_bytes("copy-of-policy.pdf", PDF)
    assert again.status == "unchanged" and again.doc_id == first.doc_id and again.chunks == first.chunks
    assert renamed.status == "unchanged"  # F-ING-2: the bytes decide, not the name
    assert store.count() == 2 and load_manifest(settings.index_dir) == before


def test_changed_file_with_the_same_name_replaces_old_chunks(workspace, ingestor):
    settings, store = workspace
    old = ingestor.ingest_bytes("notes.md", b"Refunds take five days.")
    new = ingestor.ingest_bytes("notes.md", b"Refunds take ten days now.")
    assert new.status == "replaced" and new.doc_id != old.doc_id
    assert store.chunk_ids(old.doc_id) == [] and len(store.chunk_ids(new.doc_id)) == 1
    assert store.docstore_count() == 1
    assert list(load_manifest(settings.index_dir).docs) == [new.doc_id]
    assert not (settings.data_dir / "uploads" / f"{old.doc_id}.md").exists()


def test_other_documents_survive_a_replace(workspace, ingestor):
    _, store = workspace
    keep = ingestor.ingest_bytes("keep.md", b"Warranty covers two years.")
    ingestor.ingest_bytes("notes.md", b"Version one.")
    ingestor.ingest_bytes("notes.md", b"Version two.")
    assert len(store.chunk_ids(keep.doc_id)) == 1 and store.count() == 2


def test_state_survives_a_restart(workspace):
    settings, store = workspace
    report = Ingestor(settings, store, settings.index_dir).ingest_bytes("policy.pdf", PDF)
    store.close()
    second = _reopen(settings)
    try:
        assert second.docstore_count() == 2 and second.count() == 2
        assert (
            Ingestor(settings, second, settings.index_dir).ingest_bytes("policy.pdf", PDF).status
            == "unchanged"
        )
        changed = make_pdf(["Items can be returned within sixty days."])
        result = Ingestor(settings, second, settings.index_dir).ingest_bytes("policy.pdf", changed)
        assert result.status == "replaced" and second.chunk_ids(report.doc_id) == []
        assert second.docstore_count() == 1
    finally:
        second.close()


def test_chroma_keeps_the_real_doc_id_and_filters_work(workspace, ingestor):
    _, store = workspace
    report = ingestor.ingest_bytes("policy.pdf", PDF)
    meta = store.chunk_metadata(report.doc_id)[0]
    assert (meta["doc_id"], meta["doc_name"], meta["page"], meta["kind"]) == (
        report.doc_id,
        "policy.pdf",
        1,
        "text",
    )


def test_indexed_sentence_queried_with_itself_scores_about_one(workspace, ingestor):
    _, store = workspace
    ingestor.ingest_bytes("policy.pdf", PDF)
    assert store.top_cosine(hash_vector("Items can be returned within thirty days.")) == pytest.approx(
        1.0, abs=0.01
    )


def test_failure_before_writing_leaves_the_index_unchanged(workspace, ingestor):
    settings, store = workspace
    ingestor.ingest_bytes("keep.md", b"Stays here.")
    before = load_manifest(settings.index_dir)
    for name, data in (("bad.pdf", b"MZ exe"), ("evil.pdf", b"%PDF-1.4 garbage")):
        with pytest.raises(InvalidFileTypeError):
            ingestor.ingest_bytes(name, data)
    with pytest.raises(FileTooLargeError):
        ingestor.ingest_bytes("big.txt", b"x" * (settings.max_upload_mb * 1024 * 1024 + 1))
    assert load_manifest(settings.index_dir) == before and store.count() == 1


def test_embedding_failure_keeps_the_old_version(workspace, ingestor, monkeypatch):
    settings, store = workspace
    old = ingestor.ingest_bytes("notes.md", b"Version one.")

    def boom(*args, **kwargs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(type(store.embed_model), "get_text_embedding_batch", boom)
    with pytest.raises(RuntimeError):
        ingestor.ingest_bytes("notes.md", b"Version two.")
    assert len(store.chunk_ids(old.doc_id)) == 1 and list(load_manifest(settings.index_dir).docs) == [
        old.doc_id
    ]


def test_standalone_image_is_recorded_without_chunks_until_m5(workspace, ingestor):
    _, store = workspace
    report = ingestor.ingest_bytes("chart.png", PNG)
    assert (report.status, report.chunks, report.images_found, report.images_skipped) == ("ingested", 0, 1, 1)
    assert report.caption_requests == 0 and store.count() == 0


def test_changed_chunk_size_is_refused(workspace, ingestor):
    settings, store = workspace
    ingestor.ingest_bytes("a.md", b"Some text.")
    changed = settings.model_copy(update={"chunk_size": 256})
    with pytest.raises(IndexMismatchError):
        Ingestor(changed, store, settings.index_dir)


def test_changed_embedder_dimension_is_refused(workspace, ingestor):
    settings, store = workspace
    ingestor.ingest_bytes("a.md", b"Some text.")
    store.embedding_dim = 384
    with pytest.raises(IndexMismatchError):
        Ingestor(settings, store, settings.index_dir)


def test_namespace_is_recomputed_and_cache_invalidated_on_each_change(workspace):
    settings, store = workspace
    cache = SpyCache()
    ingestor = Ingestor(settings, store, settings.index_dir, cache)
    ingestor.ingest_bytes("a.md", b"One.")
    ingestor.ingest_bytes("a.md", b"One.")  # unchanged: nothing to invalidate
    ingestor.ingest_bytes("a.md", b"Two.")
    assert len(cache.namespaces) == 2 and cache.namespaces[0] != cache.namespaces[1]


def test_folder_ingest_continues_past_bad_files(workspace, ingestor, tmp_path):
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.md").write_bytes(b"Alpha.")
    (folder / "sub" / "b.txt").write_bytes(b"Beta.")
    (folder / "bad.pdf").write_bytes(b"not a pdf")
    (folder / "SOURCES.md").write_bytes(b"Licence table, not a document.")
    reports, failures = ingestor.ingest_path(folder)
    assert sorted(r.doc_name for r in reports) == ["a.md", "b.txt"]
    assert [(f.name, f.code) for f in failures] == [("bad.pdf", "invalid_file_type")]


def test_single_file_and_missing_path(ingestor, tmp_path):
    path = tmp_path / "one.md"
    path.write_bytes(b"Alpha.")
    assert ingestor.ingest_path(path)[0][0].doc_name == "one.md"
    with pytest.raises(NotFoundError):
        ingestor.ingest_path(tmp_path / "missing")


def test_llamaindex_llm_is_disabled(workspace):
    assert type(LlamaSettings.llm).__name__ == "MockLLM"  # F-ING-6: no LLM path through LlamaIndex


def test_changed_standalone_image_replaces_without_chunks(workspace, ingestor):
    settings, _ = workspace
    old = ingestor.ingest_bytes("chart.png", PNG)
    new = ingestor.ingest_bytes("chart.png", PNG + b"\x01")
    assert new.status == "replaced"
    assert list(load_manifest(settings.index_dir).docs) == [new.doc_id] and new.doc_id != old.doc_id
