from datetime import UTC, datetime

import pytest

from citerag.errors import IndexMismatchError
from citerag.ingest.manifest import (
    DocEntry,
    check_manifest,
    current_namespace,
    load_manifest,
    manifest_path,
    new_manifest,
    save_manifest,
)


def _entry():
    return DocEntry(
        name="a.pdf",
        sha256="f" * 64,
        pages=2,
        chunks=3,
        size_bytes=10,
        ingested_at=datetime(2026, 11, 6, tzinfo=UTC),
    )


def test_missing_manifest_is_none(tmp_path):
    assert load_manifest(tmp_path) is None


def test_round_trip_and_no_temp_file_left(settings, tmp_path):
    manifest = new_manifest(settings, embedding_dim=384)
    manifest.docs["d1"] = _entry()
    save_manifest(tmp_path, manifest)
    assert load_manifest(tmp_path) == manifest
    assert [p.name for p in manifest_path(tmp_path).parent.iterdir()] == ["manifest.json"]


def test_manifest_records_the_documented_fields(settings):
    data = new_manifest(settings, embedding_dim=384).model_dump()
    assert data["schema_version"] == 1 and data["embedding_dim"] == 384
    assert data["embedding_model"] == settings.embedding_model
    assert data["chunk_size"] == 512 and data["chunk_overlap"] == 64
    assert data["caption_model"] == settings.model_vision and data["docs"] == {}


def test_corrupt_manifest_is_a_typed_error(tmp_path):
    path = manifest_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(IndexMismatchError):
        load_manifest(tmp_path)


@pytest.mark.parametrize(
    "change", [{"embedding_model": "other/model"}, {"chunk_size": 256}, {"chunk_overlap": 32}]
)
def test_changed_settings_fail_with_index_mismatch(settings, change):
    manifest = new_manifest(settings, embedding_dim=384)
    check_manifest(manifest, settings)
    with pytest.raises(IndexMismatchError):
        check_manifest(manifest, settings.model_copy(update=change))


def test_changed_embedding_dim_fails(settings):
    manifest = new_manifest(settings, embedding_dim=384)
    check_manifest(manifest, settings, embedding_dim=384)
    with pytest.raises(IndexMismatchError):
        check_manifest(manifest, settings, embedding_dim=128)


def test_a_new_vision_model_does_not_invalidate_the_index(settings):
    manifest = new_manifest(settings, embedding_dim=384)
    check_manifest(manifest, settings.model_copy(update={"model_vision": "other/vision:free"}))


def test_namespace_follows_corpus_answer_model_and_retrieval(settings):
    manifest = new_manifest(settings, embedding_dim=384)
    base = current_namespace(settings, manifest)
    assert base == current_namespace(settings, manifest)
    assert base != current_namespace(settings.model_copy(update={"model_strong": "x:free"}), manifest)
    assert base != current_namespace(settings.model_copy(update={"top_k_context": 8}), manifest)
    manifest.docs["d1"] = _entry()
    assert base != current_namespace(settings, manifest)
