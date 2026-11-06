"""The real cache backend: the reference SemanticCache plus stats, miss similarity and failure handling."""

import sqlite3

import numpy as np

from citerag.cache.base import CacheBackend, NoOpCache
from citerag.cache.store import PersistentCache, open_cache

DIM = 4


def vector(*values):
    return np.array(values, dtype=np.float32)


def make(tmp_path, **kw):
    return PersistentCache(str(tmp_path / "c.sqlite"), DIM, threshold=0.9, **kw)


def test_it_meets_the_backend_interface_and_reports_entries(tmp_path):
    cache = make(tmp_path)
    assert isinstance(cache, CacheBackend)
    cache.insert("q", vector(1, 0, 0, 0), "{}", "ns")
    stats = cache.stats()
    assert (stats.backend, stats.enabled, stats.entries) == ("semantic", True, 1)


def test_best_similarity_sees_a_miss_but_only_in_this_namespace_and_mode(tmp_path):
    cache = make(tmp_path)
    cache.insert("q", vector(1, 0, 0, 0), "{}", "ns")
    query = vector(1, 1, 0, 0)
    assert cache.lookup("other", query, "ns") is None
    assert abs(cache.best_similarity(query, "ns") - 0.7071) < 0.001
    assert cache.best_similarity(query, "other") is None
    assert cache.best_similarity(query, "ns", mode="deep") is None


def test_best_similarity_skips_expired_entries_and_handles_an_empty_cache(tmp_path):
    cache = make(tmp_path)
    assert cache.best_similarity(vector(1, 0, 0, 0), "ns") is None
    cache.insert("q", vector(1, 0, 0, 0), "{}", "ns", now=0.0)
    assert cache.best_similarity(vector(1, 0, 0, 0), "ns", now=10 * 86400) is None


def test_a_row_removed_by_another_process_is_a_miss_not_a_crash(tmp_path):
    cache = make(tmp_path)
    cache.insert("q", vector(1, 0, 0, 0), "{}", "ns")
    other = PersistentCache(str(tmp_path / "c.sqlite"), DIM)
    other.clear()
    assert cache.lookup("q", vector(1, 0, 0, 0), "ns") is None
    assert cache.stats().entries == 0


def test_an_embedding_of_another_width_is_a_miss_not_a_crash(tmp_path):
    cache = make(tmp_path)
    cache.insert("q", vector(1, 0, 0, 0), "{}", "ns")
    assert cache.lookup("q", np.ones(8, dtype=np.float32), "ns") is None
    assert cache.best_similarity(np.ones(8, dtype=np.float32), "ns") is None


def test_a_failed_insert_does_not_stop_the_request(tmp_path):
    cache = make(tmp_path)
    cache._db.close()
    cache.insert("q", vector(1, 0, 0, 0), "{}", "ns")


def test_open_cache_builds_the_file_under_the_data_folder(settings):
    cache = open_cache(settings)
    assert isinstance(cache, PersistentCache)
    assert settings.cache_db_path.exists()
    assert cache.threshold == settings.cache_threshold and cache.max_entries == settings.cache_max_entries
    assert cache.ttl_s == settings.cache_ttl_days * 86400


def test_open_cache_is_a_noop_when_turned_off(settings):
    off = settings.model_copy(update={"cache_enabled": False})
    assert isinstance(open_cache(off), NoOpCache)
    assert not off.cache_db_path.exists()


def test_a_corrupt_file_means_running_without_the_cache(settings):
    settings.cache_db_path.parent.mkdir(parents=True)
    settings.cache_db_path.write_bytes(b"this is not a database, not at all, " * 40)
    assert isinstance(open_cache(settings), NoOpCache)


def test_rows_of_mixed_width_mean_running_without_the_cache(settings):
    open_cache(settings)._db.close()
    db = sqlite3.connect(settings.cache_db_path)
    for name, width in (("a", 4), ("b", 8)):
        blob = np.ones(width, dtype=np.float32).tobytes()
        db.execute(
            "INSERT INTO entries (query, embedding, response_json, namespace, mode, created_at, expires_at,"
            " last_hit_at) VALUES (?, ?, '{}', 'ns', 'standard', 0, 9e9, 0)",
            (name, blob),
        )
    db.commit()
    db.close()
    assert isinstance(open_cache(settings), NoOpCache)


def test_noop_has_no_similarity_to_report():
    assert NoOpCache().best_similarity(vector(1, 0, 0, 0), "ns") is None
