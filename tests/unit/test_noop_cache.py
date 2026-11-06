import numpy as np

from citerag.cache.base import CacheBackend, CacheStats, NoOpCache


def test_noop_cache_never_hits_and_never_stores():
    cache = NoOpCache()
    assert isinstance(cache, CacheBackend)
    v = np.ones(4, dtype=np.float32)
    cache.insert("q", v, "{}", "ns", "standard", 0.5)
    assert cache.lookup("q", v, "ns") is None
    assert cache.invalidate_except("other") == 0
    cache.clear()
    assert cache.stats() == CacheStats(backend="noop", enabled=False, entries=0)
