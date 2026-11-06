"""The real cache backend: the reference SemanticCache plus stats, miss similarity and failure handling.

A broken cache must never break a request (NF-3), so a bad file, a row deleted by `citerag cache clear` in another
process, or a stored vector of another width each count as a miss and get logged.
"""

from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np

from citerag.cache.base import CacheBackend, CacheStats, NoOpCache
from citerag.cache.semantic_cache import CacheHit, SemanticCache, _unit
from citerag.config import Settings

logger = logging.getLogger(__name__)
_BROKEN = (sqlite3.Error, TypeError, ValueError)
_SECONDS_PER_DAY = 86_400
_DEFAULT_DIM = 384  # sizes the empty matrix only: stored rows set the real width


class PersistentCache(SemanticCache):
    def stats(self) -> CacheStats:
        return CacheStats(backend="semantic", enabled=True, entries=len(self))

    def _reload_after(self, exc: Exception) -> None:
        logger.warning("semantic cache: %s; treating this as a miss", type(exc).__name__)
        try:
            self._load()
        except _BROKEN as again:
            logger.warning("semantic cache: reload failed with %s", type(again).__name__)

    def lookup(
        self, q_text: str, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> CacheHit | None:
        try:
            return super().lookup(q_text, q_emb, namespace, mode, now)
        except _BROKEN as exc:
            self._reload_after(exc)
            return None

    def insert(
        self,
        q_text: str,
        q_emb: np.ndarray,
        response_json: str,
        namespace: str,
        mode: str = "standard",
        cost_usd: float = 0.0,
        now: float | None = None,
    ) -> None:
        try:
            super().insert(q_text, q_emb, response_json, namespace, mode, cost_usd, now)
        except _BROKEN as exc:
            self._reload_after(exc)

    def best_similarity(
        self, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> float | None:
        """Closest entry that could have hit, or None when there is none. Logged on misses too (chart 6)."""
        now = time.time() if now is None else now
        with self._lock:
            valid = (self._namespaces == namespace) & (self._modes == mode) & (self._expires > now)
            if not valid.any():
                return None
            try:
                return float((self._mat[valid] @ _unit(q_emb)).max())
            except ValueError as exc:
                self._reload_after(exc)
                return None


def open_cache(settings: Settings) -> CacheBackend:
    if not settings.cache_enabled:
        return NoOpCache()
    path = settings.cache_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return PersistentCache(
            str(path),
            _DEFAULT_DIM,
            threshold=settings.cache_threshold,
            ttl_s=settings.cache_ttl_days * _SECONDS_PER_DAY,
            max_entries=settings.cache_max_entries,
        )
    except _BROKEN as exc:
        logger.error("semantic cache at %s is unusable (%s); running without it", path, type(exc).__name__)
        return NoOpCache()
