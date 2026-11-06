"""Cache interface, plus the no-op backend used when the cache is off or its file is unusable (F-CAC-10)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from citerag.cache.semantic_cache import CacheHit


class CacheStats(BaseModel):
    backend: str
    enabled: bool
    entries: int = 0


@runtime_checkable
class CacheBackend(Protocol):
    def lookup(
        self, q_text: str, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> CacheHit | None: ...

    def insert(
        self,
        q_text: str,
        q_emb: np.ndarray,
        response_json: str,
        namespace: str,
        mode: str = "standard",
        cost_usd: float = 0.0,
        now: float | None = None,
    ) -> None: ...

    def best_similarity(
        self, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> float | None: ...

    def invalidate_except(self, namespace: str) -> int: ...

    def clear(self) -> None: ...

    def stats(self) -> CacheStats: ...


class NoOpCache:
    """Always a miss, stores nothing."""

    def lookup(
        self, q_text: str, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> CacheHit | None:
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
        return None

    def best_similarity(
        self, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> float | None:
        return None

    def invalidate_except(self, namespace: str) -> int:
        return 0

    def clear(self) -> None:
        return None

    def stats(self) -> CacheStats:
        return CacheStats(backend="noop", enabled=False, entries=0)
