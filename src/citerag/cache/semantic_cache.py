"""Semantic cache: numpy cosine search over cached *standalone* questions, SQLite for persistence."""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_NEGATIONS = {
    "not",
    "no",
    "never",
    "without",
    "cannot",
    "none",
    "can't",
    "don't",
    "doesn't",
    "isn't",
    "won't",
}


def guard_ok(new_query: str, cached_query: str) -> bool:
    """Embeddings see '30 days' and '60 days' as near-identical. Reject a hit if numbers or negation differ."""
    if sorted(_NUM.findall(new_query)) != sorted(_NUM.findall(cached_query)):
        return False

    def negated(q: str) -> int:
        return len(_NEGATIONS & set(re.findall(r"[\w']+", q.lower()))) % 2

    return negated(new_query) == negated(cached_query)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


@dataclass(frozen=True)
class CacheHit:
    entry_id: int
    similarity: float
    query: str
    response_json: str
    cost_usd: float  # what the original (uncached) answer cost


class SemanticCache:
    def __init__(
        self,
        db_path: str,
        dim: int,
        threshold: float = 0.92,
        ttl_s: float = 7 * 86400,
        max_entries: int = 10_000,
    ) -> None:
        self.dim, self.threshold, self.ttl_s, self.max_entries = dim, threshold, ttl_s, max_entries
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS entries (
                 id INTEGER PRIMARY KEY, query TEXT NOT NULL, embedding BLOB NOT NULL,
                 response_json TEXT NOT NULL, namespace TEXT NOT NULL, mode TEXT NOT NULL,
                 created_at REAL NOT NULL, expires_at REAL NOT NULL, last_hit_at REAL NOT NULL,
                 hit_count INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
                 UNIQUE (query, namespace, mode))"""
        )
        self._db.commit()
        self._load()

    # -- memory mirror of the table: one row per entry, aligned by position ----------
    def _load(self) -> None:
        rows = self._db.execute(
            "SELECT id, query, embedding, namespace, mode, expires_at FROM entries ORDER BY id"
        ).fetchall()
        self._ids = [r[0] for r in rows]
        self._queries = [r[1] for r in rows]
        self._mat = (
            np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            if rows
            else np.empty((0, self.dim), dtype=np.float32)
        )
        self._namespaces = np.array([r[3] for r in rows], dtype="U64")
        self._modes = np.array([r[4] for r in rows], dtype="U16")
        self._expires = np.array([r[5] for r in rows], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._ids)

    def lookup(
        self, q_text: str, q_emb: np.ndarray, namespace: str, mode: str = "standard", now: float | None = None
    ) -> CacheHit | None:
        now = time.time() if now is None else now
        with self._lock:
            if not self._ids:
                return None
            scores = self._mat @ _unit(q_emb)  # unit vectors, so dot product == cosine similarity
            valid = (self._namespaces == namespace) & (self._modes == mode) & (self._expires > now)
            scores = np.where(valid, scores, -1.0)
            for i in np.argsort(scores)[::-1][:3]:  # best few candidates; first one passing the guard wins
                sim = float(scores[i])
                if sim < self.threshold:
                    return None
                if guard_ok(q_text, self._queries[i]):
                    row = self._db.execute(
                        "SELECT response_json, cost_usd FROM entries WHERE id=?", (self._ids[i],)
                    ).fetchone()
                    self._db.execute(
                        "UPDATE entries SET hit_count=hit_count+1, last_hit_at=? WHERE id=?",
                        (now, self._ids[i]),
                    )
                    self._db.commit()
                    return CacheHit(self._ids[i], sim, self._queries[i], row[0], row[1])
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
        now = time.time() if now is None else now
        emb = _unit(q_emb)
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO entries (query, embedding, response_json, namespace, mode, created_at,"
                    " expires_at, last_hit_at, cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        q_text,
                        emb.tobytes(),
                        response_json,
                        namespace,
                        mode,
                        now,
                        now + self.ttl_s,
                        now,
                        cost_usd,
                    ),
                )
                self._db.commit()
                self._load()
            except sqlite3.IntegrityError:  # same question already cached for this version: refresh it
                self._db.execute(
                    "UPDATE entries SET embedding=?, response_json=?, created_at=?, expires_at=?, cost_usd=?"
                    " WHERE query=? AND namespace=? AND mode=?",
                    (emb.tobytes(), response_json, now, now + self.ttl_s, cost_usd, q_text, namespace, mode),
                )
                self._db.commit()
                self._load()
            if len(self) > self.max_entries:
                self._evict(len(self) - self.max_entries)

    def _evict(self, n: int) -> None:  # least recently used first
        self._db.execute(
            "DELETE FROM entries WHERE id IN (SELECT id FROM entries ORDER BY last_hit_at ASC LIMIT ?)", (n,)
        )
        self._db.commit()
        self._load()

    def invalidate_except(self, namespace: str) -> int:
        """Call whenever the namespace changes: corpus edit, new answer model, new prompt version, new retrieval config. Returns how many entries were dropped."""
        with self._lock:
            cur = self._db.execute("DELETE FROM entries WHERE namespace != ?", (namespace,))
            self._db.commit()
            self._load()
            return cur.rowcount

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM entries")
            self._db.commit()
            self._load()
