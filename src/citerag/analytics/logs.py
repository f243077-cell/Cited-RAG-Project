"""SQLite log (ARCHITECTURE section 6): requests, llm_calls and messages. Never stores API keys or image bytes."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from pydantic import BaseModel

from citerag.schemas import Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, ts REAL NOT NULL, session_id TEXT NOT NULL, mode TEXT NOT NULL,
    message TEXT NOT NULL, standalone_query TEXT, has_image INTEGER NOT NULL, cache_enabled INTEGER NOT NULL,
    cache_hit INTEGER NOT NULL, cache_similarity REAL, top_similarity REAL, answerable INTEGER,
    n_citations INTEGER NOT NULL, n_valid_citations INTEGER NOT NULL, confidence TEXT, n_retries INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, llm_requests INTEGER NOT NULL,
    est_cost_usd REAL NOT NULL, saved_usd REAL, saved_requests INTEGER, lat_total INTEGER NOT NULL,
    lat_condense INTEGER NOT NULL, lat_embed INTEGER NOT NULL, lat_cache INTEGER NOT NULL,
    lat_retrieve INTEGER NOT NULL, lat_rerank INTEGER NOT NULL, lat_generate INTEGER NOT NULL,
    lat_verify INTEGER NOT NULL, prompt_version TEXT NOT NULL, models_json TEXT NOT NULL, error_code TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, ts REAL NOT NULL, role TEXT NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL, attempt INTEGER NOT NULL, http_status INTEGER, error_kind TEXT,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, shadow_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL, ok INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_ts ON llm_calls (provider, ts);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, ts REAL NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, request_id TEXT
);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, id);
"""


class LLMCallRow(BaseModel):
    """One HTTP attempt, failed ones included: the daily guard counts these rows."""

    request_id: str | None = None
    ts: float
    role: str  # what the call was for: condense, rewrite, generate, ...
    provider: str  # each provider has its own daily guard
    model: str  # the model that answered, or the requested one when nothing answered
    attempt: int
    http_status: int | None = None
    error_kind: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    shadow_usd: float = 0.0
    latency_ms: int = 0
    ok: bool


class RequestRow(BaseModel):
    request_id: str
    ts: float
    session_id: str
    mode: str
    message: str
    standalone_query: str | None = None
    has_image: bool = False
    cache_enabled: bool = False
    cache_hit: bool = False
    cache_similarity: float | None = None
    top_similarity: float | None = None
    answerable: bool | None = None
    n_citations: int = 0
    n_valid_citations: int = 0
    confidence: str | None = None
    n_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_requests: int = 0
    est_cost_usd: float = 0.0
    saved_usd: float | None = None
    saved_requests: int | None = None
    lat_total: int = 0
    lat_condense: int = 0
    lat_embed: int = 0
    lat_cache: int = 0
    lat_retrieve: int = 0
    lat_rerank: int = 0
    lat_generate: int = 0
    lat_verify: int = 0
    prompt_version: str
    models_json: str = "{}"
    error_code: str | None = None


class LogStore:
    """Opens one connection per call, so any thread can use it."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    def _insert(self, table: str, row: BaseModel) -> None:
        values = row.model_dump()
        names = ", ".join(values)
        slots = ", ".join(f":{name}" for name in values)
        with closing(self._connect()) as db, db:
            db.execute(f"INSERT INTO {table} ({names}) VALUES ({slots})", values)

    def write_llm_call(self, row: LLMCallRow) -> None:
        self._insert("llm_calls", row)

    def write_request(self, row: RequestRow) -> None:
        self._insert("requests", row)

    def llm_calls_for(self, request_id: str) -> int:
        with closing(self._connect()) as db:
            found = db.execute("SELECT COUNT(*) FROM llm_calls WHERE request_id = ?", (request_id,))
            return int(found.fetchone()[0])

    def add_turn(self, session_id: str, user: str, assistant: str, request_id: str) -> None:
        """The user message and its answer go in together, so a turn is never half stored."""
        now = time.time()
        rows = [
            (session_id, now, "user", user, request_id),
            (session_id, now, "assistant", assistant, request_id),
        ]
        with closing(self._connect()) as db, db:
            db.executemany(
                "INSERT INTO messages (session_id, ts, role, content, request_id) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def _fetch(self, session_id: str, limit: int | None) -> list[Message]:
        query = "SELECT session_id, ts, role, content, request_id FROM messages WHERE session_id = ?"
        query += " ORDER BY id DESC"
        args: tuple = (session_id,)
        if limit is not None:
            query += " LIMIT ?"
            args = (session_id, limit)
        with closing(self._connect()) as db:
            rows = db.execute(query, args).fetchall()
        names = ("session_id", "ts", "role", "content", "request_id")
        return [Message(**dict(zip(names, row, strict=True))) for row in reversed(rows)]

    def messages(self, session_id: str) -> list[Message]:
        return self._fetch(session_id, None)

    def last_turns(self, session_id: str, turns: int) -> list[Message]:
        return self._fetch(session_id, 2 * turns)

    def delete_session(self, session_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
