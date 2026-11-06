"""Daily request guard (F-OPS-2), counted from llm_calls: one row per HTTP attempt, failed ones included."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from citerag.errors import BudgetExceededError

DAY_SECONDS = 86_400


def _window(provider: str | None) -> str:
    return "ts > ?" if provider is None else "ts > ? AND provider = ?"


def requests_last_24h(db_path: Path, now: float | None = None, provider: str | None = None) -> int:
    """Rows in a rolling 24 hours; each row is one HTTP attempt, failed ones included. llm_calls.ts is Unix seconds.
    Read-only, so a fresh install reports 0 without creating the database."""
    if not db_path.exists():
        return 0
    now = time.time() if now is None else now
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        args = (now - DAY_SECONDS,) if provider is None else (now - DAY_SECONDS, provider)
        row = db.execute(f"SELECT COUNT(*) FROM llm_calls WHERE {_window(provider)}", args).fetchone()
        return int(row[0])
    except sqlite3.OperationalError:  # no table until the first request is logged
        return 0
    finally:
        db.close()


def _next_free_slot(db_path: Path, now: float, provider: str | None) -> float:
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        args = (now - DAY_SECONDS,) if provider is None else (now - DAY_SECONDS, provider)
        row = db.execute(f"SELECT MIN(ts) FROM llm_calls WHERE {_window(provider)}", args).fetchone()
    finally:
        db.close()
    return (float(row[0]) if row[0] is not None else now) + DAY_SECONDS


def check_budget(
    db_path: Path, limit: int, now: float | None = None, provider: str | None = None, need: int = 1
) -> None:
    """Raises budget_exceeded unless `need` more requests fit under the limit in the rolling 24 hours, per provider
    when one is named. A crew needs several at once, and must not start with room for only some of them."""
    now = time.time() if now is None else now
    used = requests_last_24h(db_path, now, provider)
    if used + need > limit:
        free_at = datetime.fromtimestamp(_next_free_slot(db_path, now, provider), UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        raise BudgetExceededError(
            f"Daily LLM request limit reached for {provider or 'all providers'}: {used} of {limit} "
            f"in the last 24 hours{f', and this needs {need}' if need > 1 else ''}. "
            f"The next request frees up at {free_at}."
        )
