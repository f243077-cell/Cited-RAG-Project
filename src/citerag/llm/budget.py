"""Daily request count, read from llm_calls. The guard that raises budget_exceeded arrives with the gateway in M3."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DAY_SECONDS = 86_400


def requests_last_24h(db_path: Path, now: float | None = None) -> int:
    """Rows in a rolling 24 hours; each row is one HTTP attempt, failed ones included. llm_calls.ts is Unix seconds.
    Read-only, so a fresh install reports 0 without creating the database."""
    if not db_path.exists():
        return 0
    now = time.time() if now is None else now
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = db.execute("SELECT COUNT(*) FROM llm_calls WHERE ts > ?", (now - DAY_SECONDS,)).fetchone()
        return int(row[0])
    except sqlite3.OperationalError:  # the table appears with the request log in M3
        return 0
    finally:
        db.close()
