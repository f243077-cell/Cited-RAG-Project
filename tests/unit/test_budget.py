import sqlite3
import time

import pytest

from citerag.analytics.logs import LLMCallRow, LogStore
from citerag.errors import BudgetExceededError
from citerag.llm.budget import check_budget, requests_last_24h

NOW = 1_000_000.0
DAY = 86_400


def _db(path, stamps):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE llm_calls (id INTEGER PRIMARY KEY, ts REAL NOT NULL, ok INTEGER)")
    db.executemany("INSERT INTO llm_calls (ts, ok) VALUES (?, 0)", [(s,) for s in stamps])
    db.commit()
    db.close()


def test_counts_a_rolling_day_including_failed_attempts(tmp_path):
    path = tmp_path / "logs" / "app.sqlite"
    _db(path, [NOW - DAY - 1, NOW - DAY + 1, NOW - 5, NOW])
    assert requests_last_24h(path, now=NOW) == 3


def test_missing_database_counts_zero_and_is_not_created(tmp_path):
    path = tmp_path / "logs" / "app.sqlite"
    assert requests_last_24h(path, now=NOW) == 0
    assert not path.exists()


def test_database_without_the_table_counts_zero(tmp_path):
    path = tmp_path / "app.sqlite"
    sqlite3.connect(path).close()
    assert requests_last_24h(path, now=NOW) == 0


def _provider_db(path, rows):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE llm_calls (id INTEGER PRIMARY KEY, ts REAL NOT NULL, provider TEXT NOT NULL)")
    db.executemany("INSERT INTO llm_calls (ts, provider) VALUES (?, ?)", rows)
    db.commit()
    db.close()


def test_counts_one_provider_at_a_time(tmp_path):
    path = tmp_path / "app.sqlite"
    _provider_db(path, [(NOW - 5, "gemini"), (NOW - 4, "openrouter"), (NOW - 3, "openrouter")])
    assert requests_last_24h(path, now=NOW, provider="gemini") == 1
    assert requests_last_24h(path, now=NOW, provider="openrouter") == 2
    assert requests_last_24h(path, now=NOW) == 3


def test_guard_raises_at_the_limit_and_names_the_provider_and_reset(tmp_path):
    path = tmp_path / "app.sqlite"
    _provider_db(path, [(NOW - 5, "gemini"), (NOW - 3, "openrouter")])
    check_budget(path, 2, now=NOW, provider="gemini")  # one of two used
    with pytest.raises(BudgetExceededError, match="gemini: 1 of 1.*frees up at 1970-01-13"):
        check_budget(path, 1, now=NOW, provider="gemini")


def test_a_crew_needs_room_for_all_its_requests(tmp_path):
    db = tmp_path / "app.sqlite"
    log = LogStore(db)
    for _ in range(38):
        log.write_llm_call(
            LLMCallRow(ts=time.time(), role="x", provider="openrouter", model="m", attempt=1, ok=True)
        )
    check_budget(db, 40, provider="openrouter", need=2)
    with pytest.raises(BudgetExceededError, match="this needs 3"):
        check_budget(db, 40, provider="openrouter", need=3)


def test_a_limit_too_small_for_a_crew_refuses_with_an_empty_log(tmp_path):
    db = tmp_path / "app.sqlite"
    LogStore(db)
    with pytest.raises(BudgetExceededError):
        check_budget(db, 2, provider="openrouter", need=3)
