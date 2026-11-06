import sqlite3

from citerag.llm.budget import requests_last_24h

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
