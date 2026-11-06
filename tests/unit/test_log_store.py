import sqlite3
import time

from citerag.analytics.logs import LLMCallRow, LogStore, RequestRow


def test_turns_come_back_oldest_first_and_cut_to_the_limit(tmp_path):
    log = LogStore(tmp_path / "logs" / "app.sqlite")
    for n in range(3):
        log.add_turn("s", f"q{n}", f"a{n}", f"r{n}")
    log.add_turn("other", "x", "y", "r9")
    assert [m.content for m in log.last_turns("s", 2)] == ["q1", "a1", "q2", "a2"]
    assert len(log.messages("s")) == 6


def test_delete_session_touches_only_that_session(tmp_path):
    log = LogStore(tmp_path / "app.sqlite")
    log.add_turn("s", "q", "a", "r1")
    log.add_turn("t", "q", "a", "r2")
    log.delete_session("s")
    assert log.messages("s") == [] and len(log.messages("t")) == 2


def test_llm_calls_are_counted_per_request(tmp_path):
    log = LogStore(tmp_path / "app.sqlite")
    for attempt in (1, 2):
        row = LLMCallRow(
            request_id="r1",
            ts=time.time(),
            role="generate",
            provider="gemini",
            model="m",
            attempt=attempt,
            ok=False,
        )
        log.write_llm_call(row)
    assert log.llm_calls_for("r1") == 2 and log.llm_calls_for("r2") == 0


def test_a_request_row_holds_every_field(tmp_path):
    path = tmp_path / "app.sqlite"
    log = LogStore(path)
    log.write_request(
        RequestRow(request_id="r1", ts=1.0, session_id="s", mode="standard", message="q", prompt_version="1")
    )
    db = sqlite3.connect(path)
    columns = [c[1] for c in db.execute("PRAGMA table_info(requests)")]
    db.close()
    assert set(columns) == set(RequestRow.model_fields)
