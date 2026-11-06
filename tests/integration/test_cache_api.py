"""The real cache behind the API: real SQLite, FakeGateway and HashEmbedding, no network."""

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from citerag import cli
from citerag.api.main import create_app
from citerag.cache.store import open_cache
from citerag.eval import cache_tools
from citerag.llm.fake import FakeGateway
from citerag.schemas import ChatResponse, ErrorResponse
from tests.integration.test_chat_api import POLICY, QUESTION, answer, chat, factory, upload


@pytest.fixture
def api(settings):
    llm = FakeGateway()
    with TestClient(create_app(settings, services_factory=factory(llm))) as client:
        upload(client, ("policy.txt", POLICY))
        yield client, llm


def ask(client, llm, **kw):
    llm.queue("generate", answer())
    return ChatResponse.model_validate(chat(client, **kw).json())


def new_session(client):
    client.delete("/v1/sessions/s")


def test_the_same_question_is_served_from_the_cache_with_no_llm_call(api):
    client, llm = api
    first = ask(client, llm)
    assert not first.cache.hit and first.cache.enabled
    new_session(client)
    calls = len(llm.calls)
    second = ChatResponse.model_validate(chat(client).json())
    assert second.cache.hit and second.cache.similarity > 0.99 and second.cache.matched_query == QUESTION
    assert len(llm.calls) == calls and second.usage.llm_requests == 0
    assert second.cache.saved_requests == 1 and second.answer == first.answer


def test_a_changed_number_is_not_a_hit(api):
    """The guard, through the whole stack. HashEmbedding sees these as near-identical bags of words."""
    client, llm = api
    ask(client, llm, message="return items within 30 days")
    new_session(client)
    second = ask(client, llm, message="return items within 60 days")
    assert not second.cache.hit


def test_uploading_a_document_invalidates_the_cache(api):
    client, llm = api
    ask(client, llm)
    assert client.get("/v1/cache/stats").json()["entries"] == 1
    upload(client, ("other.txt", b"A different document about shipping."))
    assert client.get("/v1/cache/stats").json()["entries"] == 0


def test_deleting_a_document_invalidates_the_cache(api):
    client, llm = api
    ask(client, llm)
    doc_id = client.get("/v1/documents").json()[0]["doc_id"]
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 204
    assert client.get("/v1/cache/stats").json()["entries"] == 0


def test_a_restart_with_another_model_drops_what_the_old_model_answered(settings):
    llm = FakeGateway()
    with TestClient(create_app(settings, services_factory=factory(llm))) as client:
        upload(client, ("policy.txt", POLICY))
        ask(client, llm)
        assert client.get("/v1/cache/stats").json()["entries"] == 1
    changed = settings.model_copy(update={"model_strong": "vendor/newer:free"})
    with TestClient(create_app(changed, services_factory=factory(llm))) as client:
        assert client.get("/v1/cache/stats").json()["entries"] == 0


def test_stats_show_the_threshold_and_clear_empties_the_cache(api):
    client, llm = api
    ask(client, llm)
    assert client.get("/v1/cache/stats").json() == {
        "backend": "semantic",
        "enabled": True,
        "entries": 1,
        "threshold": 0.92,
    }
    assert client.post("/v1/cache/clear").status_code == 204
    assert client.get("/v1/cache/stats").json()["entries"] == 0


def test_a_corrupt_cache_file_still_answers(settings):
    settings.cache_db_path.parent.mkdir(parents=True)
    settings.cache_db_path.write_bytes(b"not a database " * 100)
    llm = FakeGateway()
    with TestClient(create_app(settings, services_factory=factory(llm))) as client:
        upload(client, ("policy.txt", POLICY))
        reply = ask(client, llm)
    assert reply.answer.answerable and not reply.cache.enabled


def test_the_log_endpoints_return_rows_oldest_first_and_respect_the_limit(api):
    client, llm = api
    ask(client, llm, message="how many days to return items within")
    new_session(client)
    ask(client, llm, message="how many days to return items after")
    rows = client.get("/v1/logs/requests").json()
    assert [r["message"] for r in rows] == [
        "how many days to return items within",
        "how many days to return items after",
    ]
    assert rows[0]["cache_similarity"] is None and rows[1]["cache_enabled"] is True
    assert len(client.get("/v1/logs/requests", params={"limit": 1}).json()) == 1
    assert client.get("/v1/logs/requests", params={"since": 9e12}).json() == []


def test_the_log_limit_is_bounded(api):
    client, _ = api
    response = client.get("/v1/logs/requests", params={"limit": 0})
    assert response.status_code == 400
    assert ErrorResponse.model_validate(response.json()).error.code == "validation_error"


def test_cache_and_log_routes_need_the_key_when_one_is_set(settings):
    keyed = settings.model_copy(update={"citerag_api_key": "secret"})
    with TestClient(create_app(keyed, services_factory=factory(FakeGateway()))) as client:
        for method, path in (
            ("get", "/v1/cache/stats"),
            ("post", "/v1/cache/clear"),
            ("get", "/v1/logs/requests"),
            ("get", "/v1/logs/llm_calls"),
            ("get", "/v1/eval/cache-sweep"),
        ):
            assert getattr(client, method)(path).status_code == 401, path


def test_the_sweep_endpoint_is_null_until_a_sweep_ran(api, monkeypatch, tmp_path):
    client, _ = api
    monkeypatch.setattr(cache_tools, "SWEEP_FILE", tmp_path / "sweep.json")
    monkeypatch.setattr(cache_tools.load_sweep, "__defaults__", (tmp_path / "sweep.json",))
    assert client.get("/v1/eval/cache-sweep").json() is None


def test_cli_stats_and_clear_work_on_the_file(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    cache = open_cache(settings)
    import numpy as np

    cache.insert("q", np.ones(4, dtype=np.float32), "{}", "ns")
    runner = CliRunner()
    assert "entries: 1" in runner.invoke(cli.app, ["cache", "stats"]).output
    assert "entries: 0" in runner.invoke(cli.app, ["cache", "clear"]).output


def test_the_sweep_command_refuses_candidates_nobody_labelled(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    pair = {"id": "p", "question_a": "a", "question_b": "b", "proposed": "equivalent", "kind": "paraphrase"}
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["eval", "cache-sweep", "--pairs", str(path)])
    assert result.exit_code == 1 and "No labelled pairs" in result.output
    assert "review_cache_pairs.py --label" in result.output


def test_the_sweep_command_runs_on_labelled_pairs_and_changes_no_setting(settings, monkeypatch, tmp_path):
    from citerag.index import store as store_module
    from tests.helpers import HashEmbedding

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "make_embedder", lambda _: HashEmbedding())
    monkeypatch.chdir(tmp_path)
    pair = {
        "id": "p",
        "question_a": "same words here",
        "question_b": "same words here",
        "proposed": "equivalent",
        "kind": "paraphrase",
        "labelled": True,
        "label": "equivalent",
    }
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["eval", "cache-sweep", "--pairs", str(path)])
    assert result.exit_code == 0, result.output
    assert "PROVISIONAL" in result.output and "1 equivalent, 0 near miss" in result.output
    assert (tmp_path / "docs" / "results" / "e5_threshold.png").exists()
    assert (tmp_path / "eval" / "results" / "cache_sweep" / "sweep.json").exists()


def test_bench_estimate_prints_a_number_and_spends_nothing(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    path = tmp_path / "w.jsonl"
    path.write_text(json.dumps({"id": "w", "question": "q"}) + "\n", encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["eval", "bench-cache", "--workload", str(path), "--estimate"])
    assert result.exit_code == 0 and "1 questions: about 4 LLM requests, 8 at worst" in result.output
