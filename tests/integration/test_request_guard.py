"""F-OPS-2 end to end: the real gateway against the fake server, so a refused call is a call that never left."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from citerag import cli
from citerag.agents.deep_mode import DeepCrew
from citerag.agents.runtime import CrewRuntime
from citerag.analytics.logs import LLMCallRow, LogStore, RequestRow
from citerag.api.main import create_app
from citerag.llm.budget import requests_last_24h
from citerag.llm.gateway import OpenRouterGateway
from tests.fake_openai_server import FakeOpenRouter
from tests.integration.test_chat_api import POLICY, chat, factory, upload

ANSWER = json.dumps(
    {
        "answer": "Thirty days [1].",
        "answerable": True,
        "citations": [{"marker": 1, "quote": "returned within thirty days"}],
        "confidence": "high",
    }
)


def fill(log, n, provider="openrouter", age=0.0):
    for _ in range(n):
        row = LLMCallRow(
            ts=time.time() - age, role="generate", provider=provider, model="m", attempt=1, ok=False
        )
        log.write_llm_call(row)


@pytest.fixture
def api(settings):
    with FakeOpenRouter() as server:
        cfg = settings.model_copy(update={"openrouter_base_url": server.base_url, "max_requests_per_day": 5})
        made = {}

        def build(config, log, cache):
            gateway = OpenRouterGateway(config, log)
            services = factory(gateway)(config, log, cache)
            services.deps.deep = DeepCrew(CrewRuntime(config, gateway, log))
            made["log"] = log
            return services

        with TestClient(create_app(cfg, services_factory=build)) as client:
            upload(client, ("policy.txt", POLICY))
            yield client, server, made["log"], cfg


def error_of(response):
    return response.json()["error"]


def test_a_full_guard_answers_429_and_no_request_leaves(api):
    client, server, log, cfg = api
    fill(log, cfg.max_requests_per_day)
    response = chat(client)
    assert response.status_code == 429 and error_of(response)["code"] == "budget_exceeded"
    message = error_of(response)["message"]
    assert "5 of 5" in message and "frees up at" in message and error_of(response)["request_id"]
    assert server.requests == []


def test_a_refused_request_is_logged_with_its_error(api):
    client, _, log, cfg = api
    fill(log, cfg.max_requests_per_day)
    chat(client)
    assert [r.error_code for r in log.recent_requests()] == ["budget_exceeded"]


def test_a_cached_answer_is_still_served_after_the_guard_fills(api):
    client, server, log, cfg = api
    server.queue_text(ANSWER)
    assert not chat(client).json()["cache"]["hit"]
    client.delete("/v1/sessions/s")
    fill(log, cfg.max_requests_per_day)
    calls = len(server.requests)
    response = chat(client)
    assert response.status_code == 200 and response.json()["cache"]["hit"]
    assert len(server.requests) == calls


def test_deep_mode_needs_room_for_all_three_agents(api):
    client, server, log, cfg = api
    fill(log, cfg.max_requests_per_day - 2)
    response = chat(client, mode="deep")
    assert response.status_code == 429 and "this needs 3" in error_of(response)["message"]
    assert server.requests == []


def test_failed_attempts_count_towards_the_guard(api):
    client, server, log, cfg = api
    server.queue_429(0)
    server.queue_429(0)
    server.queue_text(ANSWER)
    assert chat(client).status_code == 200
    assert requests_last_24h(cfg.logs_db_path, provider="openrouter") == 3


def test_requests_older_than_a_day_free_their_slots(api):
    client, server, log, cfg = api
    fill(log, cfg.max_requests_per_day, age=86_401)
    server.queue_text(ANSWER)
    assert chat(client).status_code == 200


def test_each_provider_has_its_own_guard(api):
    client, server, log, cfg = api
    fill(log, 50, provider="gemini")
    server.queue_text(ANSWER)
    assert chat(client).status_code == 200


def test_health_reports_the_count_and_the_limit(api):
    client, _, log, _ = api
    fill(log, 2)
    body = client.get("/health").json()
    assert (body["llm_requests_24h"], body["max_requests_per_day"]) == (2, 5)


def test_quota_prints_each_provider_and_spends_nothing(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    log = LogStore(settings.logs_db_path)
    fill(log, 3)
    output = CliRunner().invoke(cli.app, ["quota"]).output
    assert "openrouter: 3 of 40" in output and "gemini: 0 of 18" in output


def old_request():
    return RequestRow(
        request_id="r", ts=1.0, session_id="s", mode="standard", message="old", prompt_version="1"
    )


def test_purge_removes_old_rows_only(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    log = LogStore(settings.logs_db_path)
    log.write_request(old_request())
    log.write_request(old_request().model_copy(update={"request_id": "new", "ts": time.time()}))
    fill(log, 1, age=40 * 86_400)
    fill(log, 1)
    log.add_turn("s", "q", "a", "r")
    result = CliRunner().invoke(cli.app, ["logs", "purge", "--days", "30"])
    assert result.exit_code == 0 and "Deleted 1 requests, 1 LLM calls" in result.output
    assert [r.request_id for r in log.recent_requests()] == ["new"]
    assert len(log.recent_llm_calls()) == 1 and len(log.messages("s")) == 2


def test_purge_refuses_to_touch_what_the_guard_still_counts(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    log = LogStore(settings.logs_db_path)
    fill(log, 1)
    result = CliRunner().invoke(cli.app, ["logs", "purge", "--days", "0"])
    assert result.exit_code != 0
    assert len(log.recent_llm_calls()) == 1
