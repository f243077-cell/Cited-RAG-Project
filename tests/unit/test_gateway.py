"""The gateway at the HTTP level against the local fake server, so retries and counting run for real."""

import asyncio
import sqlite3

import httpx
import openai
import pytest
from pydantic import BaseModel

from citerag.analytics.logs import LogStore
from citerag.errors import BudgetExceededError, LLMUnavailableError, StructuredOutputError
from citerag.llm.gateway import OpenRouterGateway
from tests.fake_openai_server import FakeOpenRouter

GOOD = '{"text": "ok"}'


class Reply(BaseModel):
    text: str


def _gateway(settings, srv, **over):
    cfg = settings.model_copy(
        update={"openrouter_base_url": srv.base_url, "gemini_base_url": srv.base_url, **over}
    )
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    return OpenRouterGateway(cfg, LogStore(cfg.logs_db_path), sleep=sleep), cfg, waits


def _call(gateway, role="strong", request_id="r1", **kw):
    return asyncio.run(
        gateway.structured_call(
            role=role,
            schema=Reply,
            system="sys",
            user="question",
            call_name="generate",
            request_id=request_id,
            **kw,
        )
    )


def _rows(cfg):
    db = sqlite3.connect(cfg.logs_db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM llm_calls ORDER BY id")]
    db.close()
    return rows


def test_success_logs_one_row_with_the_served_model(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
        srv.queue_text(GOOD)
        result = _call(gateway)
    assert result.parsed == Reply(text="ok") and result.requests == 1
    assert result.model == "vendor/model:free" and (result.input_tokens, result.output_tokens) == (11, 7)
    assert srv.requests[0]["response_format"]["type"] == "json_schema"
    row = _rows(cfg)[0]
    assert row["model"] == "vendor/model:free" and row["role"] == "generate" and row["ok"] == 1
    assert row["provider"] == "openrouter" and row["request_id"] == "r1" and row["http_status"] == 200


def test_fenced_json_parses(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv)
        srv.queue_text(f"Here:\n```json\n{GOOD}\n```")
        assert _call(gateway).parsed.text == "ok"


def test_429_waits_for_retry_after_and_logs_each_attempt(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, waits = _gateway(settings, srv)
        srv.queue_429(7)
        srv.queue_text(GOOD)
        result = _call(gateway)
    assert result.requests == 2 and waits == [7.0]
    first, second = _rows(cfg)
    assert (first["ok"], first["http_status"], first["error_kind"]) == (0, 429, "rate_limit")
    assert second["ok"] == 1 and second["attempt"] == 2


def test_long_retry_after_fails_at_once(settings):
    with FakeOpenRouter() as srv:
        gateway, _, waits = _gateway(settings, srv)
        srv.queue_429(3600)
        with pytest.raises(LLMUnavailableError, match="rate limited"):
            _call(gateway)
    assert len(srv.requests) == 1 and waits == []


def test_error_inside_http_200_is_retried(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
        srv.queue_error_in_200()
        srv.queue_text(GOOD)
        assert _call(gateway).requests == 2
    assert _rows(cfg)[0]["ok"] == 0


def test_every_attempt_failing_raises_llm_unavailable(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv)
        for _ in range(3):
            srv.queue_429(0)
        with pytest.raises(LLMUnavailableError):
            _call(gateway)
    assert len(srv.requests) == 3


def test_bad_json_feeds_the_error_back_and_moves_to_tools(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
        srv.queue_text("not json at all")
        srv.queue_tool("Reply", {"text": "ok"})
        assert _call(gateway).parsed.text == "ok"
    second = srv.requests[1]
    assert "tools" in second and "response_format" not in second
    assert "failed validation" in second["messages"][-1]["content"]
    assert [r["error_kind"] for r in _rows(cfg)] == ["parse", None]


def test_two_validation_failures_raise(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv, structured_method="json_schema")
        srv.queue_text("nope")
        srv.queue_text("still nope")
        with pytest.raises(StructuredOutputError):
            _call(gateway)
    assert len(srv.requests) == 2
    assert all("response_format" in r for r in srv.requests)  # a fixed method never switches


def test_empty_content_retries_once_with_a_doubled_limit(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv)
        srv.queue_text("")
        srv.queue_text(GOOD)
        assert _call(gateway).requests == 2
    assert srv.requests[0]["max_completion_tokens"] == 2000
    assert srv.requests[1]["max_completion_tokens"] == 4000


def test_daily_guard_counts_failed_attempts(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv, max_requests_per_day=1)
        srv.queue_429(0)
        with pytest.raises(BudgetExceededError):
            _call(gateway)
    assert len(srv.requests) == 1


def test_gemini_roles_use_the_gemini_endpoint_without_openrouter_extras(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
        srv.queue_text(GOOD)
        _call(gateway, role="vision")
        body, headers = srv.requests[0], srv.headers[0]
    assert body["model"] == "vendor/vision:free" and "provider" not in body
    assert "x-title" not in headers and "http-referer" not in headers
    assert headers["authorization"] == "Bearer test-gemini-key"
    assert _rows(cfg)[0]["provider"] == "gemini"


def test_a_role_can_move_provider_by_config(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv, provider_strong="gemini", provider_vision="openrouter")
        srv.queue_text(GOOD)
        srv.queue_text(GOOD)
        _call(gateway, role="strong")
        _call(gateway, role="vision")
    assert [r["provider"] for r in _rows(cfg)] == ["gemini", "openrouter"]


def test_each_provider_has_its_own_daily_guard(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv, gemini_max_requests_per_day=1, max_requests_per_day=2)
        srv.queue_text(GOOD)
        srv.queue_text(GOOD)
        _call(gateway, role="vision")
        with pytest.raises(BudgetExceededError, match="gemini"):
            _call(gateway, role="vision")
        _call(gateway, role="strong")  # gemini at its limit does not block openrouter
    assert len(srv.requests) == 2


def test_each_provider_has_its_own_rate_limiter(settings):
    with FakeOpenRouter() as srv:
        gateway, _, waits = _gateway(settings, srv, gemini_max_rpm=1)
        srv.queue_text(GOOD)
        srv.queue_text(GOOD)
        srv.queue_text(GOOD)
        _call(gateway, role="vision")
        _call(gateway, role="strong")
        assert waits == []  # a full gemini window does not hold back openrouter

        async def expire(seconds: float) -> None:
            waits.append(seconds)
            gateway._limiters["gemini"]._stamps.clear()

        gateway._sleep = expire
        _call(gateway, role="vision")
    assert len(waits) == 1 and waits[0] > 0


def test_missing_gemini_key_names_the_setting_and_makes_no_request(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv, gemini_api_key=None)
        with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY"):
            _call(gateway, role="vision")
    assert srv.requests == []


class _Stub:
    """A client that raises a given error, for HTTP statuses the fake server does not serve."""

    def __init__(self, errors, temperature=None):
        self.errors, self.temperature = list(errors), temperature
        self.methods = []

    def bind(self, **_):
        self.methods.append("json_schema")
        return self

    def bind_tools(self, *_, **__):
        self.methods.append("tools")
        return self

    async def ainvoke(self, _):
        raise self.errors.pop(0)


def _status_error(kind, status, message):
    request = httpx.Request("POST", "http://x")
    return kind(message, response=httpx.Response(status, request=request), body=None)


def test_bad_request_moves_to_tools_then_stops(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
    stub = _Stub([_status_error(openai.BadRequestError, 400, "bad")] * 2)
    gateway._clients["strong"] = stub
    with pytest.raises(LLMUnavailableError):
        _call(gateway)
    assert stub.methods == ["json_schema", "tools"] and len(_rows(cfg)) == 2


def test_rejected_key_fails_at_once_without_echoing_it(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv)
    gateway._clients["vision"] = _Stub(
        [_status_error(openai.AuthenticationError, 401, "bad key test-gemini-key")]
    )
    with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY") as caught:
        _call(gateway, role="vision")
    assert "test-gemini-key" not in caught.value.message


def test_a_rejected_temperature_is_dropped_once(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
        srv.queue_text(GOOD)
        error = _status_error(openai.BadRequestError, 400, "temperature is not supported")
        gateway._clients["vision"] = _Stub([error], temperature=0.0)
        assert _call(gateway, role="vision").requests == 2
    assert "temperature" not in srv.requests[0]
    assert [r["ok"] for r in _rows(cfg)] == [0, 1]


def test_daily_count_on_health_stays_on_openrouter(settings):
    from citerag.llm.budget import requests_last_24h

    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
        srv.queue_text(GOOD)
        _call(gateway, role="vision")
    assert requests_last_24h(cfg.logs_db_path, provider="openrouter") == 0


def test_an_unclassified_failure_names_the_exception_type_without_its_text(settings):
    with FakeOpenRouter() as srv:
        gateway, cfg, _ = _gateway(settings, srv)
    gateway._clients["strong"] = _Stub([RuntimeError("prompt text that must not be logged")])
    with pytest.raises(LLMUnavailableError, match="RuntimeError") as caught:
        _call(gateway)
    assert "prompt text" not in caught.value.message
    assert _rows(cfg)[0]["error_kind"] == "unknown:RuntimeError"


def test_an_image_block_reaches_the_provider_as_a_data_url(settings):
    with FakeOpenRouter() as srv:
        gateway, _, _ = _gateway(settings, srv)
        srv.queue_text(GOOD)
        asyncio.run(
            gateway.structured_call(
                role="vision",
                schema=Reply,
                system="sys",
                user=[
                    {"type": "image", "base64": "aGk=", "mime_type": "image/png"},
                    {"type": "text", "text": "what is this"},
                ],
                call_name="vision",
            )
        )
    parts = srv.requests[0]["messages"][-1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == "data:image/png;base64,aGk="
    assert parts[1] == {"type": "text", "text": "what is this"}
