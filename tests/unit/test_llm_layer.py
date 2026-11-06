"""Tests for the OpenRouter-facing LLM layer helpers (parse, rate limiter, error classification) and the fake server."""

import httpx
import openai
import pytest
from pydantic import BaseModel, ValidationError

from citerag.eval.metrics import estimate_cost_usd
from citerag.llm.errors_map import classify_error
from citerag.llm.parse import parse_model, schema_response_format
from citerag.llm.ratelimit import SlidingWindowLimiter
from tests.fake_openai_server import FakeOpenRouter


class Ans(BaseModel):
    answer: str
    answerable: bool


GOOD = '{"answer": "30 days [1]", "answerable": true}'


# ---------------- extract_json / parse_model ----------------
@pytest.mark.parametrize(
    "raw",
    [
        GOOD,
        "Sure! Here you go:\n```json\n" + GOOD + "\n```\nHope that helps.",
        "```\n" + GOOD + "\n```",
        "<think>The user wants JSON {not this}</think>\n" + GOOD,
        "Answer: " + GOOD + " (done)",
    ],
)
def test_extract_json_variants(raw):
    assert parse_model(Ans, raw) == Ans(answer="30 days [1]", answerable=True)


def test_parse_failure_raises_validation_error():
    with pytest.raises(ValidationError):
        parse_model(Ans, "I could not find that.")
    with pytest.raises(ValidationError):
        parse_model(Ans, '{"answer": "x"}')  # missing field


def test_braces_inside_strings_survive():
    raw = '```json\n{"answer": "use {braces} freely", "answerable": false}\n```'
    assert parse_model(Ans, raw).answer == "use {braces} freely"


def test_schema_response_format_shape():
    rf = schema_response_format(Ans)
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "Ans"
    assert rf["json_schema"]["strict"] is False
    assert set(rf["json_schema"]["schema"]["properties"]) == {"answer", "answerable"}


# ---------------- rate limiter ----------------
def test_limiter_blocks_then_slides():
    now = [100.0]
    lim = SlidingWindowLimiter(3, 60.0, clock=lambda: now[0])
    assert [lim.try_acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = lim.try_acquire()
    assert wait == pytest.approx(60.0)  # must wait for the first stamp to expire
    now[0] += 30
    assert lim.try_acquire() == pytest.approx(30.0)
    now[0] += 30.5
    assert lim.try_acquire() == 0.0  # window slid: all three old stamps expired together
    assert lim.in_window() == 1


def test_limiter_reserve_for_crew_runs():
    lim = SlidingWindowLimiter(15, 60.0, clock=lambda: 0.0)
    assert lim.try_acquire(3) == 0.0  # a 3-task crew reserves 3 slots up front
    assert lim.in_window() == 3
    assert lim.try_acquire(13) > 0  # would exceed 15: nothing is taken
    assert lim.in_window() == 3
    with pytest.raises(ValueError):
        lim.try_acquire(16)


# ---------------- error classification ----------------
def _status_error(cls, status, headers=None):
    req = httpx.Request("POST", "http://x/api/v1/chat/completions")
    resp = httpx.Response(status, headers=headers or {}, request=req)
    return cls("boom", response=resp, body=None)


def test_classify_rate_limit_uses_retry_after():
    c = classify_error(_status_error(openai.RateLimitError, 429, {"retry-after": "7"}))
    assert (c.kind, c.retry, c.retry_after_s, c.http_status) == ("rate_limit", True, 7.0, 429)


def test_classify_status_codes():
    mk = lambda cls, s, h=None: classify_error(_status_error(cls, s, h))  # noqa: E731
    assert mk(openai.InternalServerError, 503).kind == "overloaded"
    assert mk(openai.InternalServerError, 502).kind == "server"
    assert mk(openai.APIStatusError, 408).kind == "timeout"
    c402 = mk(openai.APIStatusError, 402)
    assert c402.kind == "credits" and c402.retry is False  # no Retry-After: add credits, do not loop
    assert mk(openai.APIStatusError, 402, {"retry-after": "5"}).retry is True
    assert mk(openai.AuthenticationError, 401).retry is False
    c400 = mk(openai.BadRequestError, 400)
    assert (
        c400.kind == "bad_request" and c400.retry is False
    )  # gateway switches to the tool-calling method instead


def test_classify_non_http_errors():
    req = httpx.Request("POST", "http://x")
    assert classify_error(openai.APITimeoutError(request=req)).kind == "timeout"
    assert classify_error(openai.APIConnectionError(request=req)).kind == "server"
    assert classify_error(TypeError("'NoneType' object is not iterable")).kind == "provider_body"
    assert (
        classify_error(ValueError({"code": 502, "message": "Provider returned error"})).kind
        == "provider_body"
    )
    with pytest.raises(ValidationError) as ei:
        parse_model(Ans, "nope")
    assert classify_error(ei.value).kind == "parse"
    assert classify_error(KeyError("x")).retry is False


# ---------------- shadow cost fallback ----------------
def test_shadow_cost_falls_back_to_reference():
    pr = {"paid/ref": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
    pr["_reference"] = pr["paid/ref"]
    assert estimate_cost_usd("vendor/free-model:free", 1_000_000, 100_000, pr) == pytest.approx(4.5)
    assert estimate_cost_usd("paid/ref", 1_000_000, 0, pr) == pytest.approx(3.0)
    with pytest.raises(KeyError):
        estimate_cost_usd("x", 1, 1, {})


# ---------------- fake server + langchain-openai: what the gateway relies on ----------------
def _llm(srv, **kw):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="vendor/model:free", base_url=srv.base_url, api_key="test", max_retries=0, **kw)


def test_gateway_style_call_json_schema_with_fenced_reply_and_image():
    from langchain_core.messages import HumanMessage

    with FakeOpenRouter() as srv:
        srv.queue_text("```json\n" + GOOD + "\n```")
        llm = _llm(
            srv,
            max_tokens=300,
            default_headers={"X-Title": "CiteRAG"},
            extra_body={"provider": {"require_parameters": True}},
        )
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "What does it show?"},
                {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
            ]
        )
        raw = llm.bind(response_format=schema_response_format(Ans)).invoke([msg])
        out = parse_model(Ans, raw.content)
        body = srv.requests[0]
    assert out.answerable is True
    assert raw.usage_metadata["input_tokens"] == 11 and raw.usage_metadata["output_tokens"] == 7
    assert body["response_format"]["type"] == "json_schema"
    assert body["provider"] == {"require_parameters": True}
    assert any(
        p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/png;base64,")
        for p in body["messages"][0]["content"]
    )
    assert not {"temperature", "top_p", "top_k"} & set(body)  # no sampling knobs unless configured


def test_gateway_style_tool_calling_fallback():
    with FakeOpenRouter() as srv:
        srv.queue_tool("Ans", {"answer": "x", "answerable": False})
        raw = _llm(srv).bind_tools([Ans], tool_choice="Ans").invoke("q")
        assert Ans.model_validate(raw.tool_calls[0]["args"]).answerable is False


def test_429_then_success_counts_every_http_attempt():
    with FakeOpenRouter() as srv:
        srv.queue_429(retry_after_s=2)
        srv.queue_text(GOOD)
        llm = _llm(srv)
        attempts, result = 0, None
        for _ in range(3):  # the gateway's own retry loop
            attempts += 1
            try:
                result = llm.invoke("q")
                break
            except Exception as e:  # noqa: BLE001
                c = classify_error(e)
                assert (c.kind, c.retry_after_s) == ("rate_limit", 2.0)
        assert result is not None and attempts == 2 and len(srv.requests) == 2


def test_error_body_inside_http_200_is_classified():
    with FakeOpenRouter() as srv:
        srv.queue_error_in_200()
        with pytest.raises(Exception) as ei:  # noqa: PT011
            _llm(srv).invoke("q")
        assert classify_error(ei.value).kind == "provider_body"
