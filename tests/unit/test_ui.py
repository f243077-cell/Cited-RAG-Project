"""Streamlit pages through AppTest with a stub client, plus the HTTP client against a mock transport."""

import base64
import io
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from citerag.schemas import (
    CacheInfo,
    ChatResponse,
    Citation,
    CitedAnswer,
    DocumentInfo,
    ErrorBody,
    ErrorResponse,
    UsageInfo,
)
from citerag.ui import common
from citerag.ui.api_client import ApiClient, ApiError

PAGES = Path(common.__file__).parent / "pages"


def response(valid=True):
    citation = Citation(
        marker=1,
        chunk_id="c",
        doc_id="d",
        doc_name="policy.pdf",
        page=3,
        quote="<b>thirty days</b>",
        valid=valid,
        match_score=99,
    )
    answer = CitedAnswer(answer="Thirty days [1].", answerable=True, confidence="high", citations=[citation])
    return ChatResponse(
        request_id="r",
        session_id="s",
        standalone_query="q",
        answer=answer,
        cache=CacheInfo(enabled=False),
        usage=UsageInfo(),
    )


class StubClient:
    def __init__(self, reply=None):
        self.reply, self.requests, self.deleted = reply, [], []

    def documents(self):
        info = DocumentInfo(
            doc_id="d", doc_name="policy.pdf", chunks=2, size_bytes=9, ingested_at=datetime.now(UTC)
        )
        return [info]

    def chat(self, request):
        self.requests.append(request)
        if isinstance(self.reply, ApiError):
            raise self.reply
        return self.reply

    def delete_session(self, session_id):
        self.deleted.append(session_id)

    def delete_document(self, doc_id):
        self.deleted.append(doc_id)

    def upload(self, files):
        return []


def run_page(monkeypatch, name, client):
    monkeypatch.setattr(common, "get_client", lambda: client)
    return AppTest.from_file(str(PAGES / name), default_timeout=20).run()


def test_chat_shows_the_answer_with_a_citation_card(monkeypatch):
    client = StubClient(response())
    app = run_page(monkeypatch, "1_Chat.py", client)
    app.chat_input[0].set_value("how long?").run()
    assert not app.exception
    assert client.requests[0].message == "how long?" and client.requests[0].doc_ids is None
    assert any("Thirty days" in m.value for m in app.markdown)
    assert "policy.pdf, p3" in app.expander[0].label and "check_circle" in app.expander[0].label


def test_a_failed_citation_shows_a_warning_card(monkeypatch):
    app = run_page(monkeypatch, "1_Chat.py", StubClient(response(valid=False)))
    app.chat_input[0].set_value("how long?").run()
    assert "warning" in app.expander[0].label


def test_chat_shows_backend_errors(monkeypatch):
    client = StubClient(ApiError("budget_exceeded", "limit reached"))
    app = run_page(monkeypatch, "1_Chat.py", client)
    app.chat_input[0].set_value("how long?").run()
    assert not app.exception and app.error[0].value == "limit reached"


def test_library_lists_documents(monkeypatch):
    app = run_page(monkeypatch, "2_Library.py", StubClient())
    assert not app.exception and app.selectbox[0].options == ["policy.pdf"]


def test_library_delete_calls_the_backend(monkeypatch):
    client = StubClient()
    app = run_page(monkeypatch, "2_Library.py", client)
    app.button[0].click().run()
    assert client.deleted == ["d"]


def test_ui_source_stays_behind_the_http_client():
    forbidden = ("citerag.index", "citerag.rag", "citerag.llm", "citerag.ingest", "citerag.api", "chromadb")
    for path in [
        *PAGES.glob("*.py"),
        PAGES.parent / "app.py",
        PAGES.parent / "common.py",
        PAGES.parent / "api_client.py",
    ]:
        source = path.read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden), path.name


def _client(handler):
    client = ApiClient("http://backend")
    client._http = httpx.Client(base_url="http://backend", transport=httpx.MockTransport(handler))
    return client


def test_client_turns_the_error_body_into_api_error():
    body = ErrorResponse(error=ErrorBody(code="not_found", message="no such document")).model_dump()
    client = _client(lambda _: httpx.Response(404, json=body))
    with pytest.raises(ApiError, match="no such document") as caught:
        client.delete_document("x")
    assert caught.value.code == "not_found"


def test_client_reports_an_unreachable_backend():
    def refuse(_):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError) as caught:
        _client(refuse).documents()
    assert caught.value.code == "unreachable"


def test_api_key_header_is_sent_only_when_configured():
    assert "x-api-key" not in ApiClient("http://b")._http.headers
    assert ApiClient("http://b", "k")._http.headers["x-api-key"] == "k"


def test_chat_shows_verification_warnings_and_low_confidence(monkeypatch):
    reply = response(valid=False)
    reply.warnings = ["Some claims could not be verified against the documents."]
    reply.answer.confidence = "low"
    app = run_page(monkeypatch, "1_Chat.py", StubClient(reply))
    app.chat_input[0].set_value("how long?").run()
    assert not app.exception
    assert app.warning[0].value == "Some claims could not be verified against the documents."
    assert any("low" in c.value for c in app.caption)
    assert any("Quote not found" in c.value for c in app.caption)


def test_warnings_stay_visible_after_the_next_rerun(monkeypatch):
    reply = response(valid=False)
    reply.warnings = ["Some claims could not be verified against the documents."]
    app = run_page(monkeypatch, "1_Chat.py", StubClient(reply))
    app.chat_input[0].set_value("how long?").run()
    app.run()
    assert len(app.warning) == 1


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, "PNG")
    return buffer.getvalue()


def test_an_image_citation_shows_its_thumbnail(monkeypatch, tmp_path):
    picture = tmp_path / "figure.png"
    picture.write_bytes(png_bytes())
    reply = response()
    reply.answer.citations[0].kind = "image"
    reply.answer.citations[0].image_path = str(picture)
    app = run_page(monkeypatch, "1_Chat.py", StubClient(reply))
    app.chat_input[0].set_value("show me").run()
    assert not app.exception and app.expander[0].get("image")


def test_a_missing_thumbnail_file_still_shows_the_quote(monkeypatch):
    reply = response()
    reply.answer.citations[0].image_path = "data/images/gone.png"
    app = run_page(monkeypatch, "1_Chat.py", StubClient(reply))
    app.chat_input[0].set_value("show me").run()
    assert not app.exception and not app.expander[0].get("image")


def test_an_attached_image_is_sent_as_base64_and_shown_as_your_image(monkeypatch):
    data = png_bytes()
    upload = SimpleNamespace(getvalue=lambda: data, type="image/png")
    submitted = SimpleNamespace(text="what is this?", files=[upload])
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: submitted)
    client = StubClient(response())
    app = run_page(monkeypatch, "1_Chat.py", client)
    assert not app.exception
    sent = client.requests[0]
    assert base64.b64decode(sent.image_b64) == data and sent.image_media_type == "image/png"


def test_an_image_with_no_text_gets_a_default_question(monkeypatch):
    upload = SimpleNamespace(getvalue=png_bytes, type="image/png")
    monkeypatch.setattr(st, "chat_input", lambda *a, **k: SimpleNamespace(text="", files=[upload]))
    client = StubClient(response())
    run_page(monkeypatch, "1_Chat.py", client)
    assert client.requests[0].message == "What does this image show?"


def uploading_client(monkeypatch):
    client = StubClient()
    client.sent = []
    client.upload = lambda files, captions=False: client.sent.append(captions) or []
    monkeypatch.setattr(
        st, "file_uploader", lambda *a, **k: [SimpleNamespace(name="a.pdf", getvalue=lambda: b"x")]
    )
    return client


def ingest_button(app):
    return next(b for b in app.button if b.label == "Ingest")


def test_the_caption_toggle_is_visible_and_off_by_default(monkeypatch):
    client = uploading_client(monkeypatch)
    app = run_page(monkeypatch, "2_Library.py", client)
    assert app.toggle[0].label == "Caption images inside documents" and app.toggle[0].value is False
    ingest_button(app).click().run()
    assert client.sent == [False]


def test_turning_the_toggle_on_warns_and_sends_captions(monkeypatch):
    client = uploading_client(monkeypatch)
    app = run_page(monkeypatch, "2_Library.py", client)
    app.toggle[0].set_value(True).run()
    assert any("one LLM request" in c.value for c in app.caption)
    ingest_button(app).click().run()
    assert client.sent == [True]


def test_the_client_sends_the_caption_choice_as_a_form_field():
    seen = []

    def handler(request):
        seen.append(request.content)
        return httpx.Response(200, json=[])

    _client(handler).upload([("a.md", b"x")])
    _client(handler).upload([("a.md", b"x")], captions=True)
    assert b'name="captions"' in seen[0] and b"false" in seen[0] and b"true" in seen[1]


def log_rows():
    base = dict.fromkeys(
        (
            "lat_condense",
            "lat_embed",
            "lat_cache",
            "lat_retrieve",
            "lat_rerank",
            "lat_generate",
            "lat_verify",
        ),
        5,
    )
    return [
        {**base, "message": "q1", "cache_enabled": True, "cache_hit": False, "cache_similarity": 0.4,
         "llm_requests": 1, "saved_requests": None, "lat_total": 900, "error_code": None},
        {**base, "message": "q2", "cache_enabled": True, "cache_hit": True, "cache_similarity": 0.97,
         "llm_requests": 0, "saved_requests": 1, "lat_total": 20, "error_code": None},
    ]  # fmt: skip


class DashboardClient(StubClient):
    rows = None
    sweep = None
    cleared = False

    def request_log(self, limit=1000):
        return self.rows if self.rows is not None else log_rows()

    def llm_call_log(self, limit=1000):
        return [{"role": "generate", "input_tokens": 10, "output_tokens": 5}]

    def cache_status(self):
        return {"backend": "semantic", "enabled": True, "entries": 3, "threshold": 0.92}

    def cache_sweep(self):
        return self.sweep

    def clear_cache(self):
        self.cleared = True


def test_the_dashboard_shows_hits_and_savings_from_the_log(monkeypatch):
    st.cache_data.clear()
    app = run_page(monkeypatch, "3_Dashboard.py", DashboardClient())
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "2 requests" in text and "1 cache hits of 2 lookups" in text and "1 LLM requests saved" in text
    assert "threshold 0.92" in " ".join(c.value for c in app.caption)


def test_the_dashboard_says_when_there_is_no_sweep_or_no_request(monkeypatch):
    st.cache_data.clear()
    app = run_page(monkeypatch, "3_Dashboard.py", DashboardClient())
    assert any("No threshold sweep" in i.value for i in app.info)
    st.cache_data.clear()
    client = DashboardClient()
    client.rows = []
    app = run_page(monkeypatch, "3_Dashboard.py", client)
    assert not app.exception and any("No requests yet" in i.value for i in app.info)


def test_the_dashboard_can_clear_the_cache(monkeypatch):
    st.cache_data.clear()
    client = DashboardClient()
    app = run_page(monkeypatch, "3_Dashboard.py", client)
    next(b for b in app.button if b.label == "Clear cache").click().run()
    assert client.cleared


def test_the_cache_toggle_is_on_by_default_and_turns_the_cache_off_for_a_question(monkeypatch):
    client = StubClient(response())
    app = run_page(monkeypatch, "1_Chat.py", client)
    assert app.toggle[0].label == "Use cache" and app.toggle[0].value is True
    app.toggle[0].set_value(False).run()
    app.chat_input[0].set_value("q").run()
    assert client.requests[-1].use_cache is False


def test_a_cached_answer_shows_the_badge_with_its_similarity(monkeypatch):
    reply = response()
    reply.cache = CacheInfo(enabled=True, hit=True, similarity=0.967, matched_query="m", saved_requests=1)
    app = run_page(monkeypatch, "1_Chat.py", StubClient(reply))
    app.chat_input[0].set_value("q").run()
    captions = " ".join(c.value for c in app.caption)
    assert (
        "Served from cache" in captions
        and "similarity 0.97" in captions
        and "1 LLM requests saved" in captions
    )


def test_the_mode_is_standard_by_default_warns_of_the_cost_and_is_sent(monkeypatch):
    client = StubClient(response())
    app = run_page(monkeypatch, "1_Chat.py", client)
    assert app.radio[0].label == "Mode" and app.radio[0].value == "Standard"
    app.chat_input[0].set_value("q").run()
    assert client.requests[-1].mode == "standard"
    app.radio[0].set_value("Deep").run()
    assert any("at least 3 LLM requests" in c.value for c in app.caption)
    app.chat_input[0].set_value("r").run()
    assert client.requests[-1].mode == "deep"
    app.radio[0].set_value("Agent").run()
    assert any("at most 8" in c.value for c in app.caption)
    app.chat_input[0].set_value("s").run()
    assert client.requests[-1].mode == "agent"


def test_an_agent_answer_shows_its_trace_as_plain_text(monkeypatch):
    from citerag.schemas import AgentStep, AgentTrace

    reply = response()
    reply.trace = AgentTrace(
        sub_questions=["What are the *tiers*?"],
        steps=[
            AgentStep(node="plan", note="1 sub-question(s)", llm_requests=1),
            AgentStep(
                node="tools", tool="search_documents", args={"query": "<b>tiers</b>"}, hits=["csf.pdf p8"]
            ),
            AgentStep(node="synthesize", llm_requests=1),
        ],
        stop_reason="enough",
        llm_requests=2,
    )
    app = run_page(monkeypatch, "1_Chat.py", StubClient(reply))
    app.chat_input[0].set_value("q").run()
    assert app.expander[-1].label == "Agent trace · 3 steps · 2 LLM requests"
    texts = [t.value for t in app.text]
    assert "1. What are the *tiers*?" in texts
    assert any("search_documents(query='<b>tiers</b>')" in t and "[csf.pdf p8]" in t for t in texts)
