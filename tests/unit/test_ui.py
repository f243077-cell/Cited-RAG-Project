"""Streamlit pages through AppTest with a stub client, plus the HTTP client against a mock transport."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
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
