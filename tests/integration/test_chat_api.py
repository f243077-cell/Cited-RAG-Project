"""The API end to end: real Chroma, BM25 and verifier; FakeGateway and HashEmbedding instead of network and models."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from citerag.analytics.logs import LogStore
from citerag.api.deps import Services
from citerag.api.main import create_app
from citerag.errors import BudgetExceededError
from citerag.index.retriever import HybridRetriever
from citerag.index.store import IndexStore
from citerag.ingest.manifest import current_namespace
from citerag.ingest.pipeline import Ingestor
from citerag.llm.fake import FakeGateway
from citerag.rag.pipeline import Deps
from citerag.schemas import (
    ChatResponse,
    DocumentInfo,
    ErrorResponse,
    IngestReport,
    LLMCitation,
    LLMCitedAnswer,
)
from tests.helpers import HashEmbedding

POLICY = b"Items can be returned within thirty days of delivery."
QUESTION = "how many days to return items"


def factory(llm):
    def build(cfg, log: LogStore, cache) -> Services:
        store = IndexStore(cfg.index_dir, HashEmbedding())
        ingestor = Ingestor(cfg, store, cfg.index_dir, cache)
        retriever = HybridRetriever(store, cfg)
        deps = Deps(
            cfg=cfg,
            llm=llm,
            retriever=retriever,
            cache=cache,
            log=log,
            namespace=lambda: current_namespace(cfg, ingestor.manifest),
        )
        return Services(deps, store, ingestor, retriever)

    return build


def answer():
    citation = LLMCitation(marker=1, quote="returned within thirty days")
    return LLMCitedAnswer(answer="Thirty days [1].", answerable=True, citations=[citation], confidence="high")


@pytest.fixture
def api(settings):
    llm = FakeGateway()
    with TestClient(create_app(settings, services_factory=factory(llm))) as client:
        yield client, llm


def upload(client, *files):
    return client.post("/v1/documents", files=[("files", f) for f in files])


def chat(client, message=QUESTION, **kw):
    return client.post("/v1/chat", json={"session_id": "s", "message": message, **kw})


def test_upload_then_ask_returns_a_verified_citation(api):
    client, llm = api
    report = IngestReport.model_validate(upload(client, ("policy.md", POLICY)).json()[0])
    assert report.status == "ingested" and report.chunks == 1
    llm.queue("generate", answer())
    response = ChatResponse.model_validate(chat(client).json())
    assert response.answer.citations[0].valid and response.answer.citations[0].doc_name == "policy.md"
    assert response.answer.confidence == "high"


def test_empty_index_answers_normally_with_no_llm_call(api):
    client, llm = api
    response = chat(client)
    assert response.status_code == 200
    assert response.json()["answer"]["answer"] == "Upload documents first." and llm.calls == []


def test_second_upload_is_unchanged_and_list_shows_one_document(api):
    client, _ = api
    upload(client, ("policy.md", POLICY))
    assert upload(client, ("policy.md", POLICY)).json()[0]["status"] == "unchanged"
    documents = [DocumentInfo.model_validate(d) for d in client.get("/v1/documents").json()]
    assert [d.doc_name for d in documents] == ["policy.md"] and documents[0].chunks == 1


def test_listing_before_any_upload_is_empty(api):
    client, _ = api
    assert client.get("/v1/documents").json() == []


def test_a_bad_file_in_a_batch_ingests_nothing(api):
    client, _ = api
    response = upload(client, ("policy.md", POLICY), ("run.exe", b"MZ"))
    assert response.status_code == 415
    assert ErrorResponse.model_validate(response.json()).error.code == "invalid_file_type"
    assert client.get("/v1/documents").json() == []


def test_oversize_upload_is_rejected(settings):
    small = settings.model_copy(update={"max_upload_mb": 1})
    with TestClient(create_app(small, services_factory=factory(FakeGateway()))) as client:
        response = upload(client, ("big.txt", b"x" * (1024 * 1024 + 1)))
    assert response.status_code == 413


def test_delete_removes_the_document_from_search(api):
    client, llm = api
    doc_id = upload(client, ("policy.md", POLICY)).json()[0]["doc_id"]
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 204
    assert client.get("/v1/documents").json() == []
    assert chat(client).json()["answer"]["answer"] == "Upload documents first."
    assert llm.calls == []


def test_deleting_an_unknown_document_is_a_404(api):
    client, _ = api
    response = client.delete("/v1/documents/nope")
    assert response.status_code == 404
    assert ErrorResponse.model_validate(response.json()).error.code == "not_found"


def test_doc_filter_with_no_matching_document_refuses_without_llm(api):
    client, llm = api
    upload(client, ("policy.md", POLICY))
    assert chat(client, doc_ids=[]).json()["answer"]["answerable"] is False
    assert chat(client, doc_ids=["missing"]).json()["answer"]["answerable"] is False
    assert llm.calls == []


def test_follow_up_history_reaches_the_session_endpoints(api):
    client, llm = api
    upload(client, ("policy.md", POLICY))
    llm.queue("generate", answer())
    chat(client)
    messages = client.get("/v1/sessions/s/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert client.delete("/v1/sessions/s").status_code == 204
    assert client.get("/v1/sessions/s/messages").json() == []


def test_budget_error_uses_the_uniform_body_with_a_request_id(api):
    client, llm = api
    upload(client, ("policy.md", POLICY))
    llm.queue("generate", BudgetExceededError("limit reached"))
    response = chat(client)
    assert response.status_code == 429
    error = ErrorResponse.model_validate(response.json()).error
    assert error.code == "budget_exceeded" and error.request_id


def test_later_milestone_features_return_400(api):
    client, _ = api
    response = chat(client, mode="deep")
    assert response.status_code == 400
    assert ErrorResponse.model_validate(response.json()).error.code == "validation_error"


def test_api_key_is_enforced_only_when_set(settings):
    keyed = settings.model_copy(update={"citerag_api_key": SecretStr("secret")})
    with TestClient(create_app(keyed, services_factory=factory(FakeGateway()))) as client:
        assert client.get("/v1/documents").status_code == 401
        assert client.get("/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/documents", headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get("/health").status_code == 200


def test_cors_allows_only_configured_origins(api):
    client, _ = api
    headers = {"Access-Control-Request-Method": "GET"}
    allowed = client.options("/v1/documents", headers={**headers, "Origin": "http://localhost:8501"})
    refused = client.options("/v1/documents", headers={**headers, "Origin": "http://evil.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "access-control-allow-origin" not in refused.headers
