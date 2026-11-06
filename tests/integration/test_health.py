from fastapi import APIRouter
from fastapi.testclient import TestClient

from citerag.api.main import create_app
from citerag.errors import BudgetExceededError
from citerag.schemas import ErrorResponse


def _client(settings, **kw):
    return TestClient(create_app(settings), **kw)


def test_health_reports_the_m0_fields(settings):
    with _client(settings) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {"fastapi", "pydantic", "llama-index-core", "chromadb", "crewai", "python"} <= set(
        body["versions"]
    )
    assert body["manifest"] is None and body["namespace"] is None  # filled from M1
    assert body["cache"] == {"backend": "noop", "enabled": False, "entries": 0}
    assert body["llm_requests_24h"] == 0 and body["max_requests_per_day"] == 40


def test_health_never_exposes_the_key(settings):
    with _client(settings) as client:
        assert "test-key" not in client.get("/health").text


def test_typed_errors_use_the_uniform_body(settings):
    app = create_app(settings)
    router = APIRouter()

    @router.get("/boom")
    def boom():
        raise BudgetExceededError("limit reached")

    @router.get("/crash")
    def crash():
        raise RuntimeError("secret detail")

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as client:
        typed = client.get("/boom")
        crashed = client.get("/crash")
        missing = client.get("/nope")
    assert typed.status_code == 429
    assert ErrorResponse.model_validate(typed.json()).error.code == "budget_exceeded"
    assert crashed.status_code == 500 and "secret detail" not in crashed.text
    assert ErrorResponse.model_validate(crashed.json()).error.code == "internal_error"
    assert missing.status_code == 404
    assert ErrorResponse.model_validate(missing.json()).error.code == "not_found"


def test_openapi_docs_are_served(settings):
    with _client(settings) as client:
        assert client.get("/docs").status_code == 200
