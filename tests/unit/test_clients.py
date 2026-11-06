"""Each role's client, checked at the HTTP level against the local fake server."""

import pytest

from citerag.llm.gateway import ROLES, build_client
from tests.fake_openai_server import FakeOpenRouter

SAMPLING = {"temperature", "top_p", "top_k"}


def _all_openrouter(settings, srv, **extra):
    update = {
        "openrouter_base_url": srv.base_url,
        "provider_vision": "openrouter",
        "provider_judge": "openrouter",
    }
    return settings.model_copy(update={**update, **extra})


@pytest.mark.parametrize("role", ROLES)
def test_role_client_sends_no_sampling_parameters(settings, role):
    with FakeOpenRouter() as srv:
        cfg = _all_openrouter(settings, srv, judge_temperature=None)
        client = build_client(cfg, role)
        client.invoke("q")
        body, headers = srv.requests[0], srv.headers[0]
    assert body["model"] == getattr(settings, f"model_{role}")
    assert not SAMPLING & set(body)
    assert body["provider"] == {"require_parameters": True, "data_collection": "allow"}
    assert headers["x-title"] == "CiteRAG" and headers["http-referer"] == "http://localhost:8501"
    assert client.max_retries == 0  # the gateway owns retries, so every attempt is counted


def test_only_the_judge_gets_the_configured_judge_temperature(settings):
    with FakeOpenRouter() as srv:
        cfg = _all_openrouter(settings, srv)
        assert cfg.judge_temperature == 0
        build_client(cfg, "judge").invoke("q")
        build_client(cfg, "strong").invoke("q")
    assert srv.requests[0]["temperature"] == 0 and "temperature" not in srv.requests[1]


def test_configured_llm_temperature_reaches_answer_roles(settings):
    with FakeOpenRouter() as srv:
        cfg = settings.model_copy(update={"openrouter_base_url": srv.base_url, "llm_temperature": 0.2})
        build_client(cfg, "fast").invoke("q")
    assert srv.requests[0]["temperature"] == 0.2


def test_unknown_role_is_rejected(settings):
    with pytest.raises(ValueError):
        build_client(settings, "writer")
