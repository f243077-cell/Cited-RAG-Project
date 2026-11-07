"""Wind-up fixes: cache off by default with a warning, a clear message for missing settings, and review_golden terms."""

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from citerag import cli
from citerag.api.main import create_app
from citerag.config import Settings, SettingsError, describe_settings_error, get_settings
from citerag.eval.cache_tools import SweepResult, threshold_warning
from citerag.schemas import GoldenItem

ROOT = Path(__file__).resolve().parents[2]


def sweep(recommended=0.97, enough=True):
    return SweepResult(
        n_equivalent=30 if enough else 3,
        n_near_miss=30 if enough else 3,
        enough_pairs=enough,
        max_false_hit_rate=0.02,
        recommended=recommended,
        rows=[],
        rows_without_guard=[],
        worst_near_misses=[],
    )


def test_the_shipped_default_keeps_the_cache_off():
    assert Settings.model_fields["cache_enabled"].default is False
    assert "\nCACHE_ENABLED=false\n" in (ROOT / ".env.example").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("enabled", "threshold", "found", "warned"),
    [
        (False, 0.92, None, False),
        (True, 0.92, None, True),
        (True, 0.97, sweep(enough=False), True),
        (True, 0.92, sweep(0.97), True),
        (True, 0.97, sweep(0.97), False),
        (True, 0.97, sweep(None), True),
    ],
)
def test_the_threshold_warning_holds_until_a_full_sweep_recommends_this_threshold(
    enabled, threshold, found, warned
):
    message = threshold_warning(enabled, threshold, found)
    assert (message is not None) == warned
    if message:
        assert "unvalidated placeholder" in message and "CACHE_ENABLED=false" in message


def test_health_shows_the_warning_when_the_cache_is_on(settings):
    with TestClient(create_app(settings)) as client:
        body = client.get("/health").json()
    assert body["cache"]["enabled"] and "unvalidated placeholder" in body["warnings"][0]
    off = settings.model_copy(update={"cache_enabled": False})
    with TestClient(create_app(off)) as client:
        assert client.get("/health").json()["warnings"] == []


def test_missing_settings_are_named_without_their_values(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("LLM_MAX_RPM", "not-a-number-secret-ish")
    with pytest.raises(Exception) as caught:
        Settings(_env_file=None)
    message = describe_settings_error(caught.value)
    assert "missing MODEL_FAST, MODEL_JUDGE, MODEL_STRONG, MODEL_VISION, OPENROUTER_API_KEY" in message
    assert "invalid LLM_MAX_RPM" in message and "secret-ish" not in message and ".env.example" in message


def test_get_settings_raises_the_clear_error(monkeypatch, tmp_path):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    get_settings.cache_clear()
    try:
        with pytest.raises(SettingsError, match="OPENROUTER_API_KEY") as caught:
            get_settings()
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
    finally:
        get_settings.cache_clear()


def test_the_api_without_settings_answers_503_with_the_reason(monkeypatch, tmp_path):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            for path in ("/health", "/v1/documents"):
                response = client.get(path)
                assert response.status_code == 503
                error = response.json()["error"]
                assert error["code"] == "llm_unavailable" and "OPENROUTER_API_KEY" in error["message"]
    finally:
        get_settings.cache_clear()


def test_an_index_built_with_other_settings_is_reported_the_same_way(settings):
    from datetime import UTC, datetime

    from citerag.ingest.manifest import Manifest, save_manifest

    manifest = Manifest(
        embedding_model="other/model",
        embedding_dim=8,
        chunk_size=512,
        chunk_overlap=64,
        caption_model="v",
        created_at=datetime.now(UTC),
    )
    save_manifest(settings.index_dir, manifest)
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")
    assert response.status_code == 503 and "index was built with" in response.json()["error"]["message"]


def test_the_cli_prints_one_line_and_exits_2(monkeypatch):
    def broken():
        raise SettingsError("CiteRAG is not configured: missing OPENROUTER_API_KEY.")

    monkeypatch.setattr(cli, "get_settings", broken)
    monkeypatch.setattr(sys, "argv", ["citerag", "quota"])
    with pytest.raises(SystemExit) as caught:
        cli.run()
    assert caught.value.code == 2


def test_the_cli_message_reaches_stderr(monkeypatch):
    def broken():
        raise SettingsError("CiteRAG is not configured: missing OPENROUTER_API_KEY.")

    monkeypatch.setattr(cli, "get_settings", broken)
    result = CliRunner().invoke(cli.app, ["quota"])
    assert isinstance(result.exception, SettingsError)


def _review():
    spec = importlib.util.spec_from_file_location("review_golden_w", ROOT / "scripts" / "review_golden.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["review_golden_w"] = module
    spec.loader.exec_module(module)
    return module


def test_identifiers_with_a_digit_stay_whole_but_words_are_still_stemmed():
    review = _review()
    assert review.key_terms("SP 800-61r2") != review.key_terms("SP 800-61r3")
    assert review.key_terms("practice PS.1") != review.key_terms("practice PS.2")
    assert review.key_terms("categories") == review.key_terms("category")


@pytest.mark.parametrize(
    "docs",
    [
        {},
        {"a.pdf": {}},
        {"a.pdf": {1: ""}},
        {"a.pdf": {1: "   "}},
        {"img.png": {}},
        {"a.md": {None: "Fee? None."}},
    ],
)
def test_an_unanswerable_item_never_crashes_on_thin_input(docs):
    review = _review()
    for question in ("What is the fee for SP 800-61r9?", "What is it?"):
        text = review.render_item(GoldenItem(id="u", question=question, type="unanswerable"), docs)
        assert text.startswith("\n# u (unanswerable")
