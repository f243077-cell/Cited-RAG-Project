from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from citerag.config import Settings

EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_every_env_example_key_is_a_setting_and_back():
    assert {k.lower() for k in dotenv_values(EXAMPLE)} == set(Settings.model_fields)


def test_env_example_loads_with_documented_defaults():
    cfg = Settings(_env_file=EXAMPLE, openrouter_api_key="k")
    assert cfg.llm_max_rpm == 15 and cfg.max_requests_per_day == 40
    assert cfg.gemini_max_rpm == 5 and cfg.gemini_max_requests_per_day == 18 and cfg.gemini_timeout_s == 45
    assert cfg.provider_vision == cfg.provider_judge == "openrouter"
    assert cfg.llm_temperature is None and cfg.judge_temperature == 0
    assert cfg.citerag_api_key is None
    assert cfg.min_top_similarity == 0.55 and cfg.cache_threshold == 0.92
    assert cfg.cors_origin_list == ["http://localhost:8501"]


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="openrouter_api_key"):
        Settings(_env_file=EXAMPLE)


def test_model_ids_have_no_default_in_code(settings):
    for role in ("fast", "strong", "vision", "judge"):
        assert Settings.model_fields[f"model_{role}"].is_required()


def test_key_is_masked(settings):
    assert "test-key" not in repr(settings) and "test-key" not in str(settings.model_dump())
    assert settings.openrouter_api_key.get_secret_value() == "test-key"


def test_paths_follow_data_dir(settings, tmp_path):
    assert settings.logs_db_path == tmp_path / "logs" / "app.sqlite"


def test_a_fresh_clone_uses_no_gemini_role():
    roles = ("fast", "strong", "vision", "judge")
    assert {Settings.model_fields[f"provider_{role}"].default for role in roles} == {"openrouter"}
