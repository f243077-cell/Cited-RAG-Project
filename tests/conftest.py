import pytest

from citerag.config import Settings


@pytest.fixture(autouse=True)
def _no_settings_in_the_environment(monkeypatch):
    """Importing crewai runs load_dotenv(), so .env would otherwise leak into every test that follows it."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings: no .env file, placeholder model ids, a temp data folder. The cache is on here so its paths stay
    tested; the shipped default is off (tests/unit/test_config.py)."""
    return Settings(
        _env_file=None,
        cache_enabled=True,
        openrouter_api_key="test-key",
        gemini_api_key="test-gemini-key",
        model_fast="vendor/fast:free",
        model_strong="vendor/strong:free",
        model_vision="vendor/vision:free",
        model_judge="other/judge:free",
        data_dir=tmp_path,
    )
