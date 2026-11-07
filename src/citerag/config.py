"""Application settings. Every key in .env.example is a field here; nothing else reads the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openrouter", "gemini"]


class Settings(BaseSettings):
    # env_ignore_empty: a blank line such as LLM_TEMPERATURE= means "use the default"
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str = "http://localhost:8501"
    openrouter_title: str = "CiteRAG"
    openrouter_require_parameters: bool = True
    openrouter_data_collection: Literal["allow", "deny"] = "allow"

    gemini_api_key: SecretStr | None = None  # checked when a gemini role makes its first call
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_max_rpm: int = 5
    gemini_max_requests_per_day: int = 18
    gemini_timeout_s: float = 45

    provider_fast: Provider = "openrouter"
    provider_strong: Provider = "openrouter"
    provider_vision: Provider = "openrouter"
    provider_judge: Provider = "openrouter"

    # Model ids change often, so they have no default in code (F-OPS-7): they come from .env only.
    model_fast: str
    model_strong: str
    model_vision: str
    model_judge: str
    structured_method: Literal["auto", "json_schema", "tools"] = "auto"

    llm_timeout_s: float = 60
    llm_max_retries: int = 2
    llm_max_tokens: int = 2000
    llm_temperature: float | None = None
    judge_temperature: float | None = 0
    llm_max_rpm: int = 15
    max_requests_per_day: int = 40

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_k_context: int = 5
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_top_similarity: float = 0.30

    context_token_budget: int = 6000
    history_turns: int = 4

    cache_enabled: bool = (
        False  # off until a human-labelled sweep validates CACHE_THRESHOLD (README, HANDOFF.md)
    )
    cache_threshold: float = 0.92
    cache_ttl_days: int = 7
    cache_max_entries: int = 10_000

    max_upload_mb: int = 25
    max_images_per_doc: int = 10
    min_image_px: int = 100
    min_image_bytes: int = 5000

    crew_max_seconds: int = 120

    agent_enabled: bool = True
    agent_max_subquestions: int = 3
    agent_max_steps: int = 6  # LLM calls of plan, act and reflect; synthesis is outside it
    agent_recursion_limit: int = 25  # LangGraph's own backstop; the step count is the real limit
    agent_timeout_s: float = 240
    agent_context_max_passages: int = 8
    agent_action_method: Literal["auto", "tools", "json"] = "auto"
    agent_architecture: Literal["single", "supervisor"] = "single"  # supervisor: the M12 team, for comparison
    agent_tool_result_chars: int = 6000  # per tool call: hits are cut to this many characters
    mcp_mode: Literal["api", "local"] = "api"  # api: call the backend over HTTP; local: open the index here
    crewai_disable_telemetry: bool = True
    otel_sdk_disabled: bool = True

    data_dir: Path = Path("./data")
    cors_origins: str = "http://localhost:8501"
    citerag_api_key: SecretStr | None = None
    api_base_url: str = "http://localhost:8000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_db_path(self) -> Path:
        return self.data_dir / "cache" / "semantic_cache.sqlite"

    @property
    def logs_db_path(self) -> Path:
        return self.data_dir / "logs" / "app.sqlite"


class UISettings(BaseSettings):
    """What the Streamlit process needs. It never sees the OpenRouter key (F-UI-6)."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    api_base_url: str = "http://localhost:8000"
    citerag_api_key: SecretStr | None = None


class SettingsError(Exception):
    """Required settings are missing or invalid. The message names environment variables, never their values."""


def describe_settings_error(exc: ValidationError) -> str:
    missing = sorted({str(e["loc"][0]).upper() for e in exc.errors() if e["type"] == "missing" and e["loc"]})
    invalid = sorted({str(e["loc"][0]).upper() for e in exc.errors() if e["type"] != "missing" and e["loc"]})
    parts = ([f"missing {', '.join(missing)}"] if missing else []) + (
        [f"invalid {', '.join(invalid)}"] if invalid else []
    )
    return (
        f"CiteRAG is not configured: {'; '.join(parts)}. Copy .env.example to .env in the folder you run from, "
        "set OPENROUTER_API_KEY and the MODEL_* lines, then start again (README, Configuration)."
    )


@lru_cache
def get_settings() -> Settings:
    """Fails fast with a clear SettingsError when a required setting is missing (F-STR-3). `from None`: the validation
    error's own text would echo the invalid values."""
    try:
        return Settings()
    except ValidationError as exc:
        raise SettingsError(describe_settings_error(exc)) from None
