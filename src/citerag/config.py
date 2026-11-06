"""Application settings. Every key in .env.example is a field here; nothing else reads the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: a blank line such as LLM_TEMPERATURE= means "use the default"
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str = "http://localhost:8501"
    openrouter_title: str = "CiteRAG"
    openrouter_require_parameters: bool = True
    openrouter_data_collection: Literal["allow", "deny"] = "allow"

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

    cache_enabled: bool = True
    cache_threshold: float = 0.92
    cache_ttl_days: int = 7
    cache_max_entries: int = 10_000

    max_upload_mb: int = 25
    max_images_per_doc: int = 10
    min_image_px: int = 100
    min_image_bytes: int = 5000

    crew_max_seconds: int = 120
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
    def logs_db_path(self) -> Path:
        return self.data_dir / "logs" / "app.sqlite"


@lru_cache
def get_settings() -> Settings:
    """Fails fast with a validation error when a required setting is missing (F-STR-3)."""
    return Settings()
