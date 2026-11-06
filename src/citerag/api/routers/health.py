"""GET /health (F-OPS-3). Package versions come from metadata, so this never imports the frameworks."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from citerag.api.deps import get_app_settings, get_cache
from citerag.cache.base import CacheBackend, CacheStats
from citerag.config import Settings
from citerag.llm.budget import requests_last_24h

router = APIRouter()
_PACKAGES = (
    "citerag",
    "fastapi",
    "pydantic",
    "llama-index-core",
    "chromadb",
    "langchain",
    "langchain-openai",
    "crewai",
    "streamlit",
)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    versions: dict[str, str]
    manifest: dict | None = None  # the index manifest, from M1
    namespace: str | None = None  # the cache namespace, from M1
    cache: CacheStats
    llm_requests_24h: int
    max_requests_per_day: int


def _versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found


@router.get("/health")
def health(
    settings: Annotated[Settings, Depends(get_app_settings)],
    cache: Annotated[CacheBackend, Depends(get_cache)],
) -> HealthResponse:
    return HealthResponse(
        versions=_versions(),
        cache=cache.stats(),
        llm_requests_24h=requests_last_24h(settings.logs_db_path),
        max_requests_per_day=settings.max_requests_per_day,
    )
