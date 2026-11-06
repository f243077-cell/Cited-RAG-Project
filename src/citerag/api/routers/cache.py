"""Cache admin (F-CAC-6)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from citerag.api.deps import get_app_settings, get_cache, require_api_key
from citerag.cache.base import CacheBackend, CacheStats
from citerag.config import Settings

router = APIRouter(prefix="/v1/cache", dependencies=[Depends(require_api_key)])


class CacheStatus(CacheStats):
    threshold: float  # the dashboard draws it on the similarity chart


@router.get("/stats")
def stats(
    cache: Annotated[CacheBackend, Depends(get_cache)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> CacheStatus:
    return CacheStatus(**cache.stats().model_dump(), threshold=settings.cache_threshold)


@router.post("/clear", status_code=204)
def clear(cache: Annotated[CacheBackend, Depends(get_cache)]) -> Response:
    cache.clear()
    return Response(status_code=204)
