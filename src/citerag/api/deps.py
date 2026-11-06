"""Request-scoped access to objects created at start-up."""

from __future__ import annotations

from fastapi import Request

from citerag.cache.base import CacheBackend
from citerag.config import Settings


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_cache(request: Request) -> CacheBackend:
    return request.app.state.cache
