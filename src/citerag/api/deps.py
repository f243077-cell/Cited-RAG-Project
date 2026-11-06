"""Request-scoped access to objects created at start-up. Services load the embedding model and open Chroma, so
they are built on first use: /health and the document list stay light."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Header, Request

from citerag.analytics.logs import LogStore
from citerag.cache.base import CacheBackend
from citerag.config import Settings
from citerag.errors import UnauthorizedError
from citerag.rag.pipeline import Deps

if TYPE_CHECKING:
    from citerag.index.retriever import HybridRetriever
    from citerag.index.store import IndexStore
    from citerag.ingest.pipeline import Ingestor


@dataclass
class Services:
    deps: Deps
    store: IndexStore
    ingestor: Ingestor
    retriever: HybridRetriever

    def close(self) -> None:
        self.store.close()


def build_services(settings: Settings, log: LogStore, cache: CacheBackend) -> Services:
    from citerag.index.retriever import HybridRetriever
    from citerag.index.store import open_store
    from citerag.ingest.manifest import current_namespace
    from citerag.ingest.pipeline import Ingestor
    from citerag.llm.gateway import OpenRouterGateway

    store = open_store(settings, settings.index_dir)
    ingestor = Ingestor(settings, store, settings.index_dir, cache)
    retriever = HybridRetriever(store, settings)
    deps = Deps(
        cfg=settings,
        llm=OpenRouterGateway(settings, log),
        retriever=retriever,
        cache=cache,
        log=log,
        namespace=lambda: current_namespace(settings, ingestor.manifest),
    )
    return Services(deps, store, ingestor, retriever)


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_cache(request: Request) -> CacheBackend:
    return request.app.state.cache


def get_log(request: Request) -> LogStore:
    return request.app.state.log


def get_services(request: Request) -> Services:
    """A sync dependency, so the slow first build runs in the thread pool, not on the event loop."""
    state = request.app.state
    with state.services_lock:
        if state.services is None:
            state.services = state.services_factory(state.settings, state.log, state.cache)
        return state.services


def require_api_key(
    settings: Annotated[Settings, Depends(get_app_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Enforced only when CITERAG_API_KEY is set (F-API-1)."""
    expected = settings.citerag_api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        raise UnauthorizedError("Missing or wrong X-API-Key")
