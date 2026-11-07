"""FastAPI entry point: uvicorn citerag.api.main:app"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from citerag import errors
from citerag.analytics.logs import LogStore
from citerag.api.deps import Services, build_services
from citerag.api.routers import cache, chat, documents, evaluation, health, logs, tools
from citerag.cache.base import CacheBackend
from citerag.cache.store import open_cache
from citerag.config import Settings, SettingsError, get_settings
from citerag.eval.cache_tools import load_sweep, threshold_warning
from citerag.ingest.manifest import check_manifest, current_namespace, load_manifest

logger = logging.getLogger(__name__)
_BY_HTTP_STATUS: dict[int, type[errors.CiteRAGError]] = {
    401: errors.UnauthorizedError,
    404: errors.NotFoundError,
}
ServicesFactory = Callable[[Settings, LogStore, CacheBackend], Services]


class _SettingsCORS:
    """CORS origins come from settings, and settings load at start-up, after middleware is declared."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._cors: CORSMiddleware | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        problem = getattr(scope["app"].state, "config_error", None)
        if problem is not None:  # not configured: every request says why, the UI shows it, nothing crashes
            body = errors.LLMUnavailableError(problem).to_response().model_dump()
            await JSONResponse(status_code=503, content=body)(scope, receive, send)
            return
        if self._cors is None:
            origins = scope["app"].state.settings.cors_origin_list
            self._cors = CORSMiddleware(
                self._app, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
            )
        await self._cors(scope, receive, send)


def _error_response(error: errors.CiteRAGError) -> JSONResponse:
    body = error.to_response(getattr(error, "request_id", None))
    return JSONResponse(status_code=error.http_status, content=body.model_dump())


def create_app(
    settings: Settings | None = None, services_factory: ServicesFactory = build_services
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config_error = None
        try:
            cfg = settings or get_settings()
            manifest = load_manifest(cfg.index_dir)
            if manifest is not None:
                check_manifest(manifest, cfg)  # F-ING-4: never serve vectors built with other settings
        except (SettingsError, errors.IndexMismatchError) as exc:
            app.state.config_error = getattr(exc, "message", str(exc))
            logger.error("%s", app.state.config_error)  # one clear line: no stack trace, no setting values
            yield
            return
        app.state.settings = cfg
        app.state.cache = open_cache(cfg)
        warning = threshold_warning(cfg.cache_enabled, cfg.cache_threshold, load_sweep())
        if warning:
            logger.warning("%s", warning)
        if manifest is not None:  # `citerag ingest` runs with the API stopped, so catch up here (ADR-7)
            app.state.cache.invalidate_except(current_namespace(cfg, manifest))
        app.state.log = LogStore(cfg.logs_db_path)
        app.state.services = None
        app.state.services_lock = threading.Lock()
        app.state.services_factory = services_factory
        yield
        if app.state.services is not None:
            app.state.services.close()

    app = FastAPI(title="CiteRAG", lifespan=lifespan)
    app.add_middleware(_SettingsCORS)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(evaluation.router)
    app.include_router(cache.router)
    app.include_router(logs.router)
    app.include_router(tools.router)

    @app.exception_handler(errors.CiteRAGError)
    async def on_typed_error(_: Request, exc: errors.CiteRAGError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(errors.ValidationFailedError(str(exc)))

    @app.exception_handler(HTTPException)
    async def on_http_error(_: Request, exc: HTTPException) -> JSONResponse:
        error_type = _BY_HTTP_STATUS.get(exc.status_code, errors.ValidationFailedError)
        return JSONResponse(
            status_code=exc.status_code, content=error_type(str(exc.detail)).to_response().model_dump()
        )

    @app.exception_handler(Exception)
    async def on_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return _error_response(errors.InternalError("Internal server error"))

    return app


app = create_app()
