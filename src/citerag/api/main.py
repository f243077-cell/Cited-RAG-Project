"""FastAPI entry point: uvicorn citerag.api.main:app"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from citerag import errors
from citerag.api.routers import health
from citerag.cache.base import NoOpCache
from citerag.config import Settings, get_settings

logger = logging.getLogger(__name__)
_BY_HTTP_STATUS: dict[int, type[errors.CiteRAGError]] = {
    401: errors.UnauthorizedError,
    404: errors.NotFoundError,
}


def _error_response(error: errors.CiteRAGError) -> JSONResponse:
    return JSONResponse(status_code=error.http_status, content=error.to_response().model_dump())


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings or get_settings()  # a missing required setting stops start-up here
        app.state.cache = NoOpCache()
        yield

    app = FastAPI(title="CiteRAG", lifespan=lifespan)
    app.include_router(health.router)

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
