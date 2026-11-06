"""Request and LLM-call rows for the dashboard (F-UI-3). Read-only."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from citerag.analytics.logs import LLMCallRow, LogStore, RequestRow
from citerag.api.deps import get_log, require_api_key

router = APIRouter(prefix="/v1/logs", dependencies=[Depends(require_api_key)])
Limit = Annotated[int, Query(ge=1, le=5000)]


@router.get("/requests")
def requests(
    log: Annotated[LogStore, Depends(get_log)], since: float = 0.0, limit: Limit = 1000
) -> list[RequestRow]:
    return log.recent_requests(since, limit)


@router.get("/llm_calls")
def llm_calls(
    log: Annotated[LogStore, Depends(get_log)], since: float = 0.0, limit: Limit = 1000
) -> list[LLMCallRow]:
    return log.recent_llm_calls(since, limit)
