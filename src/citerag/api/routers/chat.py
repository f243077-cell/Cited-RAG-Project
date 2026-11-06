"""POST /v1/chat and the session history endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from citerag.analytics.logs import LogStore
from citerag.api.deps import Services, get_log, get_services, require_api_key
from citerag.rag.pipeline import answer_question
from citerag.schemas import ChatRequest, ChatResponse, Message

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.post("/chat")
async def chat(req: ChatRequest, services: Annotated[Services, Depends(get_services)]) -> ChatResponse:
    return await answer_question(req, services.deps)


@router.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, log: Annotated[LogStore, Depends(get_log)]) -> list[Message]:
    return log.messages(session_id)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, log: Annotated[LogStore, Depends(get_log)]) -> Response:
    log.delete_session(session_id)
    return Response(status_code=204)
