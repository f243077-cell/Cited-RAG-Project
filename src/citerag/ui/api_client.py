"""HTTP client for the backend. The UI and `citerag ingest --via-api` use only this (F-UI-6)."""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from citerag.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    ErrorResponse,
    IngestReport,
    RunDetail,
    RunSummary,
)

_TIMEOUT = httpx.Timeout(300.0, connect=5.0)  # an answer can take several slow free-model calls


class ApiError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=_TIMEOUT)

    def _send(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError("unreachable", f"The backend did not respond: {type(exc).__name__}") from exc
        if response.is_success:
            return response
        try:
            error = ErrorResponse.model_validate_json(response.text).error
        except ValidationError:
            raise ApiError("internal_error", f"HTTP {response.status_code} from the backend") from None
        raise ApiError(error.code, error.message)

    def chat(self, request: ChatRequest) -> ChatResponse:
        response = self._send("POST", "/v1/chat", json=request.model_dump())
        return ChatResponse.model_validate_json(response.text)

    def documents(self) -> list[DocumentInfo]:
        return [DocumentInfo.model_validate(d) for d in self._send("GET", "/v1/documents").json()]

    def upload(self, files: list[tuple[str, bytes]], captions: bool = False) -> list[IngestReport]:
        parts = [("files", (name, data)) for name, data in files]
        fields = {"captions": str(captions).lower()}
        response = self._send("POST", "/v1/documents", files=parts, data=fields)
        return [IngestReport.model_validate(r) for r in response.json()]

    def delete_document(self, doc_id: str) -> None:
        self._send("DELETE", f"/v1/documents/{doc_id}")

    def delete_session(self, session_id: str) -> None:
        self._send("DELETE", f"/v1/sessions/{session_id}")

    def eval_runs(self) -> list[RunSummary]:
        return [RunSummary.model_validate(r) for r in self._send("GET", "/v1/eval/runs").json()]

    def eval_run(self, run_id: str) -> RunDetail:
        return RunDetail.model_validate_json(self._send("GET", f"/v1/eval/runs/{run_id}").text)

    def cache_status(self) -> dict:
        return self._send("GET", "/v1/cache/stats").json()

    def clear_cache(self) -> None:
        self._send("POST", "/v1/cache/clear")

    def request_log(self, limit: int = 1000) -> list[dict]:
        return self._send("GET", "/v1/logs/requests", params={"limit": limit}).json()

    def llm_call_log(self, limit: int = 1000) -> list[dict]:
        return self._send("GET", "/v1/logs/llm_calls", params={"limit": limit}).json()

    def cache_sweep(self) -> dict | None:
        return self._send("GET", "/v1/eval/cache-sweep").json()

    def health(self) -> dict:
        return self._send("GET", "/health").json()
