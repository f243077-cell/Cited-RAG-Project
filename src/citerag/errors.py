"""Typed errors, one per public ErrorCode. The API turns each into an ErrorResponse with its HTTP status."""

from __future__ import annotations

from typing import ClassVar

from citerag.schemas import ErrorBody, ErrorCode, ErrorResponse


class CiteRAGError(Exception):
    code: ClassVar[ErrorCode] = "internal_error"
    http_status: ClassVar[int] = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_response(self, request_id: str | None = None) -> ErrorResponse:
        return ErrorResponse(error=ErrorBody(code=self.code, message=self.message, request_id=request_id))


class InternalError(CiteRAGError):
    code = "internal_error"
    http_status = 500


class InvalidFileTypeError(CiteRAGError):
    code = "invalid_file_type"
    http_status = 415


class FileTooLargeError(CiteRAGError):
    code = "file_too_large"
    http_status = 413


class BudgetExceededError(CiteRAGError):
    code = "budget_exceeded"
    http_status = 429


class LLMUnavailableError(CiteRAGError):
    code = "llm_unavailable"
    http_status = 503


class StructuredOutputError(CiteRAGError):
    code = "structured_output_failed"
    http_status = 502


class IndexMismatchError(CiteRAGError):
    code = "index_mismatch"
    http_status = 409


class NotFoundError(CiteRAGError):
    code = "not_found"
    http_status = 404


class UnauthorizedError(CiteRAGError):
    code = "unauthorized"
    http_status = 401


class ValidationFailedError(CiteRAGError):
    code = "validation_error"
    http_status = 400
