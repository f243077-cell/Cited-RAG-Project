from typing import get_args

import pytest

from citerag import errors
from citerag.schemas import ErrorCode, ErrorResponse


def _all_error_classes():
    return [c for c in vars(errors).values() if isinstance(c, type) and issubclass(c, errors.CiteRAGError)]


def test_every_error_code_has_a_typed_error():
    codes = {c.code for c in _all_error_classes() if c is not errors.CiteRAGError}
    assert codes == set(get_args(ErrorCode))


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (errors.InvalidFileTypeError, 415),
        (errors.FileTooLargeError, 413),
        (errors.BudgetExceededError, 429),
        (errors.LLMUnavailableError, 503),
        (errors.StructuredOutputError, 502),
        (errors.NotFoundError, 404),
        (errors.UnauthorizedError, 401),
        (errors.ValidationFailedError, 400),
    ],
)
def test_documented_status_codes(cls, status):
    assert cls.http_status == status


def test_to_response_is_the_public_error_body():
    body = errors.BudgetExceededError("limit 40 reached").to_response(request_id="r1")
    assert isinstance(body, ErrorResponse)
    assert body.model_dump() == {
        "error": {"code": "budget_exceeded", "message": "limit 40 reached", "request_id": "r1"}
    }
