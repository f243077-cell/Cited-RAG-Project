"""Classify exceptions raised while calling an OpenAI-compatible provider (OpenRouter) so the gateway knows what to do.

The gateway runs the SDK with max_retries=0 and retries itself, so every HTTP attempt is logged and counted against
the daily request limit. Findings from the spike that this module encodes:
  * HTTP 200 with an {"error": ...} body makes langchain-openai raise ValueError (plain call) or TypeError
    (structured path), not an openai.* exception.
  * 429 raises openai.RateLimitError and the Retry-After header is available on e.response.headers.
"""

from __future__ import annotations

from dataclasses import dataclass

import openai
from pydantic import ValidationError


@dataclass(frozen=True)
class Classified:
    kind: str  # rate_limit | overloaded | timeout | server | provider_body | parse | auth | credits | bad_request | unknown
    retry: bool
    retry_after_s: float | None = None
    http_status: int | None = None


def _retry_after(exc: openai.APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def classify_error(exc: BaseException) -> Classified:
    if isinstance(exc, openai.APITimeoutError):
        return Classified("timeout", True)
    if isinstance(exc, openai.APIConnectionError):
        return Classified("server", True)
    if isinstance(exc, openai.APIStatusError):
        s = exc.status_code
        if s == 429:
            return Classified("rate_limit", True, _retry_after(exc), s)
        if s == 503:
            return Classified("overloaded", True, _retry_after(exc), s)
        if s in (408, 500, 502, 504):
            return Classified("timeout" if s == 408 else "server", True, None, s)
        if s == 402:  # without Retry-After this is "add credits", not "wait"
            ra = _retry_after(exc)
            return Classified("credits", ra is not None, ra, s)
        if s in (401, 403):
            return Classified("auth", False, None, s)
        if s in (400, 404, 422):
            return Classified(
                "bad_request", False, None, s
            )  # e.g. model lacks response_format: try the tool-calling method
        return Classified("unknown", False, None, s)
    if isinstance(exc, (ValidationError, ValueError)) and not isinstance(exc, (openai.OpenAIError,)):
        # pydantic.ValidationError is a ValueError subclass; langchain raises ValueError(dict) for a 200 + error body
        if isinstance(exc, ValidationError):
            return Classified("parse", True)
        if exc.args and isinstance(exc.args[0], dict) and "code" in exc.args[0]:
            code = exc.args[0].get("code")
            return Classified("provider_body", True, None, code if isinstance(code, int) else None)
        return Classified("parse", True)
    if isinstance(exc, TypeError):
        return Classified("provider_body", True)  # 'NoneType' object is not iterable on a 200 + error body
    return Classified("unknown", False)
