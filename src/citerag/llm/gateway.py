"""The one LLM path (ARCHITECTURE section 11). The SDK runs with max_retries=0, so this file owns retries, parsing,
rate limiting and the daily guard, and logs every HTTP attempt."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Generic, Literal, Protocol, TypeVar

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from citerag.analytics.logs import LLMCallRow, LogStore
from citerag.config import Provider, Settings
from citerag.errors import LLMUnavailableError, StructuredOutputError
from citerag.eval.metrics import estimate_cost_usd
from citerag.llm.budget import check_budget
from citerag.llm.errors_map import Classified, classify_error
from citerag.llm.parse import parse_model, schema_response_format
from citerag.llm.pricing import load_pricing
from citerag.llm.ratelimit import SlidingWindowLimiter

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Role = Literal["fast", "strong", "vision", "judge"]
ROLES: tuple[Role, ...] = ("fast", "strong", "vision", "judge")
_MAX_RETRY_AFTER_S = 30  # a longer wait means the quota is gone, and a chat request should not hang on it
_OPTIONAL_PARAMETERS = ("temperature", "top_p", "top_k")
_NO_PROVIDER = ("no provider", "no endpoints")


class StructuredResult(BaseModel, Generic[T]):
    parsed: T
    model: str  # the model that answered
    requests: int  # HTTP attempts used (1 if no retry)
    input_tokens: int
    output_tokens: int
    latency_ms: int
    shadow_usd: float = 0.0


class LLMGateway(Protocol):
    async def structured_call(
        self,
        *,
        role: Role,
        schema: type[T],
        system: str,
        user: str | list[dict],
        call_name: str,
        request_id: str | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> StructuredResult[T]: ...


def provider_of(settings: Settings, role: str) -> Provider:
    return getattr(settings, f"provider_{role}")


def build_client(settings: Settings, role: str) -> ChatOpenAI:
    """max_retries=0 keeps every HTTP attempt visible to the gateway. Sampling parameters go out only if configured.
    Gemini gets no OpenRouter routing body or attribution headers."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    temperature = settings.judge_temperature if role == "judge" else settings.llm_temperature
    sampling = {} if temperature is None else {"temperature": temperature}
    common = {
        "model": getattr(settings, f"model_{role}"),
        "max_retries": 0,
        "max_tokens": settings.llm_max_tokens,
        **sampling,
    }
    if provider_of(settings, role) == "gemini":
        if settings.gemini_api_key is None:
            raise LLMUnavailableError(f"role {role} uses gemini, but GEMINI_API_KEY is not set")
        return ChatOpenAI(
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key,
            timeout=settings.gemini_timeout_s,
            **common,
        )
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        timeout=settings.llm_timeout_s,
        default_headers={"HTTP-Referer": settings.openrouter_referer, "X-Title": settings.openrouter_title},
        extra_body={
            "provider": {
                "require_parameters": settings.openrouter_require_parameters,
                "data_collection": settings.openrouter_data_collection,
            }
        },
        **common,
    )


def _text(raw: AIMessage) -> str:
    if isinstance(raw.content, str):
        return raw.content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw.content)


def _parse(schema: type[T], raw: AIMessage) -> T:
    if raw.tool_calls:
        return schema.model_validate(raw.tool_calls[0]["args"])
    return parse_model(schema, _text(raw))


def _with_feedback(user: str | list[dict], exc: Exception) -> str | list[dict]:
    """F-STR-1: feed the error back. Error details leave out the input, so no model output is echoed."""
    detail = exc.errors(include_input=False, include_url=False) if isinstance(exc, ValidationError) else exc
    note = f"Your last reply failed validation: {detail}. Reply again with only the JSON object."
    if isinstance(user, str):
        return f"{user}\n\n{note}"
    return [*user, {"type": "text", "text": note}]


class OpenRouterGateway:
    def __init__(
        self,
        settings: Settings,
        log: LogStore,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._cfg, self._log = settings, log
        self._limiters: dict[Provider, SlidingWindowLimiter] = {
            "openrouter": SlidingWindowLimiter(settings.llm_max_rpm),
            "gemini": SlidingWindowLimiter(settings.gemini_max_rpm),
        }
        self._daily_limits: dict[Provider, int] = {
            "openrouter": settings.max_requests_per_day,
            "gemini": settings.gemini_max_requests_per_day,
        }
        self._sleep = sleep
        self._pricing = load_pricing()
        self._clients: dict[str, ChatOpenAI] = {}
        self._methods: dict[str, str] = {}  # per model, once one method has failed

    async def reserve(self, provider: Provider, n: int) -> None:
        """For a crew, which calls the provider outside this class: guard first, then n limiter slots."""
        check_budget(self._cfg.logs_db_path, self._daily_limits[provider], provider=provider, need=n)
        while (wait := self._limiters[provider].try_acquire(n)) > 0:
            await self._sleep(wait)

    def _client(self, role: str) -> ChatOpenAI:
        if role not in self._clients:
            self._clients[role] = build_client(self._cfg, role)
        return self._clients[role]

    def _bind(
        self, role: str, schema: type[BaseModel], method: str, max_tokens: int, timeout_s: float | None
    ) -> Runnable[LanguageModelInput, AIMessage]:
        llm = self._client(role)
        limits: dict[str, float] = {"max_tokens": max_tokens}
        if timeout_s is not None:
            limits["timeout"] = timeout_s
        if method == "tools":
            return llm.bind_tools([schema], tool_choice=schema.__name__, **limits)
        return llm.bind(response_format=schema_response_format(schema), **limits)

    def _switch_method(self, model: str) -> bool:
        """STRUCTURED_METHOD=auto: move a model from json_schema to tools, once. False when already there."""
        if self._cfg.structured_method != "auto" or self._methods.get(model) == "tools":
            return False
        self._methods[model] = "tools"
        logger.info("%s: switching to the tool-calling method", model)
        return True

    def _drop_sampling(self, role: str, exc: Exception) -> bool:
        """A 400 that names an optional parameter: rebuild the client without sampling parameters, once."""
        temperature = self._client(role).temperature
        if temperature is None or not any(name in str(exc).lower() for name in _OPTIONAL_PARAMETERS):
            return False
        bare = self._cfg.model_copy(update={"llm_temperature": None, "judge_temperature": None})
        self._clients[role] = build_client(bare, role)
        logger.warning("%s rejected an optional parameter; retrying without it", role)
        return True

    def _retry_wait(
        self, seen: Classified, exc: Exception, provider: Provider, model: str, attempt: int
    ) -> float:
        """Seconds to wait before the next attempt, or raises when retrying cannot help."""
        if seen.kind == "auth":
            raise LLMUnavailableError(
                f"{provider} rejected the API key; check {provider.upper()}_API_KEY"
            ) from exc
        if seen.kind == "credits" and not seen.retry:
            raise LLMUnavailableError(f"{provider} reports insufficient credits") from exc
        if not seen.retry:
            detail = f"{seen.kind}, {type(exc).__name__}, HTTP {seen.http_status}"
            raise LLMUnavailableError(f"{model} failed ({detail})") from exc
        if seen.retry_after_s is not None and seen.retry_after_s > _MAX_RETRY_AFTER_S:
            raise LLMUnavailableError(f"{model} is rate limited for {seen.retry_after_s:.0f} s") from exc
        if seen.http_status == 503 and any(text in str(exc).lower() for text in _NO_PROVIDER):
            self._switch_method(model)
        return seen.retry_after_s if seen.retry_after_s is not None else float(2**attempt)

    async def structured_call(
        self,
        *,
        role: Role,
        schema: type[T],
        system: str,
        user: str | list[dict],
        call_name: str,
        request_id: str | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> StructuredResult[T]:
        """timeout_s caps the whole call, retries and waits included, not each attempt."""
        requested = getattr(self._cfg, f"model_{role}")
        provider = provider_of(self._cfg, role)
        self._client(role)  # a missing key fails here, before anything is logged as an attempt
        default_method = "tools" if self._cfg.structured_method == "tools" else "json_schema"
        limit = max_tokens or self._cfg.llm_max_tokens
        content, started = user, time.perf_counter()
        attempt_limit = self._cfg.gemini_timeout_s if provider == "gemini" else self._cfg.llm_timeout_s
        input_tokens = output_tokens = parse_failures = 0
        shadow_usd = 0.0
        raised_limit = False
        last_kind = "unknown"

        def record(attempt: int, began: float, model: str, ok: bool, **fields: object) -> None:
            self._log.write_llm_call(
                LLMCallRow(
                    request_id=request_id,
                    ts=time.time(),
                    role=call_name,
                    provider=provider,
                    model=model,
                    attempt=attempt,
                    latency_ms=int((time.perf_counter() - began) * 1000),
                    ok=ok,
                    **fields,
                )
            )

        for attempt in range(1, self._cfg.llm_max_retries + 2):
            while (wait := self._limiters[provider].try_acquire()) > 0:
                await self._sleep(wait)
            check_budget(self._cfg.logs_db_path, self._daily_limits[provider], provider=provider)
            method = self._methods.get(requested, default_method)
            messages = [SystemMessage(system), HumanMessage(content)]
            began = time.perf_counter()
            left = None
            if timeout_s is not None:
                left = timeout_s - (began - started)
                if left <= 0:
                    raise LLMUnavailableError(f"{requested} used its {timeout_s:.0f} s limit ({call_name})")
                left = min(left, attempt_limit)
            try:
                raw = await self._bind(role, schema, method, limit, left).ainvoke(messages)
            except Exception as exc:
                seen = classify_error(exc)
                last_kind = (
                    "rate_limit" if seen.http_status == 429 else seen.kind
                )  # a 429 can arrive in a 200 body
                if seen.kind == "unknown":  # only the type: an exception message can echo prompt text
                    last_kind = f"unknown:{type(exc).__name__}"
                    logger.warning("%s: unclassified %s from %s", call_name, type(exc).__name__, requested)
                record(attempt, began, requested, False, http_status=seen.http_status, error_kind=last_kind)
                if provider == "gemini" and (seen.kind == "timeout" or seen.http_status == 504):
                    # every attempt counts against a small daily quota, and a hang rarely clears in seconds
                    raise LLMUnavailableError(f"{requested} timed out; not retried") from exc
                if seen.kind == "bad_request" and (
                    self._drop_sampling(role, exc) or self._switch_method(requested)
                ):
                    continue
                pause = self._retry_wait(seen, exc, provider, requested, attempt)
                if attempt <= self._cfg.llm_max_retries:
                    await self._sleep(pause)
                continue

            usage = raw.usage_metadata or {}
            tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            served = raw.response_metadata.get("model_name") or requested
            cost = estimate_cost_usd(served, tokens_in, tokens_out, self._pricing)
            input_tokens, output_tokens = input_tokens + tokens_in, output_tokens + tokens_out
            shadow_usd += cost
            counts = {"input_tokens": tokens_in, "output_tokens": tokens_out, "shadow_usd": cost}
            try:
                parsed = _parse(schema, raw)
            except (ValidationError, ValueError) as exc:
                record(attempt, began, served, False, http_status=200, error_kind="parse", **counts)
                last_kind = "parse"
                cut_short = raw.response_metadata.get("finish_reason") == "length"
                if (cut_short or not (_text(raw).strip() or raw.tool_calls)) and not raised_limit:
                    raised_limit, limit = True, limit * 2  # hidden reasoning can use the whole limit
                    continue
                parse_failures += 1
                if parse_failures >= 2:
                    break
                self._switch_method(requested)
                content = _with_feedback(user, exc)
                continue

            record(attempt, began, served, True, http_status=200, **counts)
            return StructuredResult[schema](
                parsed=parsed,
                model=served,
                requests=attempt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                shadow_usd=shadow_usd,
            )

        if last_kind == "parse":
            raise StructuredOutputError(f"{requested} gave no valid {schema.__name__} ({call_name})")
        raise LLMUnavailableError(f"{requested} is unavailable after retries: {last_kind}")
