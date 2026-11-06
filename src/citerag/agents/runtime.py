"""What every crew run shares: LLM objects, the request guard, the timeout and logging.

CrewAI calls the provider outside the gateway, so this file does by hand what the gateway does per call. It takes
limiter slots and checks the daily guard first, then writes llm_calls rows from the crew's usage. The guard counts
rows, so a crew writes one row per request; a single row would let a Deep question count as one request.

CrewAI imports only inside functions: it is slow to load, and most processes never run a crew. Nothing from CrewAI
leaves this package.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from citerag.analytics.logs import LLMCallRow, LogStore
from citerag.config import Settings
from citerag.errors import BudgetExceededError, LLMUnavailableError, StructuredOutputError
from citerag.eval.metrics import estimate_cost_usd
from citerag.llm.errors_map import Classified, classify_error
from citerag.llm.gateway import OpenRouterGateway, provider_of
from citerag.llm.pricing import load_pricing

if TYPE_CHECKING:
    from crewai import LLM, Crew

logger = logging.getLogger(__name__)
MAX_ITER = 2  # no tools, so one step is the normal case and a second is the only repair CrewAI may try
PROVIDER = "openrouter"
_HIDDEN_429_ATTEMPTS = 2  # CrewAI retries a 429 twice inside one LLM call, and no setting turns that off


class CrewUsage(BaseModel):
    role: str
    model: str
    requests: int
    input_tokens: int
    output_tokens: int
    shadow_usd: float
    latency_ms: int


@dataclass(frozen=True)
class CrewRun:
    outputs: list[BaseModel]  # one per task, in task order
    usage: CrewUsage


def prepare_environment(settings: Settings) -> None:
    """These must be set before crewai is imported. Writes only: config still comes from Settings."""
    os.environ["CREWAI_DISABLE_TELEMETRY"] = str(settings.crewai_disable_telemetry).lower()
    os.environ["OTEL_SDK_DISABLED"] = str(settings.otel_sdk_disabled).lower()
    os.environ["CREWAI_TRACING_ENABLED"] = "false"


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """CrewAI wraps provider errors in RuntimeError, and its chains can be circular."""
    found: BaseException | None = exc
    seen: set[int] = set()
    while found is not None and id(found) not in seen:
        seen.add(id(found))
        yield found
        found = found.__cause__ or found.__context__


def _is_bad_output(exc: BaseException) -> bool:
    """CrewAI parses strictly, unlike llm/parse.py."""
    return any(isinstance(e, ValidationError) or "validation error for" in str(e) for e in _chain(exc))


def _classify(exc: BaseException) -> Classified:
    known = (seen for seen in map(classify_error, _chain(exc)) if seen.kind != "unknown")
    return next(known, classify_error(exc))


class CrewRuntime:
    def __init__(self, settings: Settings, gateway: OpenRouterGateway, log: LogStore) -> None:
        self._cfg, self._gateway, self._log = settings, gateway, log
        self._pricing = load_pricing()

    @property
    def settings(self) -> Settings:
        return self._cfg

    def llm(self, role: Literal["fast", "strong"]) -> LLM:
        """Call once per agent: agents sharing one object made usage read 9 requests for 3 real ones."""
        if provider_of(self._cfg, role) != PROVIDER:
            raise LLMUnavailableError(f"crews run on OpenRouter only, and role {role} uses another provider")
        prepare_environment(self._cfg)
        from crewai import LLM

        sampling = {} if self._cfg.llm_temperature is None else {"temperature": self._cfg.llm_temperature}
        return LLM(
            model=f"openrouter/{getattr(self._cfg, f'model_{role}')}",
            base_url=self._cfg.openrouter_base_url,
            api_key=self._cfg.openrouter_api_key.get_secret_value(),
            max_tokens=self._cfg.llm_max_tokens,
            timeout=self._cfg.llm_timeout_s,
            max_retries=0,  # never SDK-level retries: each attempt must be visible
            default_headers={
                "HTTP-Referer": self._cfg.openrouter_referer,
                "X-Title": self._cfg.openrouter_title,
            },
            extra_body={
                "provider": {
                    "require_parameters": self._cfg.openrouter_require_parameters,
                    "data_collection": self._cfg.openrouter_data_collection,
                }
            },
            **sampling,
        )

    def agent_limits(self) -> dict[str, Any]:
        """respect_context_window=False: with an unknown model CrewAI assumes 8,192 tokens and would summarise the
        passages. A real overflow should fail, not quietly change what the agents read."""
        return {
            "allow_delegation": False,
            "verbose": False,
            "max_iter": MAX_ITER,
            "max_retry_limit": 0,  # CrewAI would rerun a failed task twice, outside the guard
            "max_execution_time": self._cfg.crew_max_seconds,
            "respect_context_window": False,
        }

    def _write_rows(self, usage: CrewUsage, request_id: str | None, ok: bool, kind: str | None) -> None:
        """Tokens go on the first row, so totals stay right whichever way the rows are summed."""
        for attempt in range(1, usage.requests + 1):
            first = attempt == 1
            self._log.write_llm_call(
                LLMCallRow(
                    request_id=request_id,
                    ts=time.time(),
                    role=usage.role,
                    provider=PROVIDER,
                    model=usage.model,
                    attempt=attempt,
                    http_status=200 if ok else None,
                    error_kind=kind,
                    input_tokens=usage.input_tokens if first else 0,
                    output_tokens=usage.output_tokens if first else 0,
                    shadow_usd=usage.shadow_usd if first else 0.0,
                    latency_ms=usage.latency_ms if first else 0,
                    ok=ok,
                )
            )

    def _failed(
        self, role: str, model: str, reserved: int, started: float, request_id: str | None, kind: str
    ) -> None:
        """A failed crew reports no usage. Counting every reserved slot errs towards the stricter guard."""
        latency = int((time.perf_counter() - started) * 1000)
        usage = CrewUsage(
            role=role,
            model=model,
            requests=reserved,
            input_tokens=0,
            output_tokens=0,
            shadow_usd=0.0,
            latency_ms=latency,
        )
        self._write_rows(usage, request_id, False, kind)

    async def kickoff(
        self,
        crew: Crew,
        *,
        role: str,
        model: str,
        inputs: dict[str, str],
        request_id: str | None = None,
    ) -> CrewRun:
        """Raises budget_exceeded before any HTTP request when the guard lacks a slot for each task."""
        reserved = len(crew.tasks)
        await self._gateway.reserve(PROVIDER, reserved)
        started = time.perf_counter()
        limit = self._cfg.crew_max_seconds
        try:
            result = await asyncio.wait_for(asyncio.to_thread(crew.kickoff, inputs=inputs), limit)
        except BudgetExceededError:
            raise
        except TimeoutError as exc:
            self._failed(role, model, reserved, started, request_id, "timeout")
            raise LLMUnavailableError(f"crew {role} ran past its {limit} s limit") from exc
        except Exception as exc:
            if _is_bad_output(exc):
                self._failed(role, model, reserved, started, request_id, "parse")
                raise StructuredOutputError(f"crew {role} gave output that fits no schema") from exc
            seen = _classify(exc)
            kind = f"unknown:{type(exc).__name__}" if seen.kind == "unknown" else seen.kind
            logger.warning("crew %s failed: %s", role, kind)  # only the kind: a message can echo prompt text
            attempts = reserved + (_HIDDEN_429_ATTEMPTS if seen.kind == "rate_limit" else 0)
            self._failed(role, model, attempts, started, request_id, kind)
            raise LLMUnavailableError(f"crew {role} failed ({kind})") from exc

        outputs = [task.pydantic for task in result.tasks_output]
        tokens = result.token_usage
        requests = max(tokens.successful_requests, reserved if None in outputs else 0)
        usage = CrewUsage(
            role=role,
            model=model,
            requests=requests,
            input_tokens=tokens.prompt_tokens,
            output_tokens=tokens.completion_tokens,
            shadow_usd=estimate_cost_usd(
                model, tokens.prompt_tokens, tokens.completion_tokens, self._pricing
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        if None in outputs:
            self._write_rows(usage, request_id, False, "parse")
            raise StructuredOutputError(f"crew {role} gave output that fits no schema")
        self._write_rows(usage, request_id, True, None)
        return CrewRun(outputs=outputs, usage=usage)
