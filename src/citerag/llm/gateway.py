"""LLM gateway contract and per-role clients. The gateway itself (retries, parsing, logging) arrives in M3."""

from __future__ import annotations

from typing import Generic, Literal, Protocol, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from citerag.config import Settings

T = TypeVar("T", bound=BaseModel)
Role = Literal["fast", "strong", "vision", "judge"]
ROLES: tuple[Role, ...] = ("fast", "strong", "vision", "judge")


class StructuredResult(BaseModel, Generic[T]):
    parsed: T
    model: str  # the model that answered
    requests: int  # HTTP attempts used (1 if no retry)
    input_tokens: int
    output_tokens: int
    latency_ms: int


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
    ) -> StructuredResult[T]: ...


def build_client(settings: Settings, role: str) -> ChatOpenAI:
    """max_retries=0 keeps every HTTP attempt visible to the gateway. Sampling parameters go out only if configured."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    temperature = settings.judge_temperature if role == "judge" else settings.llm_temperature
    sampling = {} if temperature is None else {"temperature": temperature}
    return ChatOpenAI(
        model=getattr(settings, f"model_{role}"),
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        max_retries=0,
        timeout=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        default_headers={"HTTP-Referer": settings.openrouter_referer, "X-Title": settings.openrouter_title},
        extra_body={
            "provider": {
                "require_parameters": settings.openrouter_require_parameters,
                "data_collection": settings.openrouter_data_collection,
            }
        },
        **sampling,
    )
