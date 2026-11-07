"""Stand-in for the LLM gateway in pipeline tests: returns queued objects by call_name, makes no network call."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from pydantic import BaseModel

from citerag.llm.gateway import Role, StructuredResult, T
from citerag.schemas import AgentAction


@dataclass(frozen=True)
class FakeCall:
    role: str
    schema: type[BaseModel]
    system: str
    user: str | list[dict]
    call_name: str
    request_id: str | None
    max_tokens: int | None
    timeout_s: float | None = None
    tools: list[dict] | None = None


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[FakeCall] = []
        self.room_checks: list[tuple[str, int]] = []
        self.method = "tools"
        self._queues: dict[str, deque[BaseModel | Exception]] = defaultdict(deque)

    def queue(self, call_name: str, item: BaseModel | Exception) -> None:
        """Queue the next result for call_name. Queue an exception to make that call raise it."""
        self._queues[call_name].append(item)

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
        tools: list[dict] | None = None,
    ) -> StructuredResult[T]:
        self.calls.append(
            FakeCall(role, schema, system, user, call_name, request_id, max_tokens, timeout_s, tools)
        )
        if not self._queues[call_name]:
            raise LookupError(f"FakeGateway: nothing queued for {call_name!r}")
        item = self._queues[call_name].popleft()
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, schema):
            raise TypeError(
                f"queued {type(item).__name__} for {call_name!r}, caller expects {schema.__name__}"
            )
        return StructuredResult[schema](
            parsed=item, model=f"fake/{role}", requests=1, input_tokens=0, output_tokens=0, latency_ms=0
        )

    async def tool_step(
        self,
        *,
        role: Role,
        system: str,
        user: str,
        tools: list[dict],
        call_name: str,
        request_id: str | None = None,
    ) -> StructuredResult[AgentAction]:
        return await self.structured_call(
            role=role,
            schema=AgentAction,
            system=system,
            user=user,
            call_name=call_name,
            request_id=request_id,
            tools=tools,
        )

    def check_room(self, role: Role, n: int) -> None:
        """Queue an exception under "room" to make the check fail."""
        self.room_checks.append((role, n))
        if self._queues["room"]:
            raise self._queues["room"].popleft()

    def action_method(self, role: Role) -> str:
        return self.method
