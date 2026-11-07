"""M12: a supervisor with two workers (orchestrator-worker with handoffs), to compare with the single agent.

The supervisor reads the question and the evidence so far and hands off: to the researcher with a task, or to the writer.
The researcher is the single agent's loop on that task, with a smaller step budget. The writer is the verified synthesis in
rag/pipeline.py, so "answer" ends the graph here. Handoffs pass state (hits, steps), never free text into another prompt.

Worst case: 3 supervisor turns, 2 research handoffs of at most 3 steps each, synthesis and one retry: 11 requests.
"""

from __future__ import annotations

import time
from xml.sax.saxutils import escape

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from citerag.agent_graph.graph import AgentGraph, ToolRunner, to_chunk
from citerag.config import Settings
from citerag.errors import InternalError
from citerag.llm.gateway import LLMGateway
from citerag.prompts import load_prompt
from citerag.rag.context import build_context, render_passages
from citerag.rag.pipeline import AgentFindings
from citerag.schemas import AgentStep, AgentTrace, Route
from citerag.tools.contracts import ToolHit

SUPERVISOR_TURNS = 3
RESEARCH_HANDOFFS = 2
RESEARCH_STEPS = 3  # the researcher's own plan, reflect and act calls per handoff


class TeamState(BaseModel):
    question: str
    request_id: str
    hits: dict[str, ToolHit] = Field(default_factory=dict)
    tasks: list[str] = Field(default_factory=list)  # research tasks handed off so far
    task: str | None = None
    turns: int = 0
    steps: list[AgentStep] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    shadow_usd: float = 0.0
    stop_reason: str = ""


class SupervisorGraph:
    def __init__(self, settings: Settings, llm: LLMGateway) -> None:
        self._cfg, self._llm = settings, llm
        self._researcher = AgentGraph(settings.model_copy(update={"agent_max_steps": RESEARCH_STEPS}), llm)

    def worst_requests(self) -> int:
        return SUPERVISOR_TURNS + RESEARCH_HANDOFFS * RESEARCH_STEPS + 2

    async def supervise(self, state: TeamState) -> dict:
        began = time.perf_counter()
        chunks = [to_chunk(hit) for hit in state.hits.values()]
        passages, _ = build_context(chunks, self._cfg.context_token_budget, max_passages=12)
        done = "\n".join(f"- {escape(t)}" for t in state.tasks) or "- none"
        user = "\n".join(
            [
                f"<question>{escape(state.question)}</question>",
                f"<research_done>\n{done}\n</research_done>",
                render_passages(passages) if passages else "<documents></documents>",
                f"Research handoffs left: {RESEARCH_HANDOFFS - len(state.tasks)}.",
            ]
        )
        result = await self._llm.structured_call(
            role="fast",
            schema=Route,
            system=load_prompt("supervisor").text,
            user=user,
            call_name="agent_supervise",
            request_id=state.request_id,
        )
        route = result.parsed
        research = route.action == "research" and len(state.tasks) < RESEARCH_HANDOFFS and route.task.strip()
        task = route.task.strip()[:500] if research else None
        step = AgentStep(
            node="supervise",
            note=f"research: {task}" if task else "hand off to the writer",
            llm_requests=result.requests,
            latency_ms=int((time.perf_counter() - began) * 1000),
        )
        return {
            "task": task,
            "turns": state.turns + 1,
            "steps": [*state.steps, step],
            "llm_requests": state.llm_requests + result.requests,
            "input_tokens": state.input_tokens + result.input_tokens,
            "output_tokens": state.output_tokens + result.output_tokens,
            "shadow_usd": state.shadow_usd + result.shadow_usd,
            "stop_reason": "" if task else "writer",
        }

    def research_node(self, run_tool: ToolRunner):  # noqa: ANN201 - a LangGraph node
        async def research(state: TeamState) -> dict:
            found = await self._researcher.run(state.task or state.question, run_tool, state.request_id)
            hits = dict(state.hits)
            for chunk in found.chunks:
                hits.setdefault(chunk.chunk_id, _to_hit(chunk))
            return {
                "hits": hits,
                "tasks": [*state.tasks, state.task or state.question],
                "task": None,
                "sub_questions": [*state.sub_questions, *found.trace.sub_questions],
                "steps": [*state.steps, *found.trace.steps],
                "llm_requests": state.llm_requests + found.requests,
                "input_tokens": state.input_tokens + found.input_tokens,
                "output_tokens": state.output_tokens + found.output_tokens,
                "shadow_usd": state.shadow_usd + found.shadow_usd,
            }

        return research

    def build(self, run_tool: ToolRunner):  # noqa: ANN201 - a compiled LangGraph, which stays in this package
        graph = StateGraph(TeamState)
        graph.add_node("supervise", self.supervise)
        graph.add_node("research", self.research_node(run_tool))
        graph.add_edge(START, "supervise")
        graph.add_conditional_edges("supervise", _after_supervise, {"research": "research", "end": END})
        graph.add_conditional_edges("research", _after_research, {"supervise": "supervise", "end": END})
        return graph.compile()

    async def run(self, question: str, run_tool: ToolRunner, request_id: str) -> AgentFindings:
        try:
            raw = await self.build(run_tool).ainvoke(
                TeamState(question=question, request_id=request_id),
                {"recursion_limit": self._cfg.agent_recursion_limit},
            )
        except GraphRecursionError as exc:
            raise InternalError("the supervisor graph passed its recursion limit") from exc
        final = TeamState.model_validate(raw)
        trace = AgentTrace(
            sub_questions=final.sub_questions,
            steps=final.steps,
            stop_reason=final.stop_reason or "turn_limit",
            llm_requests=final.llm_requests,
            method="json" if self._llm.action_method("strong") == "json" else "tools",
        )
        return AgentFindings(
            chunks=[to_chunk(hit) for hit in final.hits.values()],
            trace=trace,
            requests=final.llm_requests,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
            shadow_usd=final.shadow_usd,
        )


def _to_hit(chunk) -> ToolHit:  # noqa: ANN001 - a Chunk from the researcher
    kind = "image" if chunk.kind == "image" else "text"
    return ToolHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_name=chunk.doc_name,
        page=chunk.page,
        kind=kind,
        text=chunk.text,
        score=chunk.score or None,
        image_path=chunk.image_path,
    )


def _after_supervise(state: TeamState) -> str:
    return "research" if state.task else "end"


def _after_research(state: TeamState) -> str:
    return "supervise" if state.turns < SUPERVISOR_TURNS else "end"
