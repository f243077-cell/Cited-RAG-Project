"""Agent mode: plan, then a ReAct loop of act, tools and reflect (PHASE3_ARCHITECTURE section 3).

The graph gathers evidence only. Synthesis stays in rag/pipeline.py, on the same path as standard mode: build_context,
generate, verify_citations and one corrective retry. Every model call goes through the gateway, so the limiter, the daily
guard and llm_calls see it. Tool calls go through a runner the pipeline provides, which holds the index lock.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from xml.sax.saxutils import escape

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from citerag.agent_graph import routing
from citerag.agent_graph.state import AgentState
from citerag.config import Settings
from citerag.errors import InternalError
from citerag.llm.gateway import LLMGateway, StructuredResult
from citerag.prompts import load_prompt
from citerag.rag.context import Chunk, build_context, render_passages
from citerag.rag.pipeline import AgentFindings
from citerag.schemas import AgentStep, AgentTrace, Plan, Reflection, ToolCall
from citerag.tools.contracts import ACT_TOOLS, ARGS, MAX_QUERY_CHARS, ToolHit, ToolResult, tool_specs

ToolRunner = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]
_GOAL_STEPS = {"plan", "act", "reflect"}
SEED_K = 4  # passages per sub-question in the first, code-driven search


def to_chunk(hit: ToolHit) -> Chunk:
    return Chunk(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        doc_name=hit.doc_name,
        page=hit.page,
        text=hit.text,
        score=hit.score or 0.0,
        kind=hit.kind,
        image_path=hit.image_path,
    )


def _label(hit: ToolHit) -> str:
    return hit.doc_name if hit.page is None else f"{hit.doc_name} p{hit.page}"


class AgentGraph:
    def __init__(self, settings: Settings, llm: LLMGateway) -> None:
        self._cfg, self._llm = settings, llm

    # -- what the model sees ---------------------------------------------------------------------
    def _evidence(self, state: AgentState) -> str:
        chunks = [to_chunk(hit) for hit in state.hits.values()]
        passages, _ = build_context(chunks, self._cfg.context_token_budget, max_passages=12)
        return render_passages(passages) if passages else "<documents></documents>"

    def _brief(self, state: AgentState) -> str:
        """Question and plan as escaped text: both are model or user words, never markup."""
        lines = [f"<question>{escape(state.question)}</question>", "<sub_questions>"]
        lines += [f"{n}. {escape(q)}" for n, q in enumerate(state.sub_questions, start=1)]
        lines.append("</sub_questions>")
        return "\n".join(lines)

    def _count(
        self, state: AgentState, result: StructuredResult, node: str, began: float, **step: Any
    ) -> dict:
        record = AgentStep(
            node=node,
            llm_requests=result.requests,
            latency_ms=int((time.perf_counter() - began) * 1000),
            **step,
        )
        return {
            "steps_used": state.steps_used + (node in _GOAL_STEPS),
            "steps": [*state.steps, record],
            "llm_requests": state.llm_requests + result.requests,
            "input_tokens": state.input_tokens + result.input_tokens,
            "output_tokens": state.output_tokens + result.output_tokens,
            "shadow_usd": state.shadow_usd + result.shadow_usd,
        }

    # -- nodes -----------------------------------------------------------------------------------
    async def plan(self, state: AgentState) -> dict:
        began = time.perf_counter()
        result = await self._llm.structured_call(
            role="fast",
            schema=Plan,
            system=load_prompt("agent_plan").text,
            user=f"<question>{escape(state.question)}</question>",
            call_name="agent_plan",
            request_id=state.request_id,
        )
        cleaned = [q.strip() for q in result.parsed.sub_questions if q.strip()]
        subs = cleaned[: self._cfg.agent_max_subquestions] or [state.question]
        # planner-executor: code searches each sub-question first, so the model's first tool choice is never wasted
        seeds = [
            ToolCall(name="search_documents", args={"query": q[:MAX_QUERY_CHARS], "k": SEED_K}) for q in subs
        ]
        note = f"{len(subs)} sub-question(s), each searched by code"
        return {
            "sub_questions": subs,
            "pending": seeds,
            **self._count(state, result, "plan", began, note=note),
        }

    async def act(self, state: AgentState) -> dict:
        began = time.perf_counter()
        hints = "\n".join(f"- {escape(q)}" for q in state.next_queries) or "- none"
        done = "\n".join(f"- {escape(q)}" for q in state.searched) or "- none"
        library = "\n".join(f"- {escape(name)}" for name in state.documents) or "- (unknown)"
        user = "\n".join(
            [
                self._brief(state),
                f"<library>\n{library}\n</library>",
                f"<suggested_searches>\n{hints}\n</suggested_searches>",
                f"<searches_already_run>\n{done}\n</searches_already_run>",
                self._evidence(state),
            ]
        )
        result = await self._llm.tool_step(
            role="strong",
            system=load_prompt("agent_act").text,
            user=user,
            tools=tool_specs(ACT_TOOLS),
            call_name="agent_act",
            request_id=state.request_id,
        )
        calls = result.parsed.calls[: self._cfg.agent_max_subquestions]
        note = f"{len(calls)} tool call(s)" if calls else "no more tool calls"
        update = {"pending": calls, **self._count(state, result, "act", began, note=note)}
        if not calls:
            update["stop_reason"] = "enough" if state.hits else "no_more_calls"
        return update

    def tools_node(self, run_tool: ToolRunner) -> Callable[[AgentState], Awaitable[dict]]:
        async def tools(state: AgentState) -> dict:
            hits, steps, searched = dict(state.hits), list(state.steps), list(state.searched)
            for call in state.pending:
                began = time.perf_counter()
                result = await run_tool(call.name, call.args)
                fresh = [h for h in result.hits if h.chunk_id not in hits]
                hits.update({h.chunk_id: h for h in fresh})
                valid = ARGS[call.name].model_validate(call.args).model_dump() if result.ok else None
                if call.name == "search_documents" and valid:
                    searched.append(valid["query"])
                note = result.error or result.message or f"{len(result.hits)} hit(s), {len(fresh)} new"
                steps.append(
                    AgentStep(
                        node="tools",
                        tool=call.name,
                        args=valid,
                        hits=[_label(h) for h in result.hits],
                        note=note if result.ok else f"error: {note}",
                        latency_ms=int((time.perf_counter() - began) * 1000),
                    )
                )
            return {"hits": hits, "steps": steps, "searched": searched, "pending": []}

        return tools

    async def reflect(self, state: AgentState) -> dict:
        began = time.perf_counter()
        result = await self._llm.structured_call(
            role="fast",
            schema=Reflection,
            system=load_prompt("agent_reflect").text,
            user=f"{self._brief(state)}\n{self._evidence(state)}",
            call_name="agent_reflect",
            request_id=state.request_id,
        )
        verdict = result.parsed
        answered = sum(verdict.enough[: len(state.sub_questions)])
        enough = answered >= len(state.sub_questions) or not verdict.next_queries
        note = f"{answered} of {len(state.sub_questions)} sub-question(s) answered"
        update = {
            "next_queries": verdict.next_queries[:3],
            **self._count(state, result, "reflect", began, note=note),
        }
        if enough:
            update |= {"done": True, "stop_reason": "enough" if answered else "no_evidence"}
        return update

    # -- the graph -------------------------------------------------------------------------------
    def build(self, run_tool: ToolRunner):  # noqa: ANN201 - a compiled LangGraph, which stays in this package
        limit = self._cfg.agent_max_steps
        graph = StateGraph(AgentState)
        graph.add_node("plan", self.plan)
        graph.add_node("act", self.act)
        graph.add_node("tools", self.tools_node(run_tool))
        graph.add_node("reflect", self.reflect)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", routing.after_plan, {"tools": "tools", "end": END})
        graph.add_conditional_edges("act", routing.after_act, {"tools": "tools", "end": END})
        graph.add_conditional_edges(
            "tools", lambda s: routing.after_tools(s, limit), {"reflect": "reflect", "end": END}
        )
        graph.add_conditional_edges(
            "reflect", lambda s: routing.after_reflect(s, limit), {"act": "act", "end": END}
        )
        return graph.compile()

    async def run(self, question: str, run_tool: ToolRunner, request_id: str) -> AgentFindings:
        app = self.build(run_tool)
        listed = await run_tool("list_documents", {})  # names for the act prompt; code, not a model call
        names = [doc.doc_name for doc in listed.documents]
        start = AgentState(question=question, request_id=request_id, documents=names)
        try:
            raw = await app.ainvoke(start, {"recursion_limit": self._cfg.agent_recursion_limit})
        except GraphRecursionError as exc:
            raise InternalError("the agent graph passed its recursion limit") from exc
        final = AgentState.model_validate(raw)
        trace = AgentTrace(
            sub_questions=final.sub_questions,
            steps=final.steps,
            stop_reason=routing.stop_reason(final, self._cfg.agent_max_steps),
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
