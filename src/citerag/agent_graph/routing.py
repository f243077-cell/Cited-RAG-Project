"""Where the graph goes next: pure functions of the state, tested without LangGraph (ADR-19).

The step count is the real limit and is counted in LLM calls (plan, act, reflect), because that is what the quota counts.
LangGraph's recursion_limit only catches a bug here.
"""

from __future__ import annotations

from typing import Literal

from citerag.agent_graph.state import AgentState

End = Literal["end"]


def steps_left(state: AgentState, max_steps: int) -> bool:
    return state.steps_used < max_steps


def after_plan(state: AgentState) -> Literal["tools"] | End:
    """The plan's own searches run next; they cost no LLM call."""
    return "tools" if state.pending else "end"


def after_act(state: AgentState) -> Literal["tools"] | End:
    return "tools" if state.pending else "end"


def after_tools(state: AgentState, max_steps: int) -> Literal["reflect"] | End:
    return "reflect" if steps_left(state, max_steps) else "end"


def after_reflect(state: AgentState, max_steps: int) -> Literal["act"] | End:
    if state.done or not steps_left(state, max_steps):
        return "end"
    return "act"


def stop_reason(state: AgentState, max_steps: int) -> str:
    """Why the loop ended, for the trace. Set once, when the graph finishes."""
    if state.stop_reason:
        return state.stop_reason
    if not steps_left(state, max_steps):
        return "step_limit"
    return "no_more_calls"
