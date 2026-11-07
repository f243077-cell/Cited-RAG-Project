"""Which agent answers in Agent mode: the single agent (default) or the M12 supervisor team (AGENT_ARCHITECTURE)."""

from __future__ import annotations

from citerag.agent_graph.graph import AgentGraph
from citerag.agent_graph.supervisor import SupervisorGraph
from citerag.config import Settings
from citerag.llm.gateway import LLMGateway


def build_agent(settings: Settings, llm: LLMGateway) -> AgentGraph | SupervisorGraph:
    if settings.agent_architecture == "supervisor":
        return SupervisorGraph(settings, llm)
    return AgentGraph(settings, llm)
