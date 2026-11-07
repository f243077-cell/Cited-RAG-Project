"""M12 supervisor and workers on FakeGateway: handoffs, budgets and the same pipeline contract as the single agent."""

import asyncio

import pytest

from citerag.agent_graph.factory import build_agent
from citerag.agent_graph.graph import AgentGraph
from citerag.agent_graph.supervisor import SupervisorGraph
from citerag.analytics.logs import LogStore
from citerag.cache.base import NoOpCache
from citerag.errors import BudgetExceededError
from citerag.llm.fake import FakeGateway
from citerag.rag.pipeline import Deps, answer_question
from citerag.schemas import AgentAction, ChatRequest, Plan, Reflection, Route
from citerag.tools.library import ToolLibrary
from tests.integration.test_agent_mode import Retriever, answer, search
from tests.unit.test_tools import MANIFEST


@pytest.fixture
def team(settings):
    cfg = settings.model_copy(update={"agent_architecture": "supervisor"})
    llm, retriever = FakeGateway(), Retriever()

    def make():
        return Deps(
            cfg,
            llm,
            retriever,
            NoOpCache(),
            LogStore(cfg.logs_db_path),
            lambda: "ns",
            agent=build_agent(cfg, llm),
            tool_library=lambda scope: ToolLibrary(cfg, retriever, lambda: MANIFEST, scope),
        )

    return llm, make, retriever


def ask(deps):
    request = ChatRequest(session_id="s", message="What are the CSF Tiers?", mode="agent")
    return asyncio.run(answer_question(request, deps))


def names(llm):
    return [c.call_name for c in llm.calls]


def test_the_factory_picks_the_architecture(settings):
    assert isinstance(build_agent(settings, FakeGateway()), AgentGraph)
    team = settings.model_copy(update={"agent_architecture": "supervisor"})
    assert isinstance(build_agent(team, FakeGateway()), SupervisorGraph)


def test_one_handoff_then_the_writer_costs_five_requests(team):
    llm, make, retriever = team
    llm.queue("agent_supervise", Route(action="research", task="Find the CSF Tiers"))
    llm.queue("agent_plan", Plan(sub_questions=["What are the CSF Tiers?"]))
    llm.queue("agent_reflect", Reflection(enough=[True]))
    llm.queue("agent_supervise", Route(action="answer"))
    llm.queue("generate", answer())
    response = ask(make())
    assert names(llm) == ["agent_supervise", "agent_plan", "agent_reflect", "agent_supervise", "generate"]
    assert response.usage.llm_requests == response.trace.llm_requests == 5
    nodes = [s.node for s in response.trace.steps]
    assert nodes == ["supervise", "plan", "tools", "reflect", "supervise", "synthesize"]
    assert response.trace.steps[0].note == "research: Find the CSF Tiers"
    assert response.answer.citations[0].valid and response.trace.stop_reason == "writer"


def test_the_supervisor_sees_what_research_found_and_did(team):
    llm, make, _ = team
    llm.queue("agent_supervise", Route(action="research", task="Find the CSF Tiers"))
    llm.queue("agent_plan", Plan(sub_questions=["What are the CSF Tiers?"]))
    llm.queue("agent_reflect", Reflection(enough=[True]))
    llm.queue("agent_supervise", Route(action="answer"))
    llm.queue("generate", answer())
    ask(make())
    second = [c for c in llm.calls if c.call_name == "agent_supervise"][1]
    assert "- Find the CSF Tiers" in second.user and "Partial, Risk Informed" in second.user
    assert "Research handoffs left: 1." in second.user


def test_handoffs_and_turns_are_capped_at_eleven_requests(team):
    llm, make, _ = team
    for _ in range(3):
        llm.queue("agent_supervise", Route(action="research", task="more"))
    for _ in range(2):
        llm.queue("agent_plan", Plan(sub_questions=["CSF Tiers"]))
        llm.queue("agent_reflect", Reflection(enough=[False], next_queries=["CSF Tiers again"]))
        llm.queue("agent_act", AgentAction(calls=[search("CSF Tiers")]))
    llm.queue("generate", answer(quote="a quote that is in no passage at all, none"))
    llm.queue("generate", answer())
    response = ask(make())
    assert response.usage.llm_requests == 11 == make().agent.worst_requests()
    assert names(llm).count("agent_supervise") == 3 and names(llm).count("agent_plan") == 2


def test_the_writer_is_chosen_when_no_handoff_is_left(team):
    llm, make, _ = team
    llm.queue("agent_supervise", Route(action="research", task="Find the CSF Tiers"))
    llm.queue("agent_plan", Plan(sub_questions=["What are the CSF Tiers?"]))
    llm.queue("agent_reflect", Reflection(enough=[True]))
    llm.queue("agent_supervise", Route(action="research", task="again"))
    llm.queue("agent_plan", Plan(sub_questions=["What are the CSF Tiers?"]))
    llm.queue("agent_reflect", Reflection(enough=[True]))
    llm.queue(
        "agent_supervise", Route(action="research", task="a third time")
    )  # no handoff left: the writer goes
    llm.queue("generate", answer())
    response = ask(make())
    assert names(llm).count("agent_plan") == 2 and response.trace.steps[-2].note == "hand off to the writer"


def test_the_guard_needs_room_for_eleven(team):
    llm, make, _ = team
    llm.queue("room", BudgetExceededError("Daily LLM request limit reached; this needs 11"))
    with pytest.raises(BudgetExceededError):
        ask(make())
    assert llm.room_checks == [("strong", 11)] and llm.calls == []


def test_answering_at_once_with_no_evidence_is_a_refusal(team):
    llm, make, _ = team
    llm.queue("agent_supervise", Route(action="answer"))
    response = ask(make())
    assert not response.answer.answerable and names(llm) == ["agent_supervise"]


def test_the_m12_configs_load_with_the_supervisor_override():
    from pathlib import Path

    from citerag.eval.experiments import load_experiment

    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    team = load_experiment(root / "m12_supervisor.yaml")
    assert team.mode == "agent" and team.overrides == {"agent_architecture": "supervisor"}
    assert load_experiment(root / "m12_single.yaml").overrides == {}


def test_the_estimate_knows_the_supervisor_costs_more():
    from citerag.eval.modes import estimate
    from citerag.schemas import GoldenItem

    items = [GoldenItem(id="a", question="q", type="answerable")]
    assert estimate(items, "agent", skip_judge=False) == (4, 9)
    assert estimate(items, "agent", skip_judge=False, supervisor=True) == (6, 12)
