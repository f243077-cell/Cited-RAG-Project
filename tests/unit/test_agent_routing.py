"""Graph routing as pure functions: every edge, without LangGraph or a model."""

from citerag.agent_graph import routing
from citerag.agent_graph.state import AgentState
from citerag.schemas import ToolCall


def state(**kw):
    return AgentState(question="q", request_id="r", **kw)


CALL = ToolCall(name="search_documents", args={"query": "q"})


def test_the_plan_runs_its_own_searches_next():
    assert routing.after_plan(state(pending=[CALL])) == "tools"
    assert routing.after_plan(state()) == "end"


def test_act_runs_tools_only_when_it_chose_some():
    assert routing.after_act(state(pending=[CALL])) == "tools"
    assert routing.after_act(state()) == "end"


def test_tools_go_to_reflect_unless_the_steps_are_used_up():
    assert routing.after_tools(state(steps_used=2), 6) == "reflect"
    assert routing.after_tools(state(steps_used=6), 6) == "end"


def test_reflect_loops_back_until_enough_or_out_of_steps():
    assert routing.after_reflect(state(steps_used=3), 6) == "act"
    assert routing.after_reflect(state(steps_used=3, done=True), 6) == "end"
    assert routing.after_reflect(state(steps_used=6), 6) == "end"


def test_the_worst_legal_path_uses_six_steps_and_ten_nodes():
    """plan, tools, reflect, act, tools, reflect, act, tools, reflect, end: 6 LLM steps."""
    s, nodes, node = state(), [], "plan"
    while node != "end":
        nodes.append(node)
        if node in ("plan", "act", "reflect"):
            s.steps_used += 1
        if node in ("plan", "act"):
            s.pending = [CALL]
        if node == "tools":
            s.pending = []
        node = {
            "plan": lambda: routing.after_plan(s),
            "act": lambda: routing.after_act(s),
            "tools": lambda: routing.after_tools(s, 6),
            "reflect": lambda: routing.after_reflect(s, 6),
        }[node]()
    assert s.steps_used == 6 and len(nodes) + 1 == 10


def test_stop_reason_prefers_what_a_node_said():
    assert routing.stop_reason(state(stop_reason="enough"), 6) == "enough"
    assert routing.stop_reason(state(steps_used=6), 6) == "step_limit"
    assert routing.stop_reason(state(steps_used=2), 6) == "no_more_calls"
