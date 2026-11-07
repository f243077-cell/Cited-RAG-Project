"""E8 rehearsal with no network: the three modes run through the real runner on FakeGateway, then report-e8 builds the table."""

import asyncio
import json

import pytest
from typer.testing import CliRunner

from citerag import cli
from citerag.agent_graph.graph import AgentGraph
from citerag.analytics.logs import LogStore
from citerag.cache.base import NoOpCache
from citerag.eval import report as report_module
from citerag.eval.modes import (
    ModeColumn,
    context_coverage,
    estimate,
    parse_label,
    render,
    tool_argument_leaks,
)
from citerag.eval.runner import run_items
from citerag.eval.scoring import summarize
from citerag.llm.fake import FakeGateway
from citerag.llm.gateway import StructuredResult
from citerag.rag.pipeline import DeepAnswer, Deps
from citerag.schemas import (
    AgentStep,
    AgentTrace,
    GoldenItem,
    GoldenSource,
    ItemRecord,
    JudgeVerdict,
    LLMCitation,
    LLMCitedAnswer,
    Plan,
    Reflection,
    RunSummary,
)
from citerag.tools.library import ToolLibrary
from tests.integration.test_agent_mode import CHUNKS, GOVERN, TIERS, Retriever
from tests.unit.test_tools import MANIFEST

MULTI = GoldenItem(
    id="m-1",
    question="What are the CSF Tiers, and what is GOVERN?",
    type="multi_chunk",
    gold_answer="Partial to Adaptive; GOVERN is cross-cutting.",
    gold_sources=[
        GoldenSource(doc="policy.pdf", page=1, part=0),
        GoldenSource(doc="policy.pdf", page=2, part=1),
    ],
)


class StandardRetriever(Retriever):
    def retrieve(self, query, doc_ids=None):
        return CHUNKS[:1]  # standard mode finds only the first part


class StubDeep:
    async def run(self, question, passages, request_id):
        answer = LLMCitedAnswer(
            answer="Partial [1].",
            answerable=True,
            confidence="high",
            citations=[LLMCitation(marker=1, quote=TIERS)],
        )
        result = StructuredResult[LLMCitedAnswer](
            parsed=answer, model="fake/crew", requests=3, input_tokens=0, output_tokens=0, latency_ms=0
        )
        return DeepAnswer(result=result, removed_claims=0)


def both_cited():
    citations = [LLMCitation(marker=1, quote=TIERS), LLMCitation(marker=2, quote=GOVERN)]
    return LLMCitedAnswer(
        answer="Partial [1]; cross-cutting [2].", answerable=True, confidence="high", citations=citations
    )


def run_mode(settings, tmp_path, mode):
    llm = FakeGateway()
    retriever = StandardRetriever()
    deps = Deps(
        settings,
        llm,
        retriever,
        NoOpCache(),
        LogStore(settings.logs_db_path),
        lambda: "eval",
        deep=StubDeep(),
        agent=AgentGraph(settings, llm),
        tool_library=lambda scope: ToolLibrary(settings, retriever, lambda: MANIFEST, scope),
    )
    if mode == "standard":
        llm.queue("generate", LLMCitedAnswer(
            answer="Partial [1].", answerable=True, confidence="high", citations=[LLMCitation(marker=1, quote=TIERS)]
        ))  # fmt: skip
    if mode == "agent":
        llm.queue(
            "agent_plan", Plan(sub_questions=["What are the CSF Tiers?", "What is GOVERN cross-cutting?"])
        )
        llm.queue("agent_reflect", Reflection(enough=[True, True]))
        llm.queue("generate", both_cited())
    llm.queue(
        "judge",
        JudgeVerdict(faithful=True, correctness="correct" if mode == "agent" else "partial", reasoning="r"),
    )
    run_dir = tmp_path / f"run-{mode}"
    run_dir.mkdir()
    result = asyncio.run(run_items([MULTI], deps, run_dir, mode=mode))
    return run_dir, result


def test_the_agent_covers_both_parts_where_standard_mode_found_one(settings, tmp_path):
    runs = {mode: run_mode(settings, tmp_path, mode) for mode in ("standard", "deep", "agent")}
    coverage = {mode: context_coverage(result.records) for mode, (_, result) in runs.items()}
    assert coverage["standard"].n == 0 and coverage["deep"].n == 0 and coverage["agent"].n == 1
    agent = runs["agent"][1].records[0]
    assert agent.trace is not None and agent.usage.llm_requests == 3
    assert [r.records[0].usage.llm_requests for r in (runs["standard"][1], runs["deep"][1])] == [1, 3]


def test_report_e8_renders_three_columns_from_saved_runs(settings, tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir()
    for mode in ("standard", "deep", "agent"):
        run_dir, result = run_mode(settings, tmp_path, mode)
        summary = RunSummary(
            run_id=f"r-{mode}",
            status="complete",
            experiment=f"e8_{mode}",
            golden="g",
            n_items=1,
            n_done=1,
            unverified_items=1,
            answers=summarize(result.records),
        )
        target = root / f"r-{mode}"
        run_dir.rename(target)
        (target / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(report_module, "RESULTS_DIR", root)
    for fn in (report_module.load_detail, report_module.load_summary, report_module.run_path):
        monkeypatch.setattr(fn, "__defaults__", (root,))
    out = tmp_path / "docs" / "E8_modes.md"
    result = CliRunner().invoke(
        cli.app, ["eval", "report-e8", "r-standard", "r-deep", "r-agent", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "| Metric | e8_standard | e8_deep | e8_agent |" in text and "(provisional)" in text
    assert "| multi_chunk context coverage | 0 of 1 (0%) | 0 of 1 (0%) | 1 of 1 (100%) |" in text
    assert "| Mean LLM requests per item | 1.0 | 3.0 | 3.0 |" in text
    assert (out.parent / "E8_modes.png").exists()


def test_a_canary_in_a_tool_argument_is_counted():
    trace = AgentTrace(
        steps=[AgentStep(node="tools", tool="search_documents", args={"query": "canary-5520 passwords"})]
    )
    record = ItemRecord(
        id="i", type="injection", question="q", status="answered", canary="CANARY-5520", trace=trace
    )
    clean = record.model_copy(update={"trace": AgentTrace(steps=[AgentStep(node="plan")])})
    assert tool_argument_leaks([record, clean]) == 1


def test_labels_parse_back_to_document_and_page():
    assert parse_label("NIST.CSWP.29.pdf p8") == ("NIST.CSWP.29.pdf", 8)
    assert parse_label("uv-readme.md") == ("uv-readme.md", None)
    assert parse_label("my p notes.md") == ("my p notes.md", None)


@pytest.mark.parametrize(
    ("mode", "expected", "worst"),
    [("standard", 3, 7), ("deep", 7, 11), ("agent", 7, 17)],
)
def test_the_estimate_counts_answers_and_judging(mode, expected, worst):
    items = [MULTI, GoldenItem(id="u", question="q", type="unanswerable")]
    assert estimate(items, mode, skip_judge=False) == (expected, worst)


def test_eval_run_estimate_spends_nothing_and_skips_images_in_agent_mode(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    golden = tmp_path / "g.jsonl"
    items = [MULTI, GoldenItem(id="v", question="q", type="image", image_path="x.png")]
    golden.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")
    config = tmp_path / "e8_agent.yaml"
    config.write_text("name: e8_agent\nmode: agent\n", encoding="utf-8")
    args = ["eval", "run", "--golden", str(golden), "--config", str(config), "--estimate"]
    result = CliRunner().invoke(cli.app, args)
    assert (
        result.exit_code == 0 and "1 items in agent mode: about 4 LLM requests, 9 at worst" in result.output
    )


def test_the_report_says_when_nothing_is_verified():
    summary = RunSummary(run_id="r", status="complete", golden="g", n_items=1, unverified_items=1)
    text = render([ModeColumn("e8_agent", summary, [])])
    assert "no number here is a result" in text and "not measured" in text


def test_the_e8_configs_load():
    from pathlib import Path

    from citerag.eval.experiments import load_experiment

    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    assert [load_experiment(root / f"e8_{m}.yaml").mode for m in ("standard", "deep", "agent")] == [
        "standard",
        "deep",
        "agent",
    ]


def test_the_hijack_fixture_carries_its_canary():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "eval/fixtures/poisoned/agent_research_memo.md").read_text(
        "utf-8"
    )
    assert "CANARY-5520" in text and "search_documents" in text


def test_records_keep_the_trace_through_json():
    trace = AgentTrace(steps=[AgentStep(node="plan")], llm_requests=1)
    record = ItemRecord(id="a", type="answerable", question="q", status="answered", trace=trace)
    assert ItemRecord.model_validate(json.loads(record.model_dump_json())).trace == trace
