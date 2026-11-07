"""The owner's tasks, rehearsed with no LLM request: accepting golden items, and a capped run resumed later."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from citerag import cli
from citerag.eval import execute as execute_module
from citerag.eval import report as report_module
from citerag.eval.golden import load_golden
from citerag.index import store as store_module
from citerag.index.store import IndexStore
from citerag.ingest.pipeline import Ingestor
from citerag.llm.fake import FakeGateway
from citerag.schemas import GoldenItem, GoldenSource, JudgeVerdict, LLMCitation, LLMCitedAnswer
from tests.helpers import HashEmbedding

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("accept_golden", ROOT / "scripts" / "accept_golden.py")
accept_golden = importlib.util.module_from_spec(spec)
sys.modules["accept_golden"] = accept_golden
spec.loader.exec_module(accept_golden)


def item(item_id, answer="A."):
    return GoldenItem(id=item_id, question=f"q {item_id}", type="answerable", gold_answer=answer)


def test_only_the_named_ids_become_verified_and_unknown_ids_are_refused(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "".join(item(i).model_dump_json() + "\n" for i in ("a-1", "a-2", "a-3")), encoding="utf-8"
    )
    golden = tmp_path / "golden.jsonl"
    runner = CliRunner()
    app = typer.Typer()
    app.command()(accept_golden.main)
    args = ["--ids", "a-1,a-3", "--candidates", str(candidates), "--golden", str(golden)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert [(i.id, i.verified) for i in load_golden(golden, require_verified=True)] == [
        ("a-1", True),
        ("a-3", True),
    ]
    assert not any(i.verified for i in load_golden(candidates))
    bad = runner.invoke(app, ["--ids", "zz-9", "--candidates", str(candidates), "--golden", str(golden)])
    assert bad.exit_code != 0 and "not in the candidates: zz-9" in bad.output


def test_accepting_again_replaces_the_corrected_item():
    before = [item("a-1", "old").model_copy(update={"verified": True})]
    after = accept_golden.accept([item("a-1", "new")], before, ["a-1"])
    assert [(i.id, i.gold_answer, i.verified) for i in after] == [("a-1", "new", True)]


@pytest.fixture
def capped(settings, monkeypatch, tmp_path):
    """A real `citerag eval run` on a tiny index, with FakeGateway in place of OpenRouter."""
    store = IndexStore(settings.index_dir, HashEmbedding())
    Ingestor(settings, store, settings.index_dir).ingest_bytes(
        "policy.txt", b"Items can be returned within thirty days."
    )
    store.close()
    items = [
        GoldenItem(
            id=f"a-{n}",
            question="how many days to return items",
            type="answerable",
            gold_answer="Thirty days.",
            gold_sources=[GoldenSource(doc="policy.txt")],
            verified=True,
        )
        for n in range(1, 4)
    ]
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")
    llm = FakeGateway()
    for _ in items:
        quote = LLMCitation(marker=1, quote="returned within thirty days")
        llm.queue(
            "generate",
            LLMCitedAnswer(answer="Thirty [1].", answerable=True, confidence="high", citations=[quote]),
        )
        llm.queue("judge", JudgeVerdict(faithful=True, correctness="correct", reasoning="r"))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "make_embedder", lambda _: HashEmbedding())
    monkeypatch.setattr(execute_module, "OpenRouterGateway", lambda cfg, log: llm)
    monkeypatch.setattr(execute_module, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    return golden, llm


def run(*args):
    return CliRunner().invoke(cli.app, ["eval", "run", *args])


def test_a_capped_run_pauses_and_resume_finishes_it_without_repeating_work(capped):
    golden, llm = capped
    first = run("--golden", str(golden), "--require-verified", "--max-requests", "5")
    assert first.exit_code == 0, first.output
    run_id = first.output.split("Continue with --resume ")[1].split()[0]
    assert "paused" in first.output and "--max-requests 5 reached" in first.output
    done_first = len(llm.calls)
    second = run("--golden", str(golden), "--require-verified", "--resume", run_id, "--max-requests", "35")
    assert second.exit_code == 0, second.output
    assert f"Run {run_id}: complete" in second.output and "Items done: 3 of 3" in second.output
    assert len(llm.calls) == 6 and done_first < 6  # 3 answers and 3 verdicts in all: nothing paid twice
    lines = (Path("results") / run_id / "items.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["id"] for line in lines) == ["a-1", "a-2", "a-3"]


def test_require_verified_refuses_an_unverified_file(capped, tmp_path):
    golden, _ = capped
    loose = tmp_path / "loose.jsonl"
    loose.write_text(item("a-9").model_dump_json() + "\n", encoding="utf-8")
    result = run("--golden", str(loose), "--require-verified", "--offline")
    assert result.exit_code == 1 and "not verified by a human: ['a-9']" in result.output
    assert report_module is not None
