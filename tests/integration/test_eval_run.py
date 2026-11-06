"""`citerag eval run`, compare and gate through the CLI, plus the runs API and the Evaluation page."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from citerag import cli
from citerag.analytics.plots import grouped_bars, type_heatmap, type_scores
from citerag.api.main import create_app
from citerag.eval.report import compare_rows, list_runs, load_summary
from citerag.index import store as store_module
from citerag.index.store import IndexStore
from citerag.ingest.pipeline import Ingestor
from citerag.schemas import Citation, CitedAnswer, ItemRecord, RetrievalScores, RunDetail, RunSummary
from citerag.ui import common
from tests.helpers import HashEmbedding

ITEMS = [
    {"id": "a-1", "question": "warranty manufacturing defects", "type": "answerable",
     "gold_sources": [{"doc": "warranty.md"}]},
    {"id": "u-1", "question": "tungsten boiling point", "type": "unanswerable"},
    {"id": "v-1", "question": "what does the chart show", "type": "image", "gold_sources": [{"doc": "warranty.md"}]},
]  # fmt: skip


@pytest.fixture
def cli_run(settings, monkeypatch, tmp_path):
    store = IndexStore(settings.index_dir, HashEmbedding())
    Ingestor(settings, store, settings.index_dir).ingest_bytes(
        "warranty.md", b"Warranty covers manufacturing defects."
    )
    store.close()
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(i) for i in ITEMS), encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "make_embedder", lambda _: HashEmbedding())
    monkeypatch.chdir(tmp_path)
    return lambda *args: CliRunner().invoke(cli.app, ["eval", "run", "--golden", str(golden), *args])


def latest(tmp_path):
    return sorted(p.name for p in (tmp_path / "eval" / "results").iterdir())[-1]


def test_offline_run_scores_retrieval_with_no_llm_and_skips_image_items(cli_run, tmp_path):
    result = cli_run("--offline")
    assert result.exit_code == 0, result.output
    assert "UNVERIFIED" in result.output and "hit@5: 100.0%" in result.output
    summary = load_summary(latest(tmp_path))
    assert summary.status == "offline" and summary.n_items == 2 and summary.requests_used == 0
    assert summary.answers is None
    run_dir = tmp_path / "eval" / "results" / latest(tmp_path)
    assert {"config.json", "summary.json"} <= {p.name for p in run_dir.iterdir()}


def test_require_verified_stops_a_draft_set(cli_run):
    result = cli_run("--offline", "--require-verified")
    assert result.exit_code == 1 and "validation_error" in result.output


def test_experiment_vector_mode_and_rejected_config(cli_run, tmp_path):
    config = tmp_path / "e.yaml"
    config.write_text("name: e2_vector\nretrieval: vector\n", encoding="utf-8")
    assert cli_run("--offline", "--config", str(config)).exit_code == 0
    assert load_summary(latest(tmp_path)).experiment == "e2_vector"
    config.write_text("name: bad\noverrides:\n  gemini_api_key: x\n", encoding="utf-8")
    assert cli_run("--offline", "--config", str(config)).exit_code == 1


def test_gate_prints_a_proposal_and_writes_nothing(cli_run, tmp_path):
    cli_run("--offline")
    before = {p: p.read_bytes() for p in (tmp_path / "eval" / "results").rglob("*") if p.is_file()}
    result = CliRunner().invoke(cli.app, ["eval", "gate", latest(tmp_path)])
    assert result.exit_code == 0 and "Set MIN_TOP_SIMILARITY yourself" in result.output
    assert before == {p: p.read_bytes() for p in before}


def test_compare_prints_deltas_and_marks_noise(cli_run, tmp_path):
    cli_run("--offline")
    first = latest(tmp_path)
    path = tmp_path / "eval" / "results" / "other"
    path.mkdir()
    summary = load_summary(first).model_copy(update={"run_id": "other"})
    summary.before_rerank.hit_rate[1] = 0.9
    summary.before_rerank.hit_rate[3] = 0.99
    (path / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["eval", "compare", first, "other"])
    assert result.exit_code == 0, result.output
    assert "-10.0" in result.output and "(noise)" in result.output


def test_compare_unknown_run_fails_cleanly():
    result = CliRunner().invoke(cli.app, ["eval", "compare", "nope", "nada"])
    assert result.exit_code == 1 and "not_found" in result.output


def summary(run_id="r1", **kw):
    return RunSummary(run_id=run_id, status="offline", golden="g", n_items=1, unverified_items=1, **kw)


def saved_run(root):
    path = root / "eval" / "results" / "r1"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(summary().model_dump_json(), encoding="utf-8")
    citation = Citation(
        marker=1, chunk_id="c", doc_id="d", doc_name="d.pdf", page=1, quote="q", valid=True, match_score=99
    )
    answer = CitedAnswer(answer="yes [1]", answerable=True, confidence="high", citations=[citation])
    record = ItemRecord(id="a-1", type="answerable", question="q?", status="answered", answer=answer)
    (path / "items.jsonl").write_text(record.model_dump_json() + "\n", encoding="utf-8")


def test_api_lists_runs_and_blocks_path_tricks(settings, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved_run(tmp_path)
    (tmp_path / "eval" / "results" / "old-style").mkdir()
    (tmp_path / "eval" / "results" / "old-style" / "summary.json").write_text(
        '{"golden": "x"}', encoding="utf-8"
    )
    with TestClient(create_app(settings)) as client:
        runs = client.get("/v1/eval/runs").json()
        detail = RunDetail.model_validate(client.get("/v1/eval/runs/r1").json())
        assert client.get("/v1/eval/runs/nope").status_code == 404
        assert client.get("/v1/eval/runs/..%2F..%2Fetc").status_code == 404
    assert [r["run_id"] for r in runs] == ["r1"]
    assert detail.items[0].id == "a-1"


def test_compare_rows_mark_metrics_one_run_lacks(tmp_path):
    rows = compare_rows(summary(), summary())
    assert rows == []
    assert list_runs(tmp_path) == []


def test_charts_return_figures_with_an_axis():
    frame = pd.DataFrame({"A": [0.9, None], "B": [0.8, 0.7]}, index=["hit@5", "faithfulness"])
    assert isinstance(grouped_bars(frame, "t"), Figure) and grouped_bars(frame, "t").axes
    item = ItemRecord(id="a", type="answerable", question="q", status="error")
    scores = type_scores([item])
    assert isinstance(type_heatmap(scores), Figure) and type_heatmap(scores).axes


class EvalClient:
    def __init__(self, root):
        self.root = root

    def eval_runs(self):
        return list_runs(self.root)

    def eval_run(self, run_id):
        from citerag.eval.report import load_detail

        return load_detail(run_id, self.root)


def test_evaluation_page_shows_a_run_and_warns_when_unverified(monkeypatch, tmp_path):
    saved_run(tmp_path)
    monkeypatch.setattr(common, "get_client", lambda: EvalClient(tmp_path / "eval" / "results"))
    page = common.__file__.replace("common.py", "pages/4_Evaluation.py")
    app = AppTest.from_file(page, default_timeout=30).run()
    assert not app.exception
    assert "not verified" in app.warning[0].value
    assert any("yes [1]" in m.value for m in app.markdown)


def test_a_run_with_no_scored_items_reports_no_retrieval_score():
    scores = RetrievalScores(n_scored=0, ks=[5], hits={5: 0}, hit_rate={5: 0.0}, mrr_k=10, mrr=0.0)
    assert summary(before_rerank=scores).headline() == {}


def test_evaluation_page_compares_two_runs(monkeypatch, tmp_path):
    saved_run(tmp_path)
    second = tmp_path / "eval" / "results" / "r2"
    second.mkdir()
    scores = RetrievalScores(n_scored=2, ks=[5], hits={5: 2}, hit_rate={5: 1.0}, mrr_k=10, mrr=1.0)
    (second / "summary.json").write_text(
        summary("r2", before_rerank=scores).model_dump_json(), encoding="utf-8"
    )
    monkeypatch.setattr(common, "get_client", lambda: EvalClient(tmp_path / "eval" / "results"))
    page = common.__file__.replace("common.py", "pages/4_Evaluation.py")
    app = AppTest.from_file(page, default_timeout=30).run()
    assert app.selectbox[0].value == "r2"
    app.selectbox[1].select("r1").run()
    assert not app.exception
    assert any("under 3 points" in c.value for c in app.caption)


def test_parts_command_prints_each_part_and_what_was_missed(settings, monkeypatch, tmp_path):
    store = IndexStore(settings.index_dir, HashEmbedding())
    Ingestor(settings, store, settings.index_dir).ingest_bytes("warranty.md", b"Warranty covers defects.")
    store.close()
    item = {
        "id": "m-1",
        "question": "warranty defects",
        "type": "multi_chunk",
        "gold_sources": [{"doc": "warranty.md", "part": 0}, {"doc": "gone.md", "part": 1}],
    }
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps(item), encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "make_embedder", lambda _: HashEmbedding())
    result = CliRunner().invoke(cli.app, ["eval", "parts", "--golden", str(golden)])
    assert result.exit_code == 0, result.output
    assert "part 0 (ok, rank 1)" in result.output and "part 1 (MISSED, not in the top 20)" in result.output
    assert "missed at k=5: [1]" in result.output and "full_coverage@5: 0/1" in result.output
