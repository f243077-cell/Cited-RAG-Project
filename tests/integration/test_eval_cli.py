import json

import pytest
from typer.testing import CliRunner

from citerag import cli
from citerag.index import store as store_module
from citerag.index.store import IndexStore
from citerag.ingest.pipeline import Ingestor
from tests.helpers import HashEmbedding

ITEMS = [
    {"id": "a-1", "question": "warranty manufacturing defects", "type": "answerable",
     "gold_sources": [{"doc": "warranty.md"}]},
    {"id": "u-1", "question": "tungsten boiling point", "type": "unanswerable"},
]  # fmt: skip


@pytest.fixture
def run(settings, monkeypatch, tmp_path):
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
    return lambda *args: CliRunner().invoke(cli.app, ["eval", "retrieval", "--golden", str(golden), *args])


def test_report_prints_counts_and_saves_a_run(run, tmp_path):
    result = run()
    assert result.exit_code == 0, result.output
    assert "hit@5: 1/1" in result.output and "UNVERIFIED" in result.output
    summaries = list((tmp_path / "eval" / "results").glob("*/summary.json"))
    assert len(summaries) == 1
    saved = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert saved["before_rerank"]["n_scored"] == 1 and saved["config"]["chunk_size"] == 512


def test_require_verified_stops_a_draft_set(run):
    result = run("--require-verified")
    assert result.exit_code == 1 and "validation_error" in result.output


def test_empty_index_stops_with_a_clear_message(run, settings, monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "get_settings", lambda: settings.model_copy(update={"data_dir": tmp_path / "other"})
    )
    result = run()
    assert result.exit_code == 1 and "ingest documents first" in result.output
