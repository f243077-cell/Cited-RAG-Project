import pytest
from typer.testing import CliRunner

from citerag import cli
from citerag.index import store as store_module
from tests.helpers import HashEmbedding


@pytest.fixture
def run(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(store_module, "make_embedder", lambda _: HashEmbedding())
    return lambda *args: CliRunner().invoke(cli.app, ["ingest", *map(str, args)])


def test_ingest_then_noop_then_replace(run, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    doc = folder / "a.md"
    doc.write_bytes(b"Alpha version.")

    first = run(folder)
    assert first.exit_code == 0, first.output
    assert "a.md: ingested" in first.output and "1 file(s), 1 chunk(s)" in first.output
    assert "a.md: unchanged" in run(folder).output
    doc.write_bytes(b"Beta version.")
    assert "a.md: replaced" in run(folder).output


def test_bad_file_gives_exit_code_1_and_names_it(run, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    result = run(bad)
    assert result.exit_code == 1 and "bad.pdf: invalid_file_type" in result.output


def test_missing_path_gives_a_clean_message(run, tmp_path):
    result = run(tmp_path / "missing")
    assert result.exit_code == 1 and "not_found" in result.output


def test_index_dir_option(run, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_bytes(b"Alpha.")
    other = tmp_path / "other_index"
    assert run(folder, "--index-dir", other).exit_code == 0
    assert (other / "store" / "manifest.json").exists()
