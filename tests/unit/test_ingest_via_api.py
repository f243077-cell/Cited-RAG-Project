"""citerag ingest --via-api: one upload per file through the HTTP client, opening no index."""

import pytest
from typer.testing import CliRunner

from citerag import cli
from citerag.schemas import IngestReport
from citerag.ui import api_client
from citerag.ui.api_client import ApiError


class StubClient:
    uploaded: list[str] = []

    def __init__(self, base_url, api_key=None):
        pass

    def upload(self, files):
        name = files[0][0]
        StubClient.uploaded.append(name)
        if name == "bad.exe":
            raise ApiError("invalid_file_type", "not allowed")
        return [IngestReport(doc_id="d", doc_name=name, status="ingested", chunks=2)]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    StubClient.uploaded = []
    monkeypatch.setattr(api_client, "ApiClient", StubClient)
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "SOURCES.md").write_text("licences")
    return tmp_path


def test_each_file_is_uploaded_and_the_licence_table_is_skipped(folder):
    result = CliRunner().invoke(cli.app, ["ingest", str(folder), "--via-api"])
    assert result.exit_code == 0 and "a.md: ingested" in result.stdout
    assert StubClient.uploaded == ["a.md"]


def test_one_bad_file_does_not_stop_the_others_and_exits_nonzero(folder):
    (folder / "bad.exe").write_bytes(b"MZ")
    (folder / "z.md").write_text("zeta")
    result = CliRunner().invoke(cli.app, ["ingest", str(folder), "--via-api"])
    assert result.exit_code == 1 and "invalid_file_type" in result.stdout
    assert StubClient.uploaded == ["a.md", "bad.exe", "z.md"]


def test_missing_path_fails_with_a_message(tmp_path):
    result = CliRunner().invoke(cli.app, ["ingest", str(tmp_path / "nope"), "--via-api"])
    assert result.exit_code == 1 and "not_found" in result.stdout
