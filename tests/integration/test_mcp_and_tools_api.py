"""Tools over HTTP and MCP: real Chroma and BM25 with HashEmbedding, the SDK's in-process client, and one real stdio run."""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from citerag.api.main import create_app
from citerag.index import store as store_module
from citerag.index.store import IndexStore
from citerag.ingest.pipeline import Ingestor
from citerag.llm.fake import FakeGateway
from citerag.mcp_server.server import HttpBackend, LocalBackend, build_server
from citerag.tools.contracts import ToolResult
from tests.helpers import HashEmbedding
from tests.integration.test_chat_api import POLICY, factory, upload

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api(settings):
    with TestClient(create_app(settings, services_factory=factory(FakeGateway()))) as client:
        upload(client, ("policy.txt", POLICY))
        yield client


def tool(client, name, **args):
    return client.post(f"/v1/tools/{name}", json=args)


def test_the_api_runs_each_tool(api):
    docs = ToolResult.model_validate(tool(api, "list_documents").json())
    assert [d.doc_name for d in docs.documents] == ["policy.txt"]
    found = ToolResult.model_validate(
        tool(api, "search_documents", query="how many days to return items").json()
    )
    assert found.ok and found.hits[0].doc_name == "policy.txt" and found.top_similarity is not None
    page = ToolResult.model_validate(tool(api, "read_page", doc="policy.txt", page=1).json())
    assert page.ok or page.message.startswith("no text stored")


def test_bad_arguments_are_a_400_with_the_error_body(api):
    response = tool(api, "search_documents", query="q", k=50)
    assert response.status_code == 400 and response.json()["error"]["code"] == "validation_error"
    assert tool(api, "delete_everything").status_code == 400


def test_an_unknown_document_is_a_result_not_an_http_error(api):
    response = tool(api, "read_page", doc="../.env", page=1)
    assert response.status_code == 200 and response.json()["error"] == "unknown document: ../.env"


def test_tool_routes_need_the_key_when_one_is_set(settings):
    keyed = settings.model_copy(update={"citerag_api_key": "secret"})
    with TestClient(create_app(keyed, services_factory=factory(FakeGateway()))) as client:
        assert tool(client, "list_documents").status_code == 401


@pytest.fixture
def local_index(settings, monkeypatch):
    monkeypatch.setattr(store_module, "make_embedder", lambda _: HashEmbedding())
    store = IndexStore(settings.index_dir, HashEmbedding())
    Ingestor(settings, store, settings.index_dir).ingest_bytes("policy.txt", POLICY)
    store.close()
    return LocalBackend(settings, settings.index_dir)


def mcp_session(backend, check):
    async def run():
        async with create_connected_server_and_client_session(build_server(backend)._mcp_server) as client:
            return await check(client)

    return asyncio.run(run())


def test_the_server_lists_exactly_three_tools_with_schemas(local_index):
    async def check(client):
        return (await client.list_tools()).tools

    tools = {t.name: t for t in mcp_session(local_index, check)}
    assert sorted(tools) == ["list_documents", "read_page", "search_documents"]
    search = tools["search_documents"].inputSchema["properties"]
    assert search["k"]["maximum"] == 10 and search["query"]["maxLength"] == 500
    assert all(t.outputSchema for t in tools.values())


def test_each_tool_answers_through_mcp(local_index):
    async def check(client):
        listed = await client.call_tool("list_documents", {})
        found = await client.call_tool("search_documents", {"query": "how many days to return items", "k": 2})
        return listed, found

    listed, found = mcp_session(local_index, check)
    assert not listed.isError and listed.structuredContent["documents"][0]["doc_name"] == "policy.txt"
    assert not found.isError and found.structuredContent["hits"][0]["doc_name"] == "policy.txt"


def test_bad_arguments_come_back_as_is_error(local_index):
    async def check(client):
        return [
            await client.call_tool("search_documents", {"query": "q", "k": 99}),
            await client.call_tool("read_page", {"doc": "policy.txt", "page": 0}),
        ]

    assert all(result.isError for result in mcp_session(local_index, check))


def test_the_server_offers_no_prompts_and_no_resources(local_index):
    async def check(client):
        return (await client.list_prompts()).prompts, (await client.list_resources()).resources

    assert mcp_session(local_index, check) == ([], [])


def test_http_backend_uses_the_api(api):
    backend = HttpBackend("http://testserver", client=api)
    result = backend.call("search_documents", {"query": "how many days to return items"})
    assert result.ok and result.hits
    bad = backend.call("search_documents", {"query": "q", "k": 99})
    assert not bad.ok and bad.error.startswith("validation_error")


def test_http_backend_says_how_to_recover_when_the_api_is_down():
    result = HttpBackend("http://127.0.0.1:9").call("list_documents", {})
    assert not result.ok and "citerag mcp --local" in result.error


def test_stdio_server_speaks_only_json_rpc_on_stdout(tmp_path):
    """A real subprocess: one stray print or log line on stdout would break the client."""
    env = {
        **{k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE")},
        "OPENROUTER_API_KEY": "placeholder",
        "MODEL_FAST": "v/f:free",
        "MODEL_STRONG": "v/s:free",
        "MODEL_VISION": "v/v:free",
        "MODEL_JUDGE": "v/j:free",
        "DATA_DIR": str(tmp_path / "data"),
        "PYTHONPATH": str(ROOT / "src"),
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from citerag.cli import app; app()", "mcp", "--local"],
        env=env,
        cwd=str(tmp_path),
    )

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            names = sorted(t.name for t in (await session.list_tools()).tools)
            listed = await session.call_tool("list_documents", {})
            return names, listed

    names, listed = asyncio.run(asyncio.wait_for(run(), 120))
    assert names == ["list_documents", "read_page", "search_documents"]
    assert listed.structuredContent["message"] == "no documents yet"


def test_stdout_is_untouched_when_the_cli_starts(tmp_path):
    """Belt and braces for the test above: nothing before the first JSON-RPC line."""
    env = {
        **os.environ,
        "OPENROUTER_API_KEY": "x",
        "MODEL_FAST": "a",
        "MODEL_STRONG": "a",
        "MODEL_VISION": "a",
    }
    env |= {"MODEL_JUDGE": "a", "DATA_DIR": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        }  # fmt: skip
    )
    done = subprocess.run(
        [sys.executable, "-c", "from citerag.cli import app; app()", "mcp", "--local"],
        input=request + "\n",
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert lines and all(re.match(r"^\{.*\"jsonrpc\"", line) for line in lines), done.stdout[:300]


def test_warm_up_loads_the_index_and_the_embedder_before_serving(local_index):
    local_index.warm_up()
    assert local_index._library is not None and local_index._retriever is not None
    assert local_index.call("search_documents", {"query": "how many days to return items"}).ok


def test_warm_up_on_an_empty_index_loads_nothing(settings):
    backend = LocalBackend(settings, settings.index_dir)
    backend.warm_up()
    assert backend._library is None


def test_warm_up_keeps_library_prints_off_stdout(local_index, monkeypatch, capsys):
    original = local_index._open

    def noisy():
        print("a library talking on stdout")
        return original()

    monkeypatch.setattr(local_index, "_open", noisy)
    local_index.warm_up()
    captured = capsys.readouterr()
    assert "talking" not in captured.out and "talking" in captured.err
