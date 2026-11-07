# M9 test guide: tools and MCP server

Run from the repo root. Nothing here makes an LLM request.

| # | Do this | Expect | LLM requests |
|---|---|---|---|
| 1 | `uv run pytest tests/unit/test_tools.py tests/unit/test_boundaries.py tests/integration/test_mcp_and_tools_api.py` | All pass, including a real stdio subprocess whose stdout holds only JSON-RPC | 0 |
| 2 | Start the API. Run `curl -s -X POST localhost:8000/v1/tools/list_documents -H "content-type: application/json" -d "{}"` | `"ok": true` and your documents | 0 |
| 3 | `curl -s -X POST localhost:8000/v1/tools/search_documents -H "content-type: application/json" -d "{\"query\": \"CSF Tiers\", \"k\": 3}"` | 3 hits with `doc_name`, `page` and text, and `top_similarity` | 0 |
| 4 | The same with `"k": 50` | HTTP 400, `validation_error`, naming `k` | 0 |
| 5 | `curl -s -X POST localhost:8000/v1/tools/read_page -H "content-type: application/json" -d "{\"doc\": \"../.env\", \"page\": 1}"` | `"ok": false`, `"error": "unknown document: ../.env"`: a name is never a path | 0 |
| 6 | Search for something off-topic, such as `"capital of France"` | No hits, `"message": "no relevant passages"`, the gate value below `MIN_TOP_SIMILARITY` | 0 |
| 7 | With the API running, add the server to an MCP client (configuration below), then ask the client to list your CiteRAG documents | The client shows three tools and calls `list_documents` | 0 from CiteRAG; the client's own model is separate |
| 8 | Stop the API and call a tool from the client | The tool answers with `ok: false` and says to start the API or use `--local` | 0 |
| 9 | Change the configuration to `["run", "citerag", "mcp", "--local"]`, restart the client, search | Results come from the index directly | 0 |

**MCP client configuration** (Claude Desktop and similar clients). Use an absolute path for `cwd`:

```json
{
  "mcpServers": {
    "citerag": {
      "command": "uv",
      "args": ["run", "citerag", "mcp"],
      "cwd": "D:/PROJECTS/Cited RAG Project"
    }
  }
}
```

The server reads `.env` from `cwd`. With `CITERAG_API_KEY` set, it sends the key to the API.

**What the server does not do:** it has no write, delete, shell, file or network tool, no prompts and no resources, and no transport
but stdio. `citerag mcp --local` opens Chroma itself, so run it only with the API stopped.

**Failure means any of these:**

- A test above fails, or the subprocess test hangs: something printed to stdout.
- Any tool returns text from a file that is not in the library, or `read_page` accepts a path.
- `k` above 10, a query over 500 characters, or an extra argument is accepted.
- Results go past `AGENT_TOOL_RESULT_CHARS` characters without `"truncated": true`.
- `/v1/tools/*` answers without `X-API-Key` when `CITERAG_API_KEY` is set.
