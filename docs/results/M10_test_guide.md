# M10 test guide: agent graph and Agent mode

Run from the repo root.

| # | Do this | Expect | LLM requests |
|---|---|---|---|
| 1 | `uv run pytest tests/unit/test_agent_routing.py tests/integration/test_agent_mode.py tests/unit/test_ui.py` | All pass: 3 requests on the happy path, 5 with one more round, 8 at worst; the guard refuses before any call; the cache is untouched | 0 |
| 2 | `uv run citerag quota` | At least 8 left under the OpenRouter limit before step 3, or step 3 is refused with HTTP 429 | 0 |
| 3 | Start the API and UI. On Chat pick **Agent** and ask "What are the CSF Tiers, and what does GOVERN do in the AI RMF?" | A cited answer with verified cards. Under it, "Agent trace": the plan, one `search_documents` per sub-question, the reflection note, synthesis, and the request count | 3 usually, 5 with one more round, at most 8 |
| 4 | Open the trace | Arguments and notes are plain text, hits are references such as `NIST.CSWP.29.pdf p8`, and no passage text appears | 0 |
| 5 | Run `curl -s "localhost:8000/v1/logs/llm_calls?limit=10"` | Rows `agent_plan`, `agent_reflect`, maybe `agent_act`, then `generate`, all with this request's id; their count equals the trace's | 0 |
| 6 | Ask in Agent mode with an image attached | HTTP 400: "Agent mode does not take images yet" | 0 |
| 7 | Pick a document under "Search only in", ask in Agent mode | Every tool call in the trace stays in that document | as step 3 |
| 8 | Ask something the documents do not hold, such as "What is the capital of France?" | "I could not find this in the uploaded documents." The trace stops with `no_evidence` and has no synthesis step | 2 |
| 9 | Set `AGENT_ENABLED=false`, restart, ask in Agent mode | HTTP 400 "Agent mode is not available here". Restore it | 0 |

**No-request alternative for steps 3 to 9:** step 1 covers each case on `FakeGateway`. The gateway's native tool calls and the JSON
fallback run against the fake OpenAI server in the same file.

**Tool calling on the free model.** The M10 spike spent 3 requests (PHASE3_ARCHITECTURE section 10). If step 3 shows `act` steps
that call nothing useful, set `AGENT_ACTION_METHOD=json` and compare.

**Failure means any of these:**

- A step above gives another result, or the trace's request count differs from the `llm_calls` rows.
- More than 8 requests for one question, or any request after a 429 `budget_exceeded`.
- An agent answer carries a citation that is not verified, or arrives with the cached badge.
- A trace shows passage text, or a canary from `eval/fixtures/poisoned` appears in an answer or a tool argument.
- A tool call in the trace reaches outside the "Search only in" filter.
