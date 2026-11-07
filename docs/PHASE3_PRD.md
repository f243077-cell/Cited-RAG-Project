# CiteRAG Phase 3: Product Requirements (agent mode)

Status: **plan, not built.** Written 2026-11-07 for milestones M9 to M12. Read with `docs/PHASE3_ARCHITECTURE.md`. `docs/PRD.md`
stays the source for M0 to M8; this file adds to it and changes none of its requirements.

## 1. Summary

Phase 3 adds an **Agent mode** to CiteRAG. It covers the roadmap's planning and reasoning, tool calling, MCP and multi-agent
skills. The agent breaks a question into at most three sub-questions. It searches the documents with read-only tools in a ReAct
loop, checks whether the evidence is enough, then answers through the existing cited-answer path, so every citation is still
verified by code. The same three tools are exposed to other MCP clients through a `citerag mcp` server. An evaluation (E8) compares
standard, Deep and Agent modes. An optional milestone asks when multi-agent adds value.

## 2. Roadmap mapping

| Roadmap item | Priority | Where it is practised |
|---|---|---|
| 3.1 Planning and reasoning: ReAct, reflection, task decomposition, planner-executor | must | M10: planner, ReAct tool loop, reflection node |
| 3.2 Tool calling and agent protocols: function calling, MCP basics, sandboxing and tool safety | must | M9: tools and MCP server; M10: native tool calling with a JSON fallback |
| 3.3 Multi-agent: supervisor, orchestrator-worker, handoff; when it adds value | high | M12: supervisor with two workers, compared with the single agent and with CrewAI Deep mode |
| 3.4 AI-assisted development | must | M12: `docs/AI_ASSISTED_DEV.md`, drawn from git history, progress files and gotchas |

## 3. Users and scenarios

| ID | Story |
|---|---|
| US-P3-1 | As the owner, I ask a question that spans two documents, and Agent mode finds both parts and cites them, where standard mode found one |
| US-P3-2 | As the owner, I open the trace under an agent answer and see the plan, each tool call and its arguments, the reflection verdict and the request count |
| US-P3-3 | As the owner, I add `citerag mcp` to an MCP client and search my documents from there, read-only |
| US-P3-4 | As the owner, I read E8 and know which mode to use, and what each costs in requests and seconds |

## 4. Scope

**In:** three read-only tools, the MCP stdio server, a LangGraph agent (plan, act, reflect, synthesize), Agent mode in the API and
on the Chat page with a visible trace, E8, and an optional supervisor graph with its comparison.

**Out:** tools that write, delete or reach the network; a shell or code-execution tool; MCP over HTTP or SSE; image questions in
Agent mode (decision 3); a LangGraph checkpointer or long-term agent memory; autonomous agents outside a user request; token
streaming. Multi-user auth and cloud deployment stay out (PRD section 4).

## 5. Functional requirements

New prefixes, so no ID clashes with `docs/PRD.md`: F-TOOL, F-MCP, F-GRF (graph), F-AUI (agent UI and API), F-AEV (agent
evaluation), F-SUP (supervisor), F-DEV (AI-assisted development), NF-P3.

### 5.1 Tools (M9)

- **F-TOOL-1 [MUST]** Three tools: `list_documents()`, `search_documents(query, doc_ids?, k)` and `read_page(doc, page)`. They live
  in a framework-free `tools/` package and reuse the existing retriever, docstore and manifest.
- **F-TOOL-2 [MUST]** Every argument is validated by a Pydantic model with `extra="forbid"`, bounded strings and numbers. `k` is
  1 to 10 (default 5), a query is 1 to 500 characters, and `page` is at least 1.
- **F-TOOL-3 [MUST]** Results are capped: at most `k` hits, and at most `AGENT_TOOL_RESULT_CHARS` characters per call (default
  6000). Text that reaches a model is rendered with `render_passages`, so documents stay escaped, untrusted data.
- **F-TOOL-4 [MUST]** `search_documents` applies the relevance gate: below `MIN_TOP_SIMILARITY` it returns "no relevant passages"
  and makes no LLM call. Agent mode never runs the rewrite step.
- **F-TOOL-5 [MUST]** Tools are read-only. No tool writes, deletes, runs code, opens a file path, or reaches the network. `doc` is
  resolved through the manifest, never used as a path.
- **F-TOOL-6 [MUST]** A tool error (bad argument, unknown document, empty page) comes back as a typed tool result, not an
  exception, so the agent can recover and the loop cannot crash.
- **F-TOOL-7 [MUST]** A `doc_ids` filter on the request limits every tool call in that request.

### 5.2 MCP server (M9)

- **F-MCP-1 [MUST]** `citerag mcp` runs an MCP server over stdio with exactly the three tools, their input schemas and structured
  output. It exposes no prompts and no resources.
- **F-MCP-2 [MUST]** stdout carries only JSON-RPC. Logs go to stderr.
- **F-MCP-3 [MUST]** The default mode calls the running backend over HTTP, so Chroma keeps one writer; `--local` opens the index
  in-process for when the API is stopped (decision 1).
- **F-MCP-4 [MUST]** Tests use the SDK's in-process client (`create_connected_server_and_client_session`): list the tools, call
  each one, and check that bad arguments come back as `isError`.
- **F-MCP-5 [SHOULD]** One subprocess test starts `citerag mcp --local` and completes `initialize` and `tools/list` over real stdio.

### 5.3 Agent graph (M10)

- **F-GRF-1 [MUST]** A LangGraph `StateGraph` with these nodes: `plan` (at most `AGENT_MAX_SUBQUESTIONS` sub-questions, default 3),
  `act` (one ReAct step: choose tool calls), `tools` (run them, no LLM), `reflect` (is the evidence enough for each sub-question:
  stop, or one more search) and `synthesize`.
- **F-GRF-2 [MUST]** `synthesize` uses the existing path: `build_context`, `generate` with the `answer` prompt, `verify_citations`,
  and one corrective retry. The result is the same `CitedAnswer` as in standard mode.
- **F-GRF-3 [MUST]** Every LLM call in the graph goes through the gateway, so it counts in `llm_calls`, the limiter and the daily
  guard. The graph never builds its own LLM client.
- **F-GRF-4 [MUST]** Limits: `AGENT_MAX_STEPS` (default 6) counts the LLM calls of plan, act and reflect. When the count is reached,
  the graph goes to `synthesize` with what it has. LangGraph's `recursion_limit` (default 25) is a second stop, and
  `AGENT_TIMEOUT_S` (default 240) caps the whole request. The worst case is 8 LLM calls, 9 with a follow-up's condense step.
- **F-GRF-5 [MUST]** Before the graph starts, the daily guard must have room for the worst case of 8; otherwise the API returns 429
  `budget_exceeded`.
- **F-GRF-6 [MUST]** Native tool calling first. If the model does not return a valid tool call, the gateway falls back, once per
  model, to a JSON action parsed with `parse_model`. `AGENT_ACTION_METHOD` can force either method.
- **F-GRF-7 [MUST]** The response carries an `AgentTrace`: the plan, each step with its node, tool name, validated arguments, hit
  count, LLM requests and milliseconds, the stop reason, and the total requests. Tool results appear only as references (document,
  page), never as text.
- **F-GRF-8 [MUST]** Agent mode neither reads nor writes the cache, and it refuses with "not found" without calling `synthesize`
  when no tool call found a passage above the gate.
- **F-GRF-9 [MUST]** The canary injection items give 0 leaks in Agent mode, and no tool argument in a trace contains a canary.

### 5.4 Agent mode in the API and UI (M10)

- **F-AUI-1 [MUST]** `ChatRequest.mode` accepts `"agent"` (approved). `ChatResponse.trace: AgentTrace | None = None` (approved).
- **F-AUI-2 [MUST]** The Chat page offers Standard, Deep and Agent, warns of the request cost of the last two, and shows the trace in
  an expander under each agent answer: plan, steps, tool calls with arguments, requests and seconds.
- **F-AUI-3 [MUST]** An agent request with an image returns 400 with a clear message (decision 3).
- **F-AUI-4 [SHOULD]** The Dashboard splits requests by mode.

### 5.5 Evaluation E8 (M11)

- **F-AEV-1 [MUST]** E8 compares standard, Deep and Agent on the **verified** golden set, without image items: full coverage on
  `multi_chunk` items, correctness, faithfulness, citation validity, correct refusals, injection leaks, requests and latency.
- **F-AEV-2 [MUST]** Full coverage is measured on the passages the generator saw (`context coverage`), so every mode is judged on
  what reached the answer. For standard mode that is the same as before; for Agent mode it is the union of tool hits that survived
  `build_context`.
- **F-AEV-3 [MUST]** Runs reuse the M4 runner: `--resume`, `--max-requests`, `--skip-judge`, and a stop on `llm_unavailable`. One
  report, `docs/results/E8_modes.md`, with counts next to percentages and the noise rule (under 3 points is noise).
- **F-AEV-4 [MUST]** Before any live run, the command prints its request estimate. Nothing runs without the owner's go.

### 5.6 Supervisor and AI-assisted development (M12, optional)

- **F-SUP-1 [COULD]** A LangGraph supervisor with two workers, a researcher (the tool loop) and a writer (synthesis), using handoffs.
- **F-SUP-2 [COULD]** Compare supervisor, single agent and CrewAI Deep mode on the same items: requests, latency and quality, then
  write `docs/results/M12_multi_agent.md` answering "when does multi-agent add value".
- **F-DEV-1 [MUST in M12]** `docs/AI_ASSISTED_DEV.md`: per milestone, what was delegated to Claude Code, what it got wrong (for
  example the empty-cache bug), and the prompts that worked. Drawn from git history, the progress files and the gotchas.

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NF-P3-1 | **Framework boundaries:** LangGraph only in `agent_graph/`, the MCP SDK only in `mcp_server/`, tools framework-free in `tools/`. A test enforces it, as it does for CrewAI |
| NF-P3-2 | **Quota:** typical agent question 4 LLM requests, 6 with one more search round, at most 8 (9 for a follow-up). Each failed HTTP attempt adds to that, and the guard counts it |
| NF-P3-3 | **Latency:** agent p50 at most 3 times standard p50 on E8. Record the real figure, as M3 did for standard |
| NF-P3-4 | **Testability:** the default `pytest` makes no network call. The graph runs on `FakeGateway`, MCP on the in-process client, and native tool calls on the fake OpenAI server |
| NF-P3-5 | **Safety:** read-only tools, validated arguments, capped results, escaped passages, step and time limits, and no new outbound network path |
| NF-P3-6 | **Docs:** every milestone ends with `docs/results/M<n>_test_guide.md` and an updated `docs/results/PROGRESS.md` |

## 7. Milestones

Build in order and stop after each, as before. Nothing in M9 to M12 needs a live request to pass its tests. The live steps are
listed separately and run only on the owner's go.

| # | Milestone | Deliverables | Exit criteria | Live requests |
|---|---|---|---|---|
| **M9** | Tools and MCP server | `tools/`, `/v1/tools/*` endpoints (decision 1), `mcp_server/` with `citerag mcp`, boundary test, M9 test guide | In-process client lists 3 tools and calls each; bad arguments give `isError`; caps hold; a subprocess completes `initialize` and `tools/list` over stdio; `pytest` and `ruff` pass | 0 |
| **M10** | Agent graph and Agent mode | `agent_graph/`, gateway `tool_step` plus JSON fallback, prompts, config keys, API and Chat mode with trace, M10 test guide | End to end on `FakeGateway` and the fake server: plan, act, reflect, synthesize; step and recursion limits stop the graph; guard refusal before any call; trace matches `llm_calls` rows; canaries escaped | Spike of at most 3 requests plus 1 agent question (at most 8), on approval |
| **M11** | Evaluation E8 | `e8_*.yaml`, context coverage, N-run compare, E8 report writer, M11 test guide | A fake rehearsal produces the full E8 table; the estimate prints with no request; live run deferred until the golden set is verified | E8 itself, on approval (section 9) |
| **M12** | Optional: supervisor, comparison, AI-assisted development | `agent_graph/supervisor.py`, comparison report, `docs/AI_ASSISTED_DEV.md`, M12 test guide | Supervisor passes the same fake tests as the agent; the comparison command and estimate run; the dev log covers M0 to M11 | Comparison, on approval |

## 8. Risks

| Risk | Mitigation |
|---|---|
| A free model returns no valid tool call, or ignores `tool_choice` | JSON action fallback with `parse_model`; a 3-request spike in M10 before any wider use |
| Agent mode costs 4 to 8 times standard and drains the 40-a-day guard | Up-front guard check for 8, step limit, Agent toggle off by default, E8 estimate and cap |
| Injected document text steers tool calls (tool hijacking) | Tools are read-only with no outbound path, so the worst case is wasted requests and a misleading answer. Passages are escaped, canary checks run on answers and tool arguments, and a step limit applies |
| LangGraph 1.x changes its API | Pin at the installed version range; the graph is about 200 lines, kept small and in one package |
| Two processes open Chroma (API and MCP) | The MCP server calls the API by default; `--local` only when the API is stopped |
| The judge is the answerer's model | Labelled provisional, as in M4 to M7; a different-family judge is still the owner's choice |
| E8 runs on an unverified golden set | E8 is deferred until items are verified; the fake rehearsal proves the pipeline only |
| Over-engineering: multi-agent adds cost but no quality | M12 is optional and exists to measure exactly that |

## 9. Request budget (estimates, not measurements)

| Activity | LLM requests |
|---|---|
| Agent question: plan, act, reflect, synthesize | 4 typical |
| With one more search round (act and reflect again) | 6 |
| Hard cap: 6 plan, act and reflect calls, synthesize, one citation retry | 8 |
| Follow-up question (condense) | +1 |
| Each failed HTTP attempt inside any call | +1, counted by the guard |
| M10 tool-calling spike | at most 3 |
| M10 first live agent question | at most 8 |

**E8 on 38 non-image items** (26 answerable or `multi_chunk` with judge, 7 unanswerable, 5 injection), if the verified set keeps
today's mix:

| Mode | Expected | Worst case |
|---|---|---|
| Standard | about 71 | 140 |
| Deep | about 156 | 216 |
| Agent | about 216 | 330 |
| All three | about 443 | 686 |

If the E7 standard and Deep runs used the same verified set and settings, E8 adds only the agent run, about 216 requests. At a
30-a-day cap that is about 8 days for the agent run, or about 15 for all three. With credits (OPEN-5, 1,000 a day) it fits in one
day. A sample with `--limit 10` costs about 120 for all three modes.

## 10. Decisions

Approved by Coder: `langgraph` and `mcp` as direct dependencies, `ChatRequest.mode="agent"`, `ChatResponse.trace`. On 2026-11-07
Coder delegated the remaining decisions; each is taken below, with its reason.

1. **Tool contracts and `POST /v1/tools/{name}`: adopted.** `ListDocumentsArgs`, `SearchArgs`, `ReadPageArgs`, `ToolHit` and
   `ToolResult` (PHASE3_ARCHITECTURE section 5) are the public contract. The endpoints let `citerag mcp` use the running API, so
   Chroma keeps one writer; `--local` is for when the API is stopped.
2. **Package `mcp_server/`, not `mcp/`.** A package named `mcp` inside `citerag` can shadow the SDK when a script runs from inside
   the package; a distinct name removes the doubt at no cost.
3. **No images in Agent mode in v1.** An image adds a vision call and a second query to every plan. An agent request with an image
   returns 400.
4. **Agent mode skips the cache**, like Deep mode, because the trace and the cost are what it exists to show.
5. **Config keys:** `AGENT_ENABLED`, `AGENT_MAX_SUBQUESTIONS`, `AGENT_MAX_STEPS`, `AGENT_RECURSION_LIMIT`, `AGENT_TIMEOUT_S`,
   `AGENT_TOOL_RESULT_CHARS`, `AGENT_CONTEXT_MAX_PASSAGES`, `AGENT_ACTION_METHOD`, `MCP_MODE`. `k` stays fixed at 1 to 10 in the
   contract, so MCP clients and the agent see one bound and no extra keys are needed.
6. **Tool-calling model: `nvidia/nemotron-3.5-super-vl:free`.** OpenRouter's public list (2026-11-07) marks it with tools,
   structured outputs and `response_format`. It is checked in M10 with at most 3 requests (PHASE3_ARCHITECTURE section 10).
   `AGENT_ACTION_METHOD=auto` falls back to the JSON action if tool calls fail.
7. **Tool-hijack fixture: written in M11 as a fixture file only.** Golden items stay the owner's to create and verify.
8. **E8 waits for the verified golden set.** M11 builds and rehearses it on fakes; the live run is listed with its estimate and cap.
9. **Judge candidates:** `liquid/lfm-2.5-2.6b:free` now expires on 2026-11-13, so it is no longer the first choice. Recorded as a
   gotcha; the judge stays the owner's choice.
10. **ADR-10 is revisited, not overturned.** The standard path stays plain functions; only Agent mode and the supervisor use
    LangGraph.
