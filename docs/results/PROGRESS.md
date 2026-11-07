# Build progress: M6, M7, M8

Run started 2026-11-06. Quota rule for this run: no live LLM requests on either provider. Requests used so far: 0.

| Milestone | State |
|---|---|
| M6 semantic cache | built and pushed; exit criteria need your labels and a live replay |
| M7 CrewAI | built and pushed; E7 and every live crew run are left unrun |
| M8 polish | built and pushed; Docker untested; a live first answer not run |
| Phase 3 M9 tools and MCP | built and pushed; 0 LLM requests |
| Phase 3 M10 agent graph and Agent mode | built and pushed; 3 LLM requests (tool-calling spike) |
| Phase 3 M11 evaluation E8 | built and pushed; the live E8 run waits for the verified golden set |
| Phase 3 M12 supervisor, comparison, dev log | built and pushed; the live comparison waits for the verified golden set |
| **Phase 3** | **complete.** Live requests in the phase: 3 (the M10 spike). Decisions taken under delegation: PHASE3_PRD section 10 |

## M6

**Done:** `PersistentCache` over the reference cache, guard, namespace invalidation at start-up and on corpus change, `citerag cache stats|clear`, `/v1/cache/stats|clear`, `/v1/logs/*`, `eval cache-sweep`, `eval bench-cache`, Dashboard page and charts 1 to 7, Chat toggle and badge, 80 candidate pairs, `scripts/review_cache_pairs.py`, `scripts/make_workload.py`, `M6_test_guide.md`.

**Not done:** threshold chosen (needs your labels), E5 chart, E6 report (needs a live run), and tokens saved per request (decision 2).

## M7

**Done:** `agents/runtime.py`, Deep crew, question writer plus critic, failure analyst, `eval gen-candidates`, `eval analyze-failures`, `mode` in experiment files, `e7_standard.yaml`, `e7_deep.yaml`, Deep toggle on Chat, six versioned prompt files, `M7_test_guide.md`. Tested against the fake server only.

**Not done:** E7 write-up (needs a live run; command, estimate and cap are in the guide). No crew has reached a real model.

## M8

**Done:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, request guard tests against the real gateway, `citerag quota`, `citerag logs purge`, README with diagram and provisional results, `security_checklist.md`, `M8_test_guide.md`. A fresh clone on C: passed install, `citerag version`, `/health` and the Streamlit health check, and 529 of 530 tests with no key. A second clone with the fix below passed `test_chunking.py` and a multi-chunk ingest; the full suite was not repeated in a clone, but it passes 530 of 530 in the repo.

**Found by the clone:** uv hard-links packages when its cache shares a drive with the project, and NLTK refuses hard-linked data, so chunking crashed. `[tool.uv] link-mode = "copy"` fixes it.

**Not done:** an image build (the Docker daemon was not running, only `docker compose config` passed), a first answer from a clone (it needs one live request), the 15-minute clone target with cold downloads, and Whisper, skipped as you said. The SSE endpoint and the API rate limit are SHOULD items and remain unbuilt.

## Decisions for Coder

1. **The cache is on with the placeholder threshold, and that looks unsafe.** `CACHE_ENABLED` defaults to true and `CACHE_THRESHOLD` stays 0.92, as you ordered. A smoke run of the sweep on a throwaway copy, using my proposed labels and not yours, gave a false-hit rate of 33% at 0.92 (14 of 43 near misses) and 0% only at 0.98, where recall is 32%. The set is deliberately adversarial and is not a result. Until you run the sweep, set `CACHE_ENABLED=false` in `.env` if you will rely on answers.
2. **Tokens saved (F-CAC-7) needs a public schema change.** The requests table and `CacheInfo` have no such field, so the log records saved requests and saved shadow dollars only. `bench-cache` still reports tokens with and without the cache. Approve `saved_tokens` on `CacheInfo` and `requests` if you want it per request.
3. **The guard misses meaning the reference code cannot see.** `guard_ok` compares digits and a word list, so these pass it: "first" vs "second", "days" vs "hours", "address" vs "leave out". Near misses of that kind can only be stopped by the threshold, which is why it must be so high. A wider guard means changing reference code, so it is your call.
4. **A crew writes one `llm_calls` row per request, not one per crew.** `CLAUDE.md` rule 2 and ARCHITECTURE section 12 said one row, but the daily guard counts rows, so a Deep question would have counted as 1. I changed the code and both documents. Say so if you want it back.
5. **Crew requests are counted from outside.** CrewAI does not report failed attempts, and it retries a 429 twice with no setting to stop it. The log therefore overestimates on a failure, which errs towards the stricter guard, and a real run could still differ. Compare `/health` with `GET /api/v1/key` on your first Deep runs.
6. **Crews parse strictly.** If the first live Deep request fails with `structured_output_failed` on nemotron, the choice is another model for `MODEL_FAST` and `MODEL_STRONG`, or a parsing layer over CrewAI.
7. **Security gaps are listed for you to accept or close.** See the Gaps section of `security_checklist.md`: no API rate limit, uploads and chat bodies are read before they are rejected, `/health` needs no key, the container runs as root, and a timed-out crew keeps running until its agents' limit.

## Phase 3: M9

**Done:** `tools/` (three read-only tools, contracts, caps, gate, filter), `POST /v1/tools/{name}`, `mcp_server/` with
`citerag mcp` over stdio (HTTP backend by default, `--local`), a boundary test for LangGraph, MCP and CrewAI, `M9_test_guide.md`.
A real stdio subprocess test proves stdout carries only JSON-RPC.

## Phase 3: M10

**Done:** `agent_graph/` (LangGraph: plan with code-run searches, tools, reflect, act), gateway `tool_step` with native tool calls
and a once-per-model JSON fallback, `check_room` for the worst case of 8, Agent mode in the pipeline with synthesis on the standard
generate-and-verify path, `ChatRequest.mode="agent"` and `ChatResponse.trace`, a Mode selector and a plain-text trace on the Chat
page, `mode: agent` in experiment files, `M10_test_guide.md`.

**Live spend:** 3 requests, the tool-calling spike. The result changed the design (PHASE3_ARCHITECTURE section 3).

**Not done:** a live agent question. The guard was at 38 of 40 after the spike, and the M10 guide lists it as a 3-request step.

**Docker (M8), checked in passing:** the image builds (21 GB). The backend first failed on an `env_file` parsing gotcha, now fixed in
`.env.example`, and in `.env` by moving two comments without changing a value. Both services now share one image.

## Phase 3: M11

**Done:** `e8_standard.yaml`, `e8_deep.yaml`, `e8_agent.yaml`; `eval/modes.py` (context coverage on what reached the answer, canary
check on tool arguments, request estimate, the E8 table); `citerag eval run --estimate`; `citerag eval report-e8`;
`ItemRecord.trace`; the tool-hijack fixture `eval/fixtures/poisoned/agent_research_memo.md` (no golden item: that is yours);
a no-network rehearsal of all three modes; `M11_test_guide.md`.

**Not done:** the live E8 run. It needs the verified golden set and about 405 requests for all three modes.

## Phase 3: M12

**Done:** `agent_graph/supervisor.py` (supervisor, researcher and writer, with handoffs and caps), `agent_graph/factory.py`,
`AGENT_ARCHITECTURE`, the `supervisor` prompt, `m12_single.yaml` and `m12_supervisor.yaml`, `report-e8 --title` for the comparison,
the supervisor's cost in `--estimate`, `docs/AI_ASSISTED_DEV.md`, `M12_test_guide.md`.

**Not done:** the live comparison and its conclusion (about 90 to 210 requests on 15 items), which waits for the verified golden set.

## Phase 3: what needs a live run, in order

1. One Agent-mode question (3 requests usually): M10 guide, step 3.
2. One supervisor question (5 usually): M12 guide, step 2.
3. E8 on the verified set (about 405 for all three modes): M11 guide.
4. The M12 comparison (about 90 if E8's runs are reused): M12 guide.
