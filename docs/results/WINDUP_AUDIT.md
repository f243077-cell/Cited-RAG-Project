# Wind-up audit (M0 to M12)

Date: 2026-11-07. Zero LLM requests in this audit. Every check command below makes no LLM request; most are `pytest` files that
run on `FakeGateway`, the fake OpenAI server and a hash embedder.

## Suite, lint and install

| Check | Result |
|---|---|
| `uv sync --locked --extra dev` | Lock is current: 257 packages resolved, 224 checked |
| `uv run ruff check .` and `uv run ruff format --check .` | Clean |
| `uv run pytest` before the wind-up fixes | 625 passed, 1 deselected |
| `uv run pytest` after the wind-up fixes | 669 passed, 1 deselected, 0 skipped, 0 xfail |
| Skipped or xfail tests | **None.** One test is deselected by default: `tests/live/test_answer.py::test_one_real_cited_answer`, marked `live`, because it spends a real request (`uv run pytest -m live`) |
| `uv run python scripts/check_release.py` | Passes: no tracked secret, local or generated file, no attribution line, and every golden, cache-pair, experiment and `.env.example` file matches the code |

**Status words.** *done*: built, tested, and its exit criterion met. *partial*: built and tested, but the exit criterion needs a
human step or a live run that has not happened. *missing*: not built.

## Milestone exit criteria

| Milestone | Exit criterion (PRD section 8, PHASE3_PRD section 7) | Status | Why not done |
|---|---|---|---|
| M0 Scaffold | `pytest` green, `/health` 200 | done | |
| M1 Ingestion | Re-ingest is a no-op; a changed file replaces; counts correct | done | |
| M2 Retrieval | hit@k report on a first golden set | done | Numbers are PROVISIONAL: the items are unverified |
| M3 Cited answers | End-to-end Q&A with verified citations; refusals work | done | |
| M4 Evaluation | Baseline on at least 40 human-verified items; targets table updated | **partial** | 0 items verified (owner's task); no answerable run with verdicts |
| M5 Multimodal | US-5 and US-6 demonstrable; image items in the set | done | Image items exist as unverified drafts |
| M6 Semantic cache | Threshold chosen from a sweep; before/after report | **partial** | Needs your labels (sweep) and a live replay (E6) |
| M7 CrewAI | E7 Deep vs standard written up | **partial** | E7 not run; Deep mode runs live once in the wind-up smoke test if the budget allows |
| M8 Polish | A fresh clone runs from the README | done | Proven in the wind-up clean-clone step (WINDUP_SMOKE.md) |
| M9 Tools and MCP | Tools, MCP over stdio, in-process and subprocess tests | done | |
| M10 Agent | Graph on fakes, limits, guard, trace | done | One live agent question in the wind-up smoke test |
| M11 E8 | Rehearsal on fakes; estimate prints | done | The live E8 run waits for the verified set |
| M12 Supervisor | Same fake tests; comparison tooling; dev log | done | The live comparison waits for the verified set |

## Requirements, M0 to M8 (docs/PRD.md)

| ID | Status | Evidence | Check (no LLM request) |
|---|---|---|---|
| F-ING-1 | done | `ingest/loaders.py` allow-list, magic bytes, size cap | `uv run pytest tests/unit/test_loaders.py` |
| F-ING-2 | done | `ingest/pipeline.py` sha256 doc_id, unchanged/replaced | `uv run pytest tests/integration/test_ingest.py` |
| F-ING-3 | done | `ingest/chunking.py` SentenceSplitter, metadata | `uv run pytest tests/unit/test_chunking.py` |
| F-ING-4 | done | `ingest/manifest.py`; a mismatch now answers 503 with the reason | `uv run pytest tests/unit/test_manifest.py tests/integration/test_health.py` |
| F-ING-5 | done | `IngestReport` | `uv run pytest tests/integration/test_ingest.py` |
| F-ING-6 | done | Only captions call a model, and they are opt-in | `uv run pytest tests/integration/test_image_ingest.py` |
| F-ING-7 | missing | DOCX is not supported (SHOULD) | none |
| F-RET-1 | done | `index/retriever.py` hybrid RRF, top 20 to 5 | `uv run pytest tests/integration/test_retriever.py` |
| F-RET-2 | done | `doc_ids` filter | `uv run pytest tests/integration/test_retriever.py` |
| F-RET-3 | partial | Rerank built and switchable; default left off, because E3 ran only on unverified items (OPEN-2 open) | `uv run citerag eval retrieval --rerank --skip-type image --index-dir data/index_eval` |
| F-RET-4 | partial | Gate built; `MIN_TOP_SIMILARITY=0.55` is provisional until calibrated on the verified set (owner) | `uv run pytest tests/unit/test_pipeline.py` |
| F-RET-5 | done | Rewrite and regeneration, once each | `uv run pytest tests/unit/test_pipeline.py` |
| F-ANS-1 | done | `LLMCitedAnswer` to `CitedAnswer` by code | `uv run pytest tests/unit/test_reference.py` |
| F-ANS-2 | done | `rag/verify.py`; invalid citations flagged | `uv run pytest tests/unit/test_pipeline.py tests/unit/test_ui.py` |
| F-ANS-3 | partial | Refusal path built; 7 of 7 refusals measured on unverified items at gate 0.30, PROVISIONAL | `uv run pytest tests/unit/test_pipeline.py` |
| F-ANS-4 | done | Fuzzy match at 90, corrective retry, low confidence | `uv run pytest tests/unit/test_reference.py tests/unit/test_pipeline.py` |
| F-ANS-5 | done | `assemble` lowers confidence | `uv run pytest tests/unit/test_pipeline.py` |
| F-ANS-6 | done | Empty index: refusal, no LLM call | `uv run pytest tests/unit/test_pipeline.py` |
| F-CTX-1 | done | `rag/context.py` (reference code) | `uv run pytest tests/unit/test_reference.py` |
| F-CTX-2 | done | `render_passages` escaping, untrusted-data rule | `uv run pytest tests/unit/test_reference.py tests/unit/test_prompts.py` |
| F-CTX-3 | done | `version:` header logged per request and run | `uv run pytest tests/unit/test_prompts.py` |
| F-SAF-1 | partial | 0 of 5 leaks on unverified items, `i-005` through an image caption, PROVISIONAL | `uv run pytest tests/unit/test_golden_candidates.py` |
| F-SAF-2 | done | No HTML rendering; hashed upload names | `uv run pytest tests/unit/test_ui.py tests/unit/test_loaders.py` |
| F-SAF-3 | done | `SecretStr`, `.env` ignored, release check | `uv run python scripts/check_release.py` |
| F-CON-1 | done | `messages` table, last 4 turns | `uv run pytest tests/unit/test_log_store.py` |
| F-CON-2 | done | Condense for follow-ups; the cache uses the standalone question | `uv run pytest tests/unit/test_pipeline.py` |
| F-MM-1 | done | Vision analysis plus search query | `uv run pytest tests/unit/test_multimodal.py` |
| F-MM-2 | done | The user's image is never a citation | `uv run pytest tests/unit/test_multimodal.py` |
| F-MM-3 | done | Extraction, filters, cap, opt-in captions, caption cache | `uv run pytest tests/integration/test_image_ingest.py` |
| F-MM-4 | done | Thumbnail on image citations | `uv run pytest tests/unit/test_ui.py` |
| F-MM-5 | missing | Whisper voice input (COULD), skipped by owner decision | none |
| F-MM-6 | done | Image requests skip the cache | `uv run pytest tests/unit/test_multimodal.py` |
| F-STR-1 | done | `structured_call`, one feedback retry | `uv run pytest tests/unit/test_gateway.py` |
| F-STR-2 | done | Pydantic at every boundary | `uv run pytest tests/unit/test_reference.py` |
| F-STR-3 | done | Settings validated at start; a missing key is one clear message (DECISIONS.md, D-4) | `uv run pytest tests/unit/test_windup_fixes.py` |
| F-STR-4 | done | Minimal model-facing schemas | `uv run pytest tests/unit/test_reference.py` |
| F-CAC-1 | done | Lookup after condense, before retrieval | `uv run pytest tests/integration/test_cache_api.py` |
| F-CAC-2 | partial | Cosine plus threshold built; `CACHE_THRESHOLD=0.92` is an unvalidated placeholder, so the cache is off by default | `uv run pytest tests/unit/test_windup_fixes.py` |
| F-CAC-3 | done | `guard_ok` (reference code) | `uv run pytest tests/unit/test_reference.py` |
| F-CAC-4 | done | Only fully verified answers stored | `uv run pytest tests/unit/test_pipeline.py` |
| F-CAC-5 | done | Namespace and mode scoping, TTL, LRU | `uv run pytest tests/integration/test_cache_api.py tests/unit/test_cache_store.py` |
| F-CAC-6 | done | `use_cache=false`, `citerag cache clear`, `POST /v1/cache/clear` | `uv run pytest tests/integration/test_cache_api.py` |
| F-CAC-7 | partial | Hit, similarity and saved requests logged and charted; tokens saved per request not logged (needs a schema field) | `uv run pytest tests/unit/test_cache_plots.py` |
| F-CAC-8 | partial | `eval cache-sweep` built, refuses unlabelled pairs; 80 candidates wait for your labels | `uv run citerag eval cache-sweep` (prints that no labelled pairs exist) |
| F-CAC-9 | partial | `eval bench-cache` built and rehearsed on fakes; E6 not run | `uv run pytest tests/integration/test_bench_cache.py` |
| F-CAC-10 | done | `CacheBackend` interface | `uv run pytest tests/unit/test_noop_cache.py` |
| F-EVA-1 | missing | 0 verified items. 43 candidates (21 answerable, 5 multi_chunk, 7 unanswerable, 5 injection, 5 image) wait for review (owner) | `uv run python scripts/review_golden.py --out eval/results/golden_review.md` |
| F-EVA-2 | done | All metrics computed | `uv run pytest tests/unit/test_eval_scoring.py` |
| F-EVA-3 | partial | Judge built; it is the answerer's model, so verdicts are PROVISIONAL | `uv run pytest tests/unit/test_eval_runner.py` |
| F-EVA-4 | missing | Ragas cross-check (SHOULD) not built | none |
| F-EVA-5 | done | Run folders with config, items, summary; `eval compare` | `uv run pytest tests/integration/test_eval_run.py` |
| F-EVA-6 | partial | E1 to E3 charts and notes, PROVISIONAL; E4 to E7 not run | `uv run citerag eval run --golden eval/golden/candidates.jsonl --index-dir data/index_eval --offline` |
| F-EVA-7 | done | `--offline` | `uv run pytest tests/integration/test_eval_cli.py` |
| F-EVA-8 | missing | Three-run variance (SHOULD) not run | none |
| F-EVA-9 | partial | `--offline`, `--resume`, `--limit`, `--max-requests`, `--skip-judge`, `--estimate` built; `--llm-disk-cache` not built | `uv run pytest tests/integration/test_handoff_tools.py` |
| F-AGT-1 | partial | Deep crew built and tested on the fake server; E7 not run | `uv run pytest tests/integration/test_crews.py` |
| F-AGT-2 | done | Question writer plus critic, failure analyst; output `verified=false` | `uv run pytest tests/integration/test_crews.py` |
| F-AGT-3 | done | Telemetry off, limits, one LLM per agent, one log row per request | `uv run pytest tests/integration/test_crews.py` |
| F-UI-1 | done | Chat: history, image, filter, mode, cache toggle, cards, badge, usage, agent trace | `uv run pytest tests/unit/test_ui.py` |
| F-UI-2 | done | Library page | `uv run pytest tests/unit/test_ui.py` |
| F-UI-3 | done | Dashboard charts 1 to 7 | `uv run pytest tests/unit/test_ui.py tests/unit/test_cache_plots.py` |
| F-UI-4 | done | Evaluation page | `uv run pytest tests/unit/test_ui.py` |
| F-UI-5 | missing | SSE progress (SHOULD) not built | none |
| F-UI-6 | done | UI only through HTTP | `uv run pytest tests/unit/test_ui.py` |
| F-API-1 | partial | Every section 14 endpoint except `/v1/chat/stream` (SHOULD) and `/v1/transcribe` (COULD) | `uv run pytest tests/integration/test_chat_api.py` |
| F-API-2 | done | Images as base64 in JSON | `uv run pytest tests/integration/test_chat_api.py` |
| F-OPS-1 | done | `requests` and `llm_calls` tables | `uv run pytest tests/unit/test_log_store.py` |
| F-OPS-2 | done | Daily guard counts every attempt; hits still served | `uv run pytest tests/integration/test_request_guard.py` |
| F-OPS-3 | done | `/health` with versions, manifest, namespace, cache, warnings | `uv run pytest tests/integration/test_health.py` |
| F-OPS-4 | met | `docker compose build` then `up -d` on 2026-11-07: backend healthy, `/health` 200, Streamlit `/_stcore/health` 200, ports on 127.0.0.1 only, no provider key in the frontend, no `.env` in the image (WINDUP_SMOKE.md) | `docker compose up -d`, then `curl http://127.0.0.1:8000/health` |
| F-OPS-5 | done | Shadow dollars read 0 for free models | `uv run pytest tests/unit/test_pricing.py` |
| F-OPS-6 | done | Limiter, Retry-After, retries logged | `uv run pytest tests/unit/test_llm_layer.py tests/unit/test_gateway.py` |
| F-OPS-7 | done | `scripts/pick_free_models.py` | `uv run pytest tests/unit/test_pick_free_models.py` |
| NF-1 | partial | Latency logged and charted; cache-hit p95 and miss p50 not measured on a replay | `uv run pytest tests/integration/test_bench_cache.py` |
| NF-2 | partial | Requests and tokens logged and charted; the 25% reduction needs E6 | `uv run pytest tests/integration/test_request_guard.py` |
| NF-3 | done | Timeouts, gateway retries, typed errors, graceful cache and rerank failure | `uv run pytest tests/unit/test_gateway.py tests/unit/test_cache_store.py` |
| NF-4 | done | Privacy statement in the README | read README, Privacy |
| NF-5 | done | Default `pytest` has no network call; 99% line coverage on the four NF-5 modules (M8) | `uv run pytest --cov=citerag.cache --cov=citerag.rag.verify --cov=citerag.rag.context --cov=citerag.eval.metrics` |
| NF-6 | done | `uv.lock`, saved configs, `pathlib`, Python scripts only | `uv sync --locked --extra dev` |
| NF-7 | done | Boundaries test, ruff, Conventional Commits | `uv run pytest tests/unit/test_boundaries.py` |
| NF-8 | done | README with diagram, quickstart, results and limitations | the clean-clone proof in WINDUP_SMOKE.md |

## Requirements, M9 to M12 (docs/PHASE3_PRD.md)

| ID | Status | Evidence | Check (no LLM request) |
|---|---|---|---|
| F-TOOL-1 to F-TOOL-7 | done | `tools/`: three read-only tools, validation, caps, gate, typed errors, filter | `uv run pytest tests/unit/test_tools.py` |
| F-MCP-1 to F-MCP-4 | done | `citerag mcp`, stdio only, stderr logging, HTTP backend by default | `uv run pytest tests/integration/test_mcp_and_tools_api.py` |
| F-MCP-5 | done | A real subprocess completes `initialize` and `tools/list` | same file, `test_stdio_server_speaks_only_json_rpc_on_stdout` |
| F-GRF-1 to F-GRF-8 | done | `agent_graph/`: plan with code-run searches, reflect, act, limits, guard pre-check, trace, cache skipped | `uv run pytest tests/integration/test_agent_mode.py tests/unit/test_agent_routing.py` |
| F-GRF-9 | partial | Escaping and tool-argument canary checks tested on fakes; no live canary run in Agent mode | `uv run pytest tests/integration/test_agent_mode.py tests/integration/test_e8.py` |
| F-AUI-1 to F-AUI-3 | done | `mode="agent"`, trace, Chat selector, images refused | `uv run pytest tests/unit/test_ui.py tests/integration/test_agent_mode.py` |
| F-AUI-4 | missing | Dashboard split by mode (SHOULD) | none |
| F-AEV-1 | partial | E8 built and rehearsed; not run (needs the verified set) | `uv run pytest tests/integration/test_e8.py` |
| F-AEV-2 to F-AEV-4 | done | Context coverage, runner, report, estimate | `uv run pytest tests/integration/test_e8.py` |
| F-SUP-1 | done | Supervisor with researcher and writer | `uv run pytest tests/integration/test_supervisor.py` |
| F-SUP-2 | partial | Comparison configs and report built; not run | `uv run pytest tests/integration/test_supervisor.py` |
| F-DEV-1 | done | `docs/AI_ASSISTED_DEV.md` | read the file |
| NF-P3-1 | done | Boundary test | `uv run pytest tests/unit/test_boundaries.py` |
| NF-P3-2 | done | 3, 5 and 8 requests asserted | `uv run pytest tests/integration/test_agent_mode.py` |
| NF-P3-3 | partial | Agent latency not yet measured against standard | E8 |
| NF-P3-4 | done | No network in the default suite | `uv run pytest` |
| NF-P3-5 | done | Read-only tools, validation, caps, escaping, limits | `uv run pytest tests/unit/test_tools.py` |
| NF-P3-6 | done | `M9_test_guide.md` to `M12_test_guide.md`, `PROGRESS.md` | `docs/results/INDEX.md` |

## Found by this audit and fixed in the wind-up

| Finding | Fix | Commit message |
|---|---|---|
| The cache was on by default with an unvalidated threshold | Off by default; warning in the log, `/health`, the home page and `citerag cache stats` | `fix: turn the semantic cache off by default and warn on an unvalidated threshold` |
| A missing key or a mismatched index printed a stack trace | One clear message, 503 from every API route, exit code 2 in the CLI | `fix: report missing settings clearly in the API and CLI instead of a stack trace` |
| `review_golden.py` matched 800-61r2 with 800-61r3 | Identifiers with a digit stay whole | `fix: keep identifiers whole when review_golden compares terms` |
| The empty-sequence crash in the unanswerable branch | Not reproducible in 6 edge cases; now pinned by 12 regression tests | same commit's tests |
| `.claude/` and `.mcp.json` were untracked but not ignored | Ignored, not deleted | `chore: ignore local tool config and stray index and cache folders` |
| ARCHITECTURE Appendix D had drifted from `.env.example` | Synced | `docs: sync ARCHITECTURE Appendix D with .env.example` |
| `citerag mcp --local` deadlocked on Windows under a real SDK stdio client at the first search | The index and embedder load on the main thread before serving; stdout is redirected while they load | `fix: load the index before serving MCP so a stdio client cannot deadlock it` |
| `bench-cache` would run its "on" pass without a cache once the default was off | It forces the cache on for its own run folder | in the cache commit |
| No single leakage and schema check | `scripts/check_release.py` plus tests | `test: add leakage, attribution and schema release checks` |
