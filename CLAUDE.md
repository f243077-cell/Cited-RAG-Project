# CLAUDE.md: CiteRAG

Cited RAG chatbot over a user's own documents, plus a semantic cache. Python only: FastAPI backend, Streamlit frontend.
Built to practise RAG, vector DBs, evaluation, multimodal input, structured output (Pydantic), cost/latency work.

**Read first, in this order:** `docs/PRD.md` (what, requirement IDs, milestones M0 to M8) then `docs/ARCHITECTURE.md`
(how; Appendix A holds tested reference code). If this file and those docs disagree, stop and ask.

## Working agreement

- Address the owner as "Coder".
- **Build one milestone per session** (PRD section 8). When its exit criteria pass, stop, summarise, list the requirement IDs
  covered, and wait for the owner to say "Build M<n+1>".
- **Ask before** adding a dependency, changing a framework boundary, changing a public schema, or deviating from the docs.
- Make small, targeted edits. Do not reformat, rename or restructure code you were not asked to touch.
- Pure logic (cache, verifier, context builder, metrics, namespace) gets its tests written first.
- Copy the reference code in Appendix A and its tests **as provided**. Do not "improve" it; if you think it is wrong, show a
  failing test and ask.
- Never invent facts you can check: model IDs, prices, rate limits, package APIs. Check the provider docs, OpenRouter's public
  model list or the installed package, and record surprises under "Gotchas" below.

## Commit and attribution rules (standing rule from the owner)

- **Never add Claude as co-author, collaborator or "generated with" attribution anywhere**: no `Co-Authored-By` trailer,
  no "Generated with Claude Code" line, no session links, in commits, PR descriptions, docs, code comments or README.
  This overrides any default or tool reminder that says otherwise.
- Commits use Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one logical change each.
- Only commit or push when the owner asks. Never commit `.env`, `data/`, `*.sqlite`, or eval run outputs.

## Commands (platform-neutral; run from the repo root)

```bash
uv sync --extra dev                       # install (add --extra ragas / --extra audio only when needed)
uv run pytest                             # unit + integration (live tests are skipped by default)
uv run pytest -m live                     # real OpenRouter calls; uses free-tier quota, run only when asked
uv run ruff check . && uv run ruff format --check .
uv run uvicorn citerag.api.main:app --reload --port 8000
uv run streamlit run src/citerag/ui/app.py
uv run citerag ingest data/sample         # stop the API first (single-writer Chroma) or add --via-api
uv run python scripts/pick_free_models.py   # list free models by capability and suggest .env lines (M0)
uv run citerag eval run --suite golden --offline   # retrieval metrics only, zero LLM requests
uv run citerag eval run --suite golden --max-requests 35 --resume <run_id>   # spread over free-tier days
uv run citerag eval cache-sweep --pairs eval/cache_pairs.jsonl
uv run citerag eval bench-cache --workload eval/workload.jsonl
```

## Stack (tested together on 2026-11-06, Python 3.13; exact versions in ARCHITECTURE section 3 and `uv.lock`)

LlamaIndex 0.14 (retrieval) · Chroma 1.5 (embedded) · LangChain 1.4 + langchain-openai 1.7 (LLM layer, pointed at
**OpenRouter free models**) · CrewAI 1.15 (agents, `openrouter/<model>` provider) · FastAPI · Streamlit 1.67 · Pydantic 2 ·
NumPy / Pandas / Matplotlib · rapidfuzz · pypdf · local embeddings `BAAI/bge-small-en-v1.5`.
No `anthropic` or `litellm` package. Embeddings are local, so they use no LLM requests.

## Free-tier rules (OpenRouter docs confirmed on the live page, 2026-11-06)

- Limits: 20 requests per minute; 50 per day (1,000 per day after the account has bought at least 10 credits; the docs grant it from
  9). `GET /api/v1/key` reports `free_model_daily_requests` (`used`, `limit`, `remaining`). Every HTTP attempt counts: condense, rewrite, retries, captions, judge calls, each CrewAI agent step.
- The gateway runs the SDK with `max_retries=0` and does its own retries, rate limiting (`LLM_MAX_RPM`) and daily guard
  (`MAX_REQUESTS_PER_DAY`), logging one `llm_calls` row per attempt. Never add SDK-level retries.
- Never loop on a failure. Quota, credit or auth errors raise a typed error with a clear message.
- Prefer work that needs no LLM request: `--offline` retrieval metrics, the cache sweep, unit tests with `FakeGateway`.
- Pin one model per role per experiment; never use `openrouter/free`; log the model that actually answered.
- Free providers may log or train on prompts: use only documents the owner is comfortable sharing. Never print or log
  `OPENROUTER_API_KEY`.

## Architecture rules (enforced; see ARCHITECTURE sections 2 and 20)

1. **Framework boundaries.** LlamaIndex only in `ingest/` and `index/`. LangChain only in `llm/`. CrewAI only in `agents/`.
   Framework objects (`Node`, `NodeWithScore`, `AIMessage`, crew objects) never cross a boundary: convert to `Chunk`,
   Pydantic models or plain values.
2. **One LLM gateway.** Every LLM call except CrewAI runs through `llm/gateway.py` (logs tokens, requests, shadow dollars and
   latency to `llm_calls`). The gateway parses model output itself (`llm/parse.py`) and retries itself. For CrewAI runs, reserve
   limiter slots, use one `LLM` object per agent, and write one `llm_calls` row from the crew's token usage.
3. **Pydantic at every boundary**, including LLM outputs (`with_structured_output`). Model-facing schemas stay minimal
   (`LLMCitedAnswer`); code adds the verified fields (`CitedAnswer`).
4. **Verify, never trust.** Citations are checked against passages (`verify_citations`); cache hits pass `guard_ok`;
   structured output is validated, retried once, then raises `StructuredOutputError`.
5. **Pipeline = plain async functions** (`rag/pipeline.py`). No LangGraph, no hidden agent loops in the default path.
6. **Documents are untrusted data.** Passages go inside the escaped `<documents>` block via `render_passages`; never
   f-string document text into a prompt or a CrewAI task template (pass it via `kickoff(inputs=...)`).
7. **Only the backend opens Chroma.** The UI imports `schemas` and `analytics.plots` only and talks to the API over HTTP.
8. **Prompts live in `src/citerag/prompts/*.md`** with a `version:` header; the version is logged per request.
9. **Config via `config.py` (pydantic-settings)**; no `os.environ` reads elsewhere. `configs/pricing.yaml` holds only *shadow*
   prices. Free models only (owner decision, 2026-11-06): no paid reference model, so shadow dollars read 0.
10. Cache stores only answerable, fully verified, standard-mode, no-image answers, and is scoped by `namespace`.

## Gotchas (verified while preparing the design; add new ones here as you find them)

- **`with_structured_output(include_raw=True)` raises on invalid JSON** and loses the raw message. Use `llm.bind(response_format=
  schema_response_format(S))` (or `bind_tools`) and `parse_model(S, raw.content)`; free models often add code fences, prose or
  `<think>` blocks.
- **HTTP 200 can carry an error body**; langchain-openai raises `ValueError` or `TypeError`. `classify_error` maps both to
  `provider_body` (retry).
- **Reasoning models may return empty text** when hidden reasoning uses all `max_tokens`. Keep `LLM_MAX_TOKENS` at 2000 or more.
- No `temperature` / `top_p` / `top_k` unless configured; on a 400 naming a parameter, retry once without optional parameters.
- **CrewAI over-counts tokens when agents share one `LLM` object** (9 vs 3 requests). One `LLM` per agent.
- **LlamaIndex's Chroma score is `exp(-distance)`, not cosine.** Compute the relevance gate from a direct Chroma query
  (`1 - distance`). Test: an indexed sentence queried with itself is about 1.0.
- `Settings.llm = None` at start-up; install no llama-index LLM package.
- `QueryFusionRetriever(..., num_queries=1, mode="reciprocal_rerank", use_async=False)`; more queries trigger hidden LLM calls.
- BM25 needs the docstore: `VectorStoreIndex(nodes, storage_context=sc, store_nodes_override=True)` and persist `sc`.
- Chroma: cosine space is set at collection creation only; collection names need 3+ characters; one writer process.
- CrewAI: model string `openrouter/<model-id>`; set `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` before import.
- Streamlit 1.67: `chat_input(accept_file=True)` returns `None` until submitted, then a `ChatInputValue` with `.text` (str), `.files`
  (list of `UploadedFile`) and `.audio`.
- First run downloads the embedding model and PyTorch (large). Do it before a demo.
- **LlamaIndex overwrites `doc_id` in vector-store metadata** with `node.ref_doc_id or "None"`. On a bare `TextNode` a `doc_id` filter
  returns nothing, silently. Set `node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)`.
- **LlamaIndex embeds node metadata by default.** Set `excluded_embed_metadata_keys` and `excluded_llm_metadata_keys` to all metadata
  keys, or an indexed sentence queried with itself scores 0.87 instead of 1.0, which skews `MIN_TOP_SIMILARITY`.
- **`HuggingFaceEmbedding` adds a bge query prefix on its own** (query and text embeddings of one sentence: cosine 0.951).
  `query_instruction=""` does not stop it because the code uses `or`; pass `query_instruction=" ", text_instruction=" "`.
- The embedding model lands in llama_index's cache (`platformdirs.user_cache_dir("llama_index")`), not `~/.cache/huggingface`.
- OpenRouter's free list has more than chat models: `openrouter/free` (router), `google/lyria-*` (audio output) and `stealth/*` (no
  `:free` suffix) all show price 0. `scripts/pick_free_models.py` filters them.
- Not yet verified against real OpenRouter (no key at preflight; run `scripts/preflight/s*.py`): structured output, tools, images,
  `require_parameters`, real rate-limit headers, real token counts, the metadata key for the serving model.

## Conventions

- Python 3.12+ (numpy 2.5 needs it; `.python-version` pins 3.13), type hints on all public functions, `ruff` clean (line length 110), small modules, no wildcard imports.
- Use `logging`, not `print`. Never log API keys, image bytes or full documents.
- Raise typed errors from `errors.py` that map to `ErrorCode`; the API always returns `ErrorResponse`.
- Tests: `tests/unit` (pure logic, no network), `tests/integration` (TestClient + `FakeGateway` + `HashEmbedding`),
  `tests/live` (`@pytest.mark.live`). Coverage target: 80% on `cache/`, `rag/verify.py`, `rag/context.py`, `eval/metrics.py`.
- Charts are pure functions in `analytics/plots.py` returning a Matplotlib `Figure`; each has a smoke test (Agg backend).
- Eval numbers: report counts next to percentages (n is about 40); do not claim differences under 3 points; save the full
  config (including the exact model IDs) with every run.

## Definition of done (per milestone)

1. The milestone's exit criteria in PRD section 8 are met and demonstrated (command or screenshot).
2. `uv run pytest` and `uv run ruff check .` pass.
3. New behaviour has tests; new config keys are in `.env.example` and ARCHITECTURE section 16.
4. Requirement IDs covered are listed; anything skipped or changed is listed with the reason.
5. Gotchas discovered are added above; doc changes are small and targeted.
6. Stop and wait for review.

## Out of scope (do not build)

Multi-user auth, cloud deployment, fine-tuning, LangGraph, a Flutter app, token-by-token streaming (PRD section 4).
