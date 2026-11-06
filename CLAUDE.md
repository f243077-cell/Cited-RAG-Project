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
uv run citerag ingest data/sample --captions   # opt-in: one LLM request per new image; also fills in missing captions
uv run citerag eval retrieval --rerank --skip-type image --index-dir data/index_eval   # hit@k, MRR, zero LLM requests
uv run citerag eval explain a-017 --index-dir data/index_eval   # one item: ranks per retriever, gold page text
uv run python scripts/audit_gold_pages.py --index-dir data/index_eval   # pages with the answer's terms that are not gold
uv run python scripts/review_golden.py --out eval/results/golden_review.md   # each item with its gold page text, for human review
uv run python scripts/pick_free_models.py   # list free models by capability and suggest .env lines (M0)
uv run citerag eval run --golden eval/golden/candidates.jsonl --index-dir data/index_eval --offline   # zero LLM requests
uv run citerag eval run --golden F --index-dir D --skip-judge --max-requests 16   # answers now, verdicts later
uv run citerag eval run --golden F --index-dir D --resume <run_id>   # continue a run, or add missing verdicts
uv run citerag eval compare <run_a> <run_b>   # delta table; under 3 points is noise
uv run citerag eval gate <run_id>   # MIN_TOP_SIMILARITY sweep; a proposal, it changes nothing
uv run citerag eval parts --index-dir data/index_eval   # multi_chunk: hybrid rank of each gold part, zero LLM requests
uv run citerag ingest data/sample --config configs/experiments/e1_chunk_256.yaml   # build an experiment index
uv run citerag quota                      # requests in the last 24 hours against each guard, zero LLM requests
uv run citerag logs purge --days 30       # delete old log rows
docker compose config                     # checks the compose file; the images are untested (M8)
uv run citerag cache stats                # or: cache clear
uv run python scripts/review_cache_pairs.py --label   # a human labels the candidates into eval/cache_pairs.jsonl
uv run citerag eval cache-sweep --pairs eval/cache_pairs.jsonl   # labelled pairs only, zero LLM requests, changes nothing
uv run python scripts/make_workload.py    # golden questions plus labelled paraphrases
uv run citerag eval bench-cache --workload eval/workload.jsonl --index-dir data/index_eval --estimate   # cost only
uv run citerag eval bench-cache --workload eval/workload.jsonl --index-dir data/index_eval --max-requests 30   # uses quota
uv run citerag eval gen-candidates --n 12 --index-dir data/index_eval --max-requests 8   # uses quota; add --estimate for the cost only
uv run citerag eval analyze-failures <run_id>   # 1 LLM request, none if nothing failed
uv run citerag eval run --golden F --config configs/experiments/e7_deep.yaml --max-requests 30   # at least 3 requests per item
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
   limiter slots, use one `LLM` object per agent, and write one `llm_calls` row per request from the crew's token usage.
3. **Pydantic at every boundary**, including LLM outputs (`with_structured_output`). Model-facing schemas stay minimal
   (`LLMCitedAnswer`); code adds the verified fields (`CitedAnswer`).
4. **Verify, never trust.** Citations are checked against passages (`verify_citations`); cache hits pass `guard_ok`;
   structured output is validated, retried once, then raises `StructuredOutputError`.
5. **Pipeline = plain async functions** (`rag/pipeline.py`). No LangGraph, no hidden agent loops in the default path.
6. **Documents are untrusted data.** Passages go inside the escaped `<documents>` block via `render_passages`; never
   f-string document text into a prompt or a CrewAI task template (pass it via `kickoff(inputs=...)`).
7. **Only the backend opens Chroma.** The UI imports `schemas`, `analytics.plots` and one exception, `config.UISettings`, and
   talks to the API over HTTP. `UISettings` holds only `API_BASE_URL` and `CITERAG_API_KEY`, so the UI never loads provider keys.
8. **Prompts live in `src/citerag/prompts/*.md`** with a `version:` header; the version is logged per request.
9. **Config via `config.py` (pydantic-settings)**; no `os.environ` reads elsewhere. `configs/pricing.yaml` holds only *shadow*
   prices. Free models only (owner decision, 2026-11-06): no paid reference model, so shadow dollars read 0.
10. Cache stores only answerable, fully verified, standard-mode, no-image answers, and is scoped by `namespace`. Fully verified
    means every citation is valid **and** verification raised no problem: an unbacked `[n]` or a dropped citation also blocks it.

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
- langchain-openai sends `max_tokens=` as `max_completion_tokens`; the real service accepts it. `response_metadata["model_name"]` is
  the serving model (verified live). Output tokens include hidden reasoning (64 of 78 on one short answer).
- **A 429 can arrive inside an HTTP 200 body** as `ValueError({'code': 429, ...})`: `classify_error` says `provider_body` with
  `http_status=429`. The gateway must treat that as a rate limit, not a plain retry.
- **CrewAI prompts "Share this execution trace?" on first run** despite telemetry off (20 s block). Use `Crew(tracing=False)`.
- `GET /api/v1/key` read `used: 3` after 5 real attempts. Trust the local `llm_calls` count.
- **LlamaIndex overwrites `doc_id` in vector-store metadata** with `node.ref_doc_id or "None"`. On a bare `TextNode` a `doc_id` filter
  returns nothing, silently. Set `node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)`.
- **LlamaIndex embeds node metadata by default.** Set `excluded_embed_metadata_keys` and `excluded_llm_metadata_keys` to all metadata
  keys, or an indexed sentence queried with itself scores 0.87 instead of 1.0, which skews `MIN_TOP_SIMILARITY`.
- **`HuggingFaceEmbedding` adds a bge query prefix on its own** (query and text embeddings of one sentence: cosine 0.951).
  `query_instruction=""` does not stop it because the code uses `or`; pass `query_instruction=" ", text_instruction=" "`.
- The embedding model lands in llama_index's cache (`platformdirs.user_cache_dir("llama_index")`), not `~/.cache/huggingface`.
- OpenRouter's free list has more than chat models: `openrouter/free` (router), `google/lyria-*` (audio output) and `stealth/*` (no
  `:free` suffix) all show price 0. `scripts/pick_free_models.py` filters them.
- **A whole-document gold source flatters its rank.** `m-004` part 0 ranks 2 on a chunk with no installer text; the chunk that holds it ranks 4.
- **Parts follow pages, but content does not.** SSDF p26 holds RV.1.3 and RV.2, and p27 holds RV.2.2 and RV.3, so `m-005` splits by page only.
- `review_golden.py` compares terms by their first 5 characters, so 800-61r2 and 800-61r3 look identical in a marked sentence.
- **The relevance gate cannot catch on-topic questions the documents do not answer.** Near-miss unanswerables scored cosine 0.65 to
  0.77 against 0.70 to 0.88 for answerable items, so refusing them is the model's job. `MIN_TOP_SIMILARITY` is provisionally 0.55 and is recalibrated after the verified set.
- Golden `page` is the stored PDF position from 1 (pypdf), never the printed number; `doc` must equal the stored `doc_name` exactly.
- About 10 of the 29 images that pass the size filter sit on cover pages. Look at this in M5: each caption is one LLM request.
- **BM25's tokenizer drops one-character tokens**, so `PS.1` becomes `ps` and identifiers that differ by a digit look the same.
- **A page-level hit may not hold the answer.** Golden sources are pages, so a chunk on the right page counts even if it misses the
  answer. Audit gold pages with `scripts/audit_gold_pages.py`: an exact phrase missed page 3 for `a-007`.
- **BM25 pads its results with score-0 chunks** that match no query term, and rank fusion rewards those ranks. Keep the
  `_MatchingBM25` wrapper in `index/retriever.py`.
- **A gate refusal is not free:** below `MIN_TOP_SIMILARITY` the pipeline spends one rewrite request before refusing. At 0.55 an
  off-topic question such as "What is the capital of France?" (cosine 0.411) now takes that path; at 0.30 the model refused it.
- Streamlit logs two 404s for `<page>/_stcore/health` and `host-config` on a direct page URL such as `/Chat`. They are harmless.
- **`generate` can fail with an unclassified exception** (`error_kind` `unknown`, no HTTP status, 22 to 35 s in): it happened twice on
  "what is the risk of using AI" and passed on the next try, so the cause is unknown. `unknown` does not retry. `llm_calls` now records
  `unknown:<ExceptionType>`, so read that on the next failure before adding a retry.
- `eval/results` also holds older `*-retrieval` folders with another shape; `list_runs` leaves them out.
- Not yet verified against real OpenRouter: real rate-limit headers, `data_collection` on the real service.
- **All four roles run on OpenRouter; the Gemini code stays but no role uses it.** A fresh clone needs no Gemini key.
- **`gemini-3.8-flash` produced no output.** AI Studio showed 504 and 503 with no output tokens (owner's reading). The client saw only
  timeouts with no HTTP status: structured output at 60 s and 180 s, plain text and an image at 90 s. It is not `response_format`:
  `GET /models` (61 models, 1.6 s) and a POST naming an unknown model (404 in 0.5 s) both work, so network, key and URL are fine.
  Structured output, images, tool calling and `temperature` on Gemini are all unverified. The native `generateContent` path was not tried.
- **Failed Gemini attempts count against the daily limit.** The owner's AI Studio limits for `gemini-3.8-flash` are 5 per minute and 20 per
  day, so `GEMINI_MAX_REQUESTS_PER_DAY` is 18. `GEMINI_TIMEOUT_S` is 45, and a Gemini timeout or 504 is never retried; a 503 still is.
- `gemini-2.5-flash` returned 404 "no longer available to new users" in 0.9 s, though it is listed by the API and on the pricing page.
- A caption is capped at 60 s in total, retries included (`CAPTION_TIMEOUT_S`). The first caption that fails for quota or provider
  reasons stops the rest for that document; the text still goes in, and a later ingest with captions on fills in the images.
- **Document captions are opt-in:** the Library toggle, `captions=true` on `POST /v1/documents`, or `citerag ingest --captions`, all off by
  default because each new image is one free-tier request. A stored caption is free and is used even when the toggle is off.
- `MODEL_STRONG` must accept images: `generate` sends the attached image to the strong role, not the vision role.
- **A caption copies an injection into the index by design.** `whiteboard_note.png` put `CANARY-9034` into `visible_text` and `key_facts`,
  so only the untrusted-data rule in the answer prompt keeps it out of the answer. Keep `i-005` in every run.
- Indexes built before M5 hold no image chunks. `citerag ingest PATH --captions` adds them without re-ingesting the text.
- pypdf needs no direct Pillow import here: `image.image` gives size and PNG conversion, JPEG 2000 included. Pillow stays a transitive dependency.
- **Being listed does not mean a model is usable.** Trust one real request, not the listing or a pricing page.
- **The judge is a stopgap: it runs on `nvidia/nemotron-3.5-super-vl:free`, the answerer's model**, so it grades its own work. Others
  failed on 2026-11-06: `xingchen-agi/xing4.0-29b-a4b:free` (error inside HTTP 200, after an earlier upstream 429), `google/gemma-4-31b-it:free`
  (404 with `response_format`, then 429 with tools). nemotron accepted `JudgeVerdict` and `temperature` 0 (19.8 s). Prefer another model.
- With `LLM_MAX_RETRIES=0` the json_schema to tools switch never runs, so a one-shot test of a model can fail for that reason alone.
- **A hanging OpenRouter model is expensive:** with the defaults one call waits about 186 s (3 attempts of 60 s plus backoff) and takes 3
  slots of the daily guard. Long runs must stop on `llm_unavailable` instead of moving to the next item.
- Gemini limits are per project and per model, requests per day reset at midnight Pacific, and free-tier numbers appear only in
  AI Studio. Google marks free-tier content as used to improve its products. The guard counts a rolling 24 hours, not Google's reset.
- **`SemanticCache` defines `__len__`, so an empty cache is falsy.** `cache or NoOpCache()` silently replaced it, and uploads stopped invalidating. Compare with `None`.
- **The placeholder `CACHE_THRESHOLD` of 0.92 is not safe.** On the 80 candidate pairs with the drafter's proposed labels (a smoke run, not a result), 14 of 43 near misses reached 0.92 and passed the guard. bge-small puts "How many hours..." and "How many days..." at 0.97.
- **`guard_ok` sees digits and a short word list only:** "first" vs "second", "days" vs "hours" and "leave out" all pass it. Only the threshold stops those.
- A request with `doc_ids` bypasses the cache: a stored answer may cite documents outside the filter.
- `eval/results/cache_sweep/sweep.json` is what the Dashboard shows. Delete it after any trial sweep, or trial numbers look real.
- **Deep mode costs at least 3 requests per question** (1 for standard), and 5 when the rewrite and the citation retry both run. It never reads or writes the cache.
- **The guard counts `llm_calls` rows, so a crew writes one row per request**, not one per crew as first designed. A failed crew writes every reserved slot.
- **CrewAI retries a failed task twice by default, outside the guard** (`max_retry_limit`). `agent_limits()` sets it to 0.
- **CrewAI retries a 429 twice inside one LLM call** (`llms/retry.py`, a constant, up to 8 s backoff), so a rate-limited task spends 3 requests and ignores `Retry-After`. Failed attempts are not in `successful_requests`; `runtime.py` adds 2 rows on a rate-limit failure.
- **Importing `crewai` runs `load_dotenv()`**, so `.env` lands in `os.environ` for the whole process and beats `_env_file=None` in tests. `tests/conftest.py` clears every setting name per test.
- **CrewAI exception chains can be circular**: never walk `__context__` without a seen set. Mine hung here.
- **CrewAI parses strictly.** One non-JSON reply raises `RuntimeError` wrapping a `ValidationError`; `llm/parse.py` is not used, so a fenced reply can pass in standard mode and fail in a crew. Not yet seen on a real model: no crew has run live.
- **CrewAI assumes 8,192 tokens for an unknown model** and summarises above it, which would rewrite passages. `respect_context_window=False` turns that off.
- CrewAI takes `max_retries`, `default_headers`, `extra_body`, `base_url` and `api_key` on `LLM`, so the key need not sit in the environment.
- **NLTK refuses hard-linked data files, and uv hard-links by default** when its cache shares a drive with the project. Chunking then crashes with `PermissionError: refusing multiply-linked file`. `[tool.uv] link-mode = "copy"` fixes it. This repo on D: never showed it; a fresh clone on C: did.
- **The Dockerfile and compose file are untested.** The CLI is installed, but the daemon was not running in M8, so only `docker compose config` ran. The image will pull CUDA-enabled PyTorch from the lock file, so expect it to be large.
- Gemini's OpenAI endpoint returns errors as a list, `[{'error': {...}}]`, not an object; the SDK status code is still right.

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
