# CiteRAG: Product Requirements Document

| | |
|---|---|
| **Product** | CiteRAG, a cited RAG chatbot with semantic caching, multimodal input and an evaluation harness |
| **Owner** | Tanzeel |
| **Version / date** | 1.1, 2026-11-06 (LLM provider: OpenRouter free models) |
| **Status** | Ready to build |
| **Companion docs** | `docs/ARCHITECTURE.md` (how it is built), `CLAUDE.md` (rules for Claude Code) |

Priority tags: **[MUST]** required for the project to count as done. **[SHOULD]** build unless it blocks a MUST. **[COULD]** only if time remains.

---

## 1. Summary

CiteRAG answers questions about a user's own documents (PDF, Markdown, text, and the images inside them). Every answer carries citations that point to the exact page and quote the answer came from, and the app *verifies* each quote against the source text instead of trusting the model. When the documents do not contain the answer, it says so. Questions that are repeated or paraphrased are served from a **semantic cache** at near-zero cost and latency. A built-in **evaluation harness and dashboard** measure answer quality, citation validity, LLM usage and speed, so every claim about the system is backed by a number.

It is both the **main project** of Phase 2 of the GenAI roadmap ("Cited RAG Chatbot: RAG + Vector DB + Evaluation") and its **recommended improvement project** ("add caching: cost and performance, after adding Semantic Caching").

## 2. Goals

### 2.1 Learning goals (what the build must exercise)

| Roadmap item | Where it shows up in CiteRAG |
|---|---|
| Phase 0: Python, NumPy, Pandas, Pydantic, FastAPI, async, SQLite, Docker, EDA plots | Whole code base. NumPy for cache search and metrics, Pandas and Matplotlib for the dashboard, Pydantic for every contract, FastAPI for the backend, SQLite for logs and cache, Docker for the final run |
| 0.3 Vectors, cosine similarity | Semantic cache and threshold analysis |
| 1.2 Choosing a model per task (cost, quality, latency) | Model roles: fast, strong, vision, judge (ARCHITECTURE section 11) |
| 1.3 Prompt engineering, structured output, prompt-injection defence, prompt versioning | Versioned prompts, `CitedAnswer`, injection tests with canary strings |
| 2.1 Context engineering | Token-budgeted context builder, instruction/data separation |
| 2.2 Embeddings, vector DB (Chroma) | Local embeddings plus persistent Chroma |
| 2.3 RAG: chunking, hybrid search, re-ranking, evaluation, failure modes, semantic caching | Ingestion, hybrid retrieval, rerank, corrective retry, golden-set evaluation, semantic cache |
| 2.4 Structured output libraries | Pydantic schemas with LangChain structured output, plus validation and retry |
| 2.5 Orchestration (CrewAI) | Deep-answer crew and evaluation crews |
| 2.6 Multimodal | Image questions, image captions indexed from documents, optional voice |
| 2.7 Memory | Short-term conversation memory in SQLite |
| Phase 6 awareness | Citation UI, progress events, API shaped for a later Flutter client |

### 2.2 Product goals

| ID | Goal | Measured by (section 7) |
|---|---|---|
| G1 | Answers are grounded in the documents | Faithfulness, citation validity |
| G2 | The system knows what it does not know | Correct-refusal rate, injection leaks |
| G3 | Retrieval finds the right pages | hit@k, MRR |
| G4 | Repeated questions are cheap and fast, without wrong answers | Cache hit rate, false-hit rate, LLM-request, token and latency reduction |
| G5 | Quality, cost and speed are visible | Dashboard, saved experiment reports |
| G6 | A newcomer can run it in 15 minutes | Quickstart in README |

## 3. Users and scenarios

**P1: Document asker.** A student or professional with private PDFs who wants fast, trustworthy answers and wants to see *where* each answer comes from.
**P2: Builder / reviewer.** Tanzeel (and later an interviewer) who wants to see quality, LLM-usage and latency numbers and walk through the design end to end.

| ID | User story | Done when |
|---|---|---|
| US-1 | As P1, I upload PDFs and Markdown files and see them listed with page and chunk counts. | F-ING-1..2, F-UI-2 |
| US-2 | As P1, I ask a question and get a short answer with clickable citations showing page and quote. | F-ANS-1..2, F-UI-1 |
| US-3 | As P1, I ask something the documents do not cover and the app says so instead of guessing. | F-ANS-3 |
| US-4 | As P1, I ask a follow-up ("and what about shipping?") and it understands the context. | F-CON-1..2 |
| US-5 | As P1, I attach a screenshot or chart image to my question and get an answer grounded in the documents. | F-MM-1..2 |
| US-6 | As P1, I see a chart or diagram from a PDF cited as a source when it answers my question. | F-MM-3..4 |
| US-7 | As P1, I ask the same question twice in different words and the second answer is instant, and labelled as cached. | F-CAC-1..6 |
| US-8 | As P2, I run one command and get quality, citation, refusal and injection scores for the current configuration. | F-EVA-1..5 |
| US-9 | As P2, I open a dashboard showing cache hit rate, latency, LLM requests and tokens saved, and experiment results. | F-UI-3..4, F-CAC-7 |
| US-10 | As P2, I switch on Deep mode and see whether a three-agent crew improves faithfulness and what it costs. | F-AGT-1 |

## 4. Scope

### 4.1 In scope

Everything in section 5 tagged MUST or SHOULD, plus COULD items if time allows.

### 4.2 Out of scope (do not build, do not "prepare" for)

| Not building | Why |
|---|---|
| Agents with tools, MCP, LangGraph orchestration | Phase 3. CrewAI here is used only for Deep mode and evaluation. The pipeline is plain functions so LangGraph can wrap it later |
| Fine-tuning, model hosting | Phase 4 |
| Cloud deployment, managed platforms, full guardrail products, CI/CD pipelines, governance | Phase 5 |
| Flutter client | Phase 6 (the API is designed for it, see ARCHITECTURE section 14) |
| User accounts, multi-tenancy, per-user document spaces | Single-user local app |
| OCR for scanned PDFs, Graph RAG, long-term memory, web search | Not needed for the learning goals |
| Token-by-token streaming of answers | Structured output arrives as one object (ADR-8). Progress events are provided instead |

## 5. Functional requirements

### 5.1 Ingestion

- **F-ING-1 [MUST]** Accept PDF, Markdown, TXT, and standalone PNG/JPEG via UI upload, API upload, and CLI (`citerag ingest <folder>`). Reject other types, files over `MAX_UPLOAD_MB` (default 25) and empty files with a typed error. *Accept:* a `.exe` renamed to `.pdf` is rejected (check the `%PDF` magic bytes, not the extension); a 26 MB file returns `file_too_large`.
- **F-ING-2 [MUST]** `doc_id` is the first 16 hex characters of the SHA-256 of the file bytes. Re-ingesting identical bytes is a no-op (`status=unchanged`). A changed file with the same name replaces the old chunks (`status=replaced`). *Accept:* ingest twice, chunk count unchanged; edit and ingest, old chunks gone.
- **F-ING-3 [MUST]** Chunk with LlamaIndex `SentenceSplitter`; `CHUNK_SIZE` / `CHUNK_OVERLAP` configurable (defaults 512 / 64 tokens, a starting point that experiment E1 will test). Each chunk keeps metadata: `doc_id`, `doc_name`, `page` (PDF), `chunk_index`, `kind` (`text` or `image`).
- **F-ING-4 [MUST]** Embed with the local model `BAAI/bge-small-en-v1.5` (384 dimensions, normalised). The embedding model name and chunk settings are stored in an index manifest. Querying an index built with different settings fails with `index_mismatch` instead of returning garbage.
- **F-ING-5 [MUST]** The ingest result is an `IngestReport` (documents, chunks, images found / captioned / skipped, caption requests and tokens, duration).
- **F-ING-6 [MUST]** Ingestion makes no LLM call except image captioning (F-MM-3).
- **F-ING-7 [SHOULD]** DOCX support.

### 5.2 Retrieval

- **F-RET-1 [MUST]** Hybrid search: vector similarity (Chroma, cosine) plus BM25 keyword search, fused with reciprocal-rank fusion. Retrieve `TOP_K_RETRIEVE` (default 20), pass `TOP_K_CONTEXT` (default 5) to the model.
- **F-RET-2 [MUST]** Optional restriction to selected documents (`doc_ids`).
- **F-RET-3 [SHOULD]** Cross-encoder rerank of the retrieved set (`cross-encoder/ms-marco-MiniLM-L-6-v2`), switchable by config. Default on or off is decided by experiment E3 (OPEN-2).
- **F-RET-4 [MUST]** Relevance gate: if the best vector cosine similarity (a direct Chroma query, see ARCHITECTURE section 8) is below `MIN_TOP_SIMILARITY`, skip generation and refuse (this saves cost). The default is a conservative placeholder that M4 must calibrate against the unanswerable questions.
- **F-RET-5 [MUST]** Corrective retry, at most once per cause: (a) weak retrieval triggers one query rewrite (fast model) and one new retrieval; (b) a failed citation check (F-ANS-4) triggers one regeneration with the failure reason in the prompt. Worst case is four LLM calls per request (condense, rewrite, generate, regenerate).

### 5.3 Answering and citations

- **F-ANS-1 [MUST]** The model returns an `LLMCitedAnswer` (answer text with `[n]` markers, `answerable`, citations with marker and quote, confidence). Code turns it into the API's `CitedAnswer` by adding chunk, document, page and a `valid` flag. The model is never asked for page numbers or document names.
- **F-ANS-2 [MUST]** Every answer with `answerable=true` has at least one citation, and every `[n]` in the text maps to a citation. Invalid citations are shown flagged in the UI (never silently dropped) and recorded in the log.
- **F-ANS-3 [MUST]** If the passages do not contain the answer, return `answerable=false`, a one-sentence refusal, and no citations. No answers from the model's own knowledge. *Accept:* at least 90% correct refusals on the golden unanswerable items.
- **F-ANS-4 [MUST]** Citation verification (see `rag/verify.py` in ARCHITECTURE): a citation is valid when its quote matches the cited passage with a fuzzy score of at least 90. An answerable response with no valid citation, or with `[n]` markers that no valid citation backs, triggers the corrective retry (F-RET-5b). If it still fails, return the answer with `confidence="low"` and a warning.
- **F-ANS-5 [SHOULD]** Code downgrades `confidence` to `low` whenever any citation is invalid.
- **F-ANS-6 [MUST]** With an empty index, return a normal `answerable=false` response ("Upload documents first") with no LLM call.

### 5.4 Context engineering and prompt safety

- **F-CTX-1 [MUST]** Context builder: number passages `[1..n]` in relevance order, drop near-duplicate chunks, enforce `CONTEXT_TOKEN_BUDGET` (default 6000, estimated as characters / 4), remove the lowest-ranked chunks first, never cut a chunk in half.
- **F-CTX-2 [MUST]** Instruction/data separation: rules live in the system prompt; retrieved text goes inside `<documents><passage id="n" source=".." page="..">...</passage></documents>` and is declared untrusted; the question sits in its own `<question>` block.
- **F-CTX-3 [MUST]** Prompts are files under `src/citerag/prompts/` with a `version:` header. The version is written to every request log row and every eval run.
- **F-SAF-1 [MUST]** Prompt-injection defence: text inside documents or image captions must not change behaviour. *Accept:* at least 5 golden `injection` items (one of them hidden in an image caption) produce zero canary leaks.
- **F-SAF-2 [MUST]** Model output is rendered as plain Markdown, never as HTML. Uploaded file names are sanitised and stored under hashed names.
- **F-SAF-3 [MUST]** The API key comes only from the environment, is never logged, and `.env` is git-ignored.

### 5.5 Conversation memory

- **F-CON-1 [MUST]** Server-side short-term memory: the last `HISTORY_TURNS` (default 6) turns per `session_id`, stored in SQLite.
- **F-CON-2 [MUST]** When history exists, the fast model rewrites the latest message into a standalone question (`StandaloneQuery`). Retrieval **and the cache** use the standalone question. With no history the call is skipped. *Accept:* "And what about shipping?" after a refund question retrieves shipping passages.

### 5.6 Multimodal

- **F-MM-1 [MUST]** The chat accepts an optional image (PNG, JPEG, WebP, at most 5 MB). A vision call returns `ImageAnalysis`; its `search_query` combined with the user text drives retrieval, and the image is also passed to the answer model together with the passages.
- **F-MM-2 [MUST]** Answers to image questions still cite document passages. The user's image is "your image", never a citable source.
- **F-MM-3 [MUST]** Ingestion extracts embedded images from PDFs (and accepts standalone images), skips icons (under 100x100 px or under 5 KB) and duplicates (SHA-256), caps at `MAX_IMAGES_PER_DOC` (default 20), captions each with the vision model as `ImageCaption`, and indexes caption plus visible text as a chunk with `kind="image"`, page and `image_path`. Captions are cached on disk by image hash so re-ingesting never pays twice.
- **F-MM-4 [MUST]** A citation to an image chunk shows the image thumbnail next to the caption in the UI.
- **F-MM-5 [COULD]** Voice question: audio to local Whisper (`faster-whisper`) to text, then the normal flow (OPEN-3).
- **F-MM-6 [MUST]** Requests that carry an image never read or write the cache.

### 5.7 Structured output (Pydantic)

- **F-STR-1 [MUST]** Every LLM call whose result feeds code goes through one function, `structured_call(schema, ...)`, which validates against a Pydantic model, retries once with the validation error fed back, then raises `StructuredOutputError`.
- **F-STR-2 [MUST]** Everything that crosses a boundary (HTTP, cache storage, eval files, log rows) is a Pydantic model or validated JSON. No bare dicts in public function signatures.
- **F-STR-3 [MUST]** Configuration is a `pydantic-settings` class; the app fails fast at start-up if required settings are missing.
- **F-STR-4 [MUST]** Model-facing schemas contain only what the model must decide. Anything code can derive (`valid`, `page`, `doc_name`) is not in them.

### 5.8 Semantic cache (the improvement project)

- **F-CAC-1 [MUST]** After the standalone question exists and before retrieval, look up the cache. A hit returns the stored `ChatResponse` (new `request_id`, `cache.hit=true`, similarity, matched query) with **no LLM call for answering**.
- **F-CAC-2 [MUST]** Similarity is cosine on the local embedding of the standalone question, compared with `CACHE_THRESHOLD` (default 0.92, a placeholder that M6 replaces with the sweep result).
- **F-CAC-3 [MUST]** False-hit guard: numbers and negation must agree between the new and cached question (`guard_ok`, ARCHITECTURE section 9).
- **F-CAC-4 [MUST]** Store only answers that are `answerable`, have at least one citation and all citations valid. Never store refusals, errors, Deep-mode results or image-based requests.
- **F-CAC-5 [MUST]** An entry is valid only for the same `namespace` (corpus + answer model + prompt version + retrieval config) and the same mode. Any namespace change invalidates older entries immediately. TTL 7 days by default; at most 10,000 entries with least-recently-used eviction.
- **F-CAC-6 [MUST]** Per request `use_cache=false` bypasses read and write (UI toggle). `citerag cache clear` and `POST /v1/cache/clear` empty it.
- **F-CAC-7 [MUST]** Every request logs hit, similarity, LLM requests and tokens saved (plus the shadow dollar value, see F-OPS-5). The dashboard shows hit rate over time, latency hit vs miss, and cumulative LLM requests and tokens with and without the cache.
- **F-CAC-8 [MUST]** `citerag eval cache-sweep`: from at least 60 labelled question pairs (at least 30 equivalent, at least 30 near-miss with changed numbers, entities or negation) produce precision, recall and false-hit-rate against threshold, a chart, and a recommended threshold with false-hit rate at most 2%.
- **F-CAC-9 [MUST]** `citerag eval bench-cache`: replay a workload with the cache off, then on. Report hit rate, p50/p95 latency, LLM requests, tokens, shadow dollars, **and answer correctness in both runs** (a cache that is fast but wrong is a failure).
- **F-CAC-10 [SHOULD]** A `CacheBackend` interface with a no-op implementation from milestone M3, so the real cache plugs in at M6 without refactoring the pipeline. A Redis backend is a [COULD].

### 5.9 Evaluation

- **F-EVA-1 [MUST]** Golden set at `eval/golden/golden.jsonl` with at least 40 items: at least 20 `answerable`, 5 `multi_chunk`, 5 `unanswerable`, 5 `injection`, 5 `image`. Sources are given as (document, page) so the set survives re-chunking. Every item has `verified: true`, meaning a human checked question, answer and sources.
- **F-EVA-2 [MUST]** Metrics per run: hit@k and MRR (before and after rerank); faithfulness and correctness from the judge; citation validity rate; citation precision (cited source is a gold source); correct-refusal rate and false-refusal rate (answerable items wrongly refused); injection leak count; latency p50/p95; tokens; dollars per question; cache hit rate.
- **F-EVA-3 [MUST]** The judge is a LangChain structured-output call returning `JudgeVerdict`. The judge model is configurable and should differ from the answer model when possible. The judge sees the same context the generator saw.
- **F-EVA-4 [SHOULD]** A Ragas run (faithfulness, answer relevancy, context precision, context recall) as a cross-check, installed through the `ragas` extra. If it cannot be made to work with the OpenRouter judge model within about one working day, keep the own judge and write down why.
- **F-EVA-5 [MUST]** Each run writes `eval/results/<run_id>/{config.json, items.jsonl, summary.json}` including git commit and prompt versions. `citerag eval compare A B` prints a delta table.
- **F-EVA-6 [MUST]** Experiments E1 to E7 (ARCHITECTURE section 13) each produce a chart (Matplotlib) and a short written conclusion under `docs/results/`.
- **F-EVA-7 [MUST]** An offline mode runs retrieval metrics with the fake LLM gateway and no network, so tests and CI need no API key.
- **F-EVA-8 [SHOULD]** Final configuration is run three times; report mean and range, because the models are not deterministic.
- **F-EVA-9 [MUST]** Evaluation fits the free-tier quota: `--offline` computes retrieval metrics with zero LLM requests; runs append item by item so `--resume RUN_ID` continues them; `--limit N` and `--max-requests N` cap them; `--llm-disk-cache` may reuse stored model responses for judge-only reruns but is never used by `bench-cache` or repeated variance runs. A request-budget table (ARCHITECTURE section 13) is checked against real numbers at M4.

### 5.10 Agents (CrewAI)

- **F-AGT-1 [MUST]** **Deep mode** (toggle): a sequential crew of three agents (Evidence Analyst, Answer Writer, Fact Checker), no tools. Retrieval happens before the crew; the crew receives the passages as input. All task outputs are Pydantic models, and the final result is the same `ChatResponse`. The Fact Checker removes or flags unsupported claims. *Accept:* on the golden answerable set, Deep-mode faithfulness is reported next to standard mode with the cost and latency multiplier.
- **F-AGT-2 [MUST]** Evaluation crews: (a) a Question Writer plus Question Critic propose candidate golden items from sampled chunks into `eval/golden/candidates.jsonl` with `verified=false` (a human must review them, because a model must not grade itself on questions it wrote unchecked); (b) a Failure Analyst groups failed items by root cause into `docs/results/failure_analysis.md`.
- **F-AGT-3 [MUST]** CrewAI telemetry is disabled; every crew run has an execution time limit and iteration cap; token usage and request count from the crew result are added to the request totals (one `LLM` object per agent, otherwise the crew result over-counts).

### 5.11 User interface (Streamlit)

- **F-UI-1 [MUST]** *Chat page.* Message history; input with optional image attachment; sidebar with document filter, Deep-mode toggle, Use-cache toggle, new chat. Each answer shows: text with `[n]` markers; one card per citation (document, page, verified or warning icon, quote, image thumbnail for image chunks); a badge when served from cache (with similarity); a usage line (tokens in/out, dollars, seconds).
- **F-UI-2 [MUST]** *Library page.* Multi-file upload, document table (name, pages, chunks, images, date), delete, ingest report.
- **F-UI-3 [MUST]** *Dashboard page.* Matplotlib charts built with Pandas / NumPy from the request log (ARCHITECTURE section 15).
- **F-UI-4 [MUST]** *Evaluation page.* Run list, compare two runs, per-item drill-down (question, answer, citations, verdict).
- **F-UI-5 [SHOULD]** Progress status while a request runs (server-sent events).
- **F-UI-6 [MUST]** The UI talks to the backend only through the HTTP API.

### 5.12 API

- **F-API-1 [MUST]** FastAPI backend with the endpoints in ARCHITECTURE section 14, OpenAPI docs at `/docs`, a uniform error body, optional `X-API-Key`, CORS limited to localhost origins.
- **F-API-2 [MUST]** Images travel as base64 in JSON so any client (Streamlit now, Flutter later) can use the same endpoint.

### 5.13 Operations

- **F-OPS-1 [MUST]** A request log (SQLite) with the fields in ARCHITECTURE section 6; no API keys and no image bytes in it. A second table logs every individual LLM call (role, model, tokens, HTTP status, attempt number, latency).
- **F-OPS-2 [MUST]** Daily request guard `MAX_REQUESTS_PER_DAY` (default 40, below the free tier's 50 per day): every HTTP attempt to the LLM provider counts, including failed ones. Once reached, new LLM calls are refused with `budget_exceeded` (HTTP 429, message names the limit and when it resets) and cache hits are still served. Raise the value after the account has bought credits (1,000 per day).
- **F-OPS-3 [MUST]** `GET /health` reports library versions, index manifest, current namespace and cache statistics.
- **F-OPS-4 [SHOULD]** Dockerfile and `docker-compose.yml` (backend, frontend, shared data volume).
- **F-OPS-5 [MUST]** Free models cost 0 dollars, so cost is reported in **LLM requests and tokens**. The project uses free models only (owner decision, 2026-11-06), so `configs/pricing.yaml` lists a free reference model at 0 and the **shadow dollar** fields read 0. They stay in the schemas so a paid reference can be added later, copied from OpenRouter's official model pricing with the date recorded; never from memory.
- **F-OPS-6 [MUST]** Client-side rate limiter in the LLM gateway: at most `LLM_MAX_RPM` requests per minute (default 15, below the free tier's 20). On HTTP 429 or 503 wait for `Retry-After` (else exponential backoff) and retry up to `LLM_MAX_RETRIES`; every attempt is logged.
- **F-OPS-7 [MUST]** `scripts/pick_free_models.py` lists the models currently available at zero price from OpenRouter's public model list, filtered by capability (image input, structured outputs / `response_format`, tool calling), and proposes a model per role. The free list changes often, so model IDs are never hard-coded in code, only in `.env`.

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NF-1 | **Speed (targets, measured on a developer laptop, excluding first-run model downloads).** Cache hit on a standalone question: backend p95 at most 300 ms. Follow-up questions pay the condense call (one LLM request). Cache miss: p50 at most 15 s with free models (they are slower and more variable than paid ones; record the real baseline in M3 and update this target) |
| NF-2 | **Usage and quota.** LLM requests, tokens and shadow dollars per question are logged and charted. The cache must cut LLM requests on the replay workload by at least 25%. The free tier allows 20 requests per minute and 50 per day (1,000 per day after the account has bought at least 10 credits); the system must stay inside the configured limits and fail with a clear message, never loop |
| NF-3 | **Reliability.** LLM timeout 60 s, gateway-owned retries (two by default) that honour `Retry-After` on 429 / 503 and back off on 5xx / timeouts. Typed errors only. Degrade gracefully: reranker failure means continue without it; a corrupt cache file means continue without the cache and log it |
| NF-4 | **Privacy.** Documents, vectors, cache and logs stay on the machine. Retrieved passages, questions and images being captioned or analysed go to OpenRouter and to the provider serving the model. Free-model providers may retain or train on prompts depending on the account's privacy settings, so use only documents you are comfortable sharing and review the settings (ARCHITECTURE section 18). State this in the README |
| NF-5 | **Testability.** Default `pytest` makes no network call. At least 80% line coverage on `cache/`, `rag/verify.py`, `rag/context.py`, `eval/metrics.py`. Tests that call the real API are marked `live` |
| NF-6 | **Reproducibility.** Dependency lock file; saved run configs; paths via `pathlib`; no bash-only scripts, so it runs on Windows, macOS and Linux |
| NF-7 | **Maintainability.** Framework boundaries from ARCHITECTURE section 2 are respected; ruff clean; type hints on public functions; Conventional Commits |
| NF-8 | **Documentation.** README with architecture diagram, quickstart, results table and charts, and an honest limitations section |

## 7. Success metrics and targets

These are **initial targets**. Milestone M4 records the real baseline; after that, update this table and note the date, rather than silently moving goals. Golden-set size is small (about 40), so one item is worth 2.5 points: report counts (for example 34/40) next to percentages and do not claim differences under 3 points.

| Goal | Metric | Initial target | Command |
|---|---|---|---|
| G1 | Faithfulness (judge) | at least 0.90 | `citerag eval run` |
| G1 | Citation validity (verifier) | at least 0.95 | `citerag eval run` |
| G1 | Answer correct (judge, "correct") | at least 0.75 | `citerag eval run` |
| G2 | Correct refusals on unanswerable items | at least 0.90 | `citerag eval run` |
| G2 | Injection canary leaks | 0 | `citerag eval run` |
| G3 | hit@5 (hybrid) | at least 0.85 | `citerag eval run` |
| G4 | False-hit rate at chosen threshold | at most 2% | `citerag eval cache-sweep` |
| G4 | Hit rate on replay workload with at least 40% paraphrased repeats | at least 30% | `citerag eval bench-cache` |
| G4 | LLM-request reduction on that workload | at least 25% | `citerag eval bench-cache` |
| G4 | Correctness with cache minus without | at least -1 point | `citerag eval bench-cache` |
| G4 | Cache-hit p95 latency (standalone questions) | at most 300 ms | dashboard / bench |
| G6 | Clone to first answer | at most 15 minutes | README quickstart |

## 8. Milestones and build order

Build in this order and **stop for review after each milestone**. Effort is a rough guide for focused working days.

| # | Milestone | Deliverables | Exit criteria | Effort |
|---|---|---|---|---|
| **M0** | Scaffold | Repo layout, `pyproject.toml`, settings, schemas, `FakeGateway`, `NoOpCache`, `/health`, copy provided reference code and tests | `uv run pytest` green; `/health` returns 200 | 0.5 |
| **M1** | Ingestion and index (text) | Loaders, chunking, embeddings, Chroma + docstore + manifest, `citerag ingest`, idempotence | Re-ingest is a no-op; changed file replaces; counts correct | 1 |
| **M2** | Retrieval | Hybrid fusion, `doc_ids` filter, rerank switch, gate, tiny golden set, hit@k script | hit@k report on a first 10-item golden set | 1 |
| **M3** | Cited answers (**main project core**) | Condense, context builder, prompts, `structured_call`, generation, verifier, corrective retry, `/v1/chat`, Streamlit Chat and Library pages | End-to-end Q&A with verified citations; refusals work; unit tests pass | 2 |
| **M4** | Evaluation | Full golden set (at least 40, human-verified), metrics, judge, optional Ragas, runner, compare, experiments E1 to E4, Evaluation page | Baseline written to `docs/results/baseline.md`; targets table updated | 2 |
| **M5** | Multimodal | Image question flow, PDF image extraction and captioning with disk cache, image citations, image golden items | US-5 and US-6 demonstrable; image items in golden set | 1.5 |
| **M6** | Semantic cache (**improvement project**) | Cache, guard, namespace, `cache-sweep`, `bench-cache`, dashboard charts | Threshold chosen from sweep; before/after report; targets met or the gap explained | 2 |
| **M7** | CrewAI | Deep mode crew, eval crews (question generation, failure analysis) | Deep vs standard comparison (E7) written up | 1.5 |
| **M8** | Polish | Dockerfile + compose, request guard checks, README with diagram and results, security checklist, optional Whisper | A fresh clone runs from the README | 1 |

Total: about 12.5 focused days. The roadmap treats M0 to M5 as the main project and M6 as the follow-up improvement; M7 and M8 complete the stack you asked for.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Three frameworks make a tangled code base | Hard boundaries: LlamaIndex only in `ingest/` and `index/`, LangChain only in `llm/`, CrewAI only in `agents/`; framework types never cross a boundary (ARCHITECTURE section 2) |
| Dependency conflicts | Dependency set was resolved and exercised before this PRD was written (ARCHITECTURE section 3); commit the lock file; do not add packages without asking |
| Free-tier quota (20 requests per minute, 50 per day) is far below what evaluation needs | Request guard and rate limiter; retrieval metrics need no LLM calls; eval runs are resumable and capped with `--max-requests`; judge and answer responses can be cached on disk for reruns; recommended one-time credit purchase raises the cap to 1,000 per day (OPEN-5) |
| Free models vary: some lack structured output or vision, may disappear, may return empty text when reasoning uses all tokens | Gateway-owned parsing with tolerant JSON extraction, json_schema then tool-calling fallback, `require_parameters`, capability-filtered model picker, one pinned model per role per experiment, record the served model in every log row (ARCHITECTURE section 11) |
| Free providers may log or train on prompts | Documented in the README; privacy settings; `OPENROUTER_DATA_COLLECTION=deny` option; do not upload private documents |
| Judge is biased or the golden set is circular | Human-verified golden set; judge model differs from answer model when possible; spot-check 10 judge verdicts by hand per run |
| Semantic cache serves a wrong answer | High threshold chosen by sweep, number/negation guard, only verified answers stored, namespace invalidation, per-request bypass, correctness measured in the benchmark |
| Prompt injection through documents or images | Untrusted-data delimiters, canary tests, no tools, schema-constrained output, no HTML rendering |
| Running out of quota mid-experiment | Request guard, caption cache, image caps, local embeddings, offline retrieval metrics, resumable eval runs |
| VLM captions hallucinate | Show the image beside its caption in the UI; mark image citations; sample-check captions in M5 |
| Chroma embedded mode allows one writer | Only the backend opens the index; eval runs in-process only while the API is stopped, or uses `--via-api` (ARCHITECTURE section 6) |
| Sample documents carry licence limits | Use only redistributable documents, or git-ignore `data/sample/` and ship a fetch script |

## 10. Open decisions (defaults apply until Tanzeel decides)

| ID | Question | Default |
|---|---|---|
| OPEN-1 | Which documents form the demo corpus and golden set? | A set of 5 to 8 public, redistributable documents (about 100 to 200 pages) that includes at least two PDFs with figures or screenshots and one Markdown file. Put them in `data/sample/` (git-ignored if licensing is unclear) and add `scripts/fetch_sample.py` |
| OPEN-2 | Rerank on by default? | Decide from experiment E3 (accept if hit@5 improves by at least 3 points and p50 latency grows by under 300 ms) |
| OPEN-3 | Include Whisper voice input (F-MM-5)? | Out until M8; add only if everything else is done |
| OPEN-4 | Reference prices for shadow cost | Decided 2026-11-06: free models only, no paid reference model. `configs/pricing.yaml` holds a free model at 0, so shadow dollars read 0 |
| OPEN-5 | Stay on 50 requests per day or buy credits? | Strongly recommended: a one-time purchase of at least 10 credits raises the cap to 1,000 requests per day (credits are only spent if you call paid models; free models stay free). Without it, a full golden-set run takes several days (see ARCHITECTURE section 13) |

## 11. Glossary

- **Chunk**: a passage-sized piece of a document; the unit that is embedded and retrieved.
- **Hybrid search**: combining meaning-based (vector) and keyword (BM25) search.
- **RRF (reciprocal-rank fusion)**: merges ranked lists by summing 1/(k + rank) per list.
- **Faithfulness / groundedness**: every claim in the answer is supported by the retrieved text.
- **Standalone question**: the user's message rewritten so it needs no chat history.
- **Namespace**: fingerprint of everything that can change an answer (corpus, answer model, prompt version, retrieval config). The cache only serves entries from the current namespace.
- **False hit**: the cache serves an answer for a question that is not actually equivalent. The dangerous cache error.
- **Golden set**: human-verified questions with known answers and sources, used to score the system.

## 12. How to use these documents with Claude Code

1. Create the repository and place `CLAUDE.md` at the root and `PRD.md`, `ARCHITECTURE.md` in `docs/`. (The starter kit also contains tested reference code under `src/` and `tests/`.)
2. Start Claude Code in the repository. Recommended: paste the whole of `docs/PREFLIGHT_PROMPT.md`; it verifies what the design could not verify offline (real OpenRouter calls, model picks, shadow pricing, sample corpus), fixes the docs, builds M0 and pushes a branch. If you only want M0, give it this prompt instead:

> Read `CLAUDE.md`, `docs/PRD.md` and `docs/ARCHITECTURE.md` completely. Then build **Milestone M0 only** from section 8 of the PRD. Copy the reference code and tests exactly as provided, make `uv run pytest` pass, and stop. Show me what you built and which PRD requirement IDs it covers.

3. Review each milestone, then say "Build M1" and so on. Ask Claude Code to update the targets table after M4 and to write the results sections as it goes.
