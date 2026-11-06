# Preflight report

Date: 2026-11-06. Machine: Windows 10 Pro (10.0.19045), Git Bash and PowerShell 5.1, uv 0.12.7, project Python 3.13.13.

**LLM requests used: 0 of 20.** No `OPENROUTER_API_KEY` was available, so every live step is marked "skipped: no key". Public pages and the public model list need no key and use no quota.

**Free-tier status: unknown.** It needs the key. `scripts/preflight/s06_key_status.py` reports it.

## Results

| Step | Item | Result | Evidence | What changed | Your decision |
|---|---|---|---|---|---|
| 0 | Repository state | PASS | The folder was not a git repository. `origin/main` held only a one-line `README.md` | Ran `git init`, added `origin`, based `main` on `origin/main`. README title kept, project section appended | None |
| 1 | Python and uv | PASS | System Python 3.14.6, uv 0.12.7; 3.13.13 also installed | None | None |
| 2 | `uv sync --extra dev` | PASS after fix | First run failed: `numpy>=2.5` needs Python 3.12, `requires-python` said 3.11. Then 257 packages in 14m30s, at the tested versions. No `anthropic`, `litellm` or `llama-index-llms-*` | `requires-python = ">=3.12"`, `.python-version` = 3.13 (your choice), `uv.lock` added | None |
| 3 | Tests and ruff | PASS | 36 reference tests pass; `ruff check` and `ruff format --check` clean on the starter kit | No edit needed | None |
| 4 | `.env` and key | SKIPPED | `.env` was missing | `.env` created from `.env.example` with the model picks. `OPENROUTER_API_KEY` is empty | Add the key |
| 5 | Free-tier limits | PASS | Constants in the live page: 20 per minute, 50 per day, 1,000 per day, threshold 10 credits | Documented: the higher ceiling starts at 9 credits, and `GET /api/v1/key` reports the daily counter. `.env.example` limits unchanged | None |
| 6 | Key status | SKIPPED: no key | Endpoint found: `GET /api/v1/key`, fields `is_free_tier`, `free_model_daily_requests` | `scripts/preflight/s06_key_status.py` is ready | None |
| 7 | `pick_free_models.py` | PASS | Field names checked on a live response (453 models, 15 at price 0, 12 usable after filtering). 8 unit tests | Script and tests added; ARCHITECTURE section 11 corrected | None |
| 8 | Model picks | PASS | Picks below. All `.env.example` placeholders still exist with the needed capabilities, so they stay | `.env` filled; `.env.example` comment updated | None |
| 9 | Shadow pricing | CHANGED | You decided on free models only, with no paid reference | `configs/pricing.yaml` holds a free model at 0, so shadow dollars read 0. `llm/pricing.py` exposes `_reference`. PRD F-OPS-5 and OPEN-4 reworded | None |
| 10 | Structured call, usage, serving-model key | SKIPPED: no key | | `s10_structured.py` ready, about 2 requests | Run after adding the key |
| 11 | Image payload | SKIPPED: no key | | `s11_vision.py` ready, 1 request | Run after adding the key |
| 12 | `temperature` on the judge | SKIPPED: no key | | `s12_temperature.py` ready, 1 request | Run after adding the key |
| 13 | CrewAI against the real service | SKIPPED: no key | | `s13_crewai.py` ready, up to 3 requests | Run after adding the key |
| 14 | Embeddings | PASS | 384 dimensions, norm 1.000000, identical text cosine 1.000000. Download about 3.5 min, 128 MB | **Doc claim was wrong:** LlamaIndex adds a query prefix (cosine 0.951). Fix documented: `query_instruction=" "` | None |
| 15 | Chroma and LlamaIndex | PASS after fixes | Cosine space, docstore, reload, BM25, hybrid fusion, `FilterOperator.IN` and OR. `exp(-d)` equals the LlamaIndex score (0.5433 against cosine 0.3899). Self cosine 1.0000 | **Two silent bugs found:** `doc_id` overwritten (filters return nothing) and metadata embedded (self cosine 0.87). Fixes in ARCHITECTURE section 6 | None |
| 16 | Reranker | PASS | Correct top passage. 35 to 54 ms for 5 passages on CPU; model load and download 102 s, 88 MB | None | None |
| 17 | Streamlit | PASS | `None` until submitted, then `ChatInputValue(text, files, audio)`. `st.pyplot` ran without error through `streamlit.testing`. Not checked in a real browser | Docs updated | None |
| 18 | Docker | PARTIAL | Client 29.8.2 installed; daemon not running, so a build could not be judged | None | Start Docker Desktop before M8 |
| 19 | Sample corpus | PASS | 6 documents, 164 PDF pages (below) | `scripts/fetch_sample.py`, `data/sample/SOURCES.md` (git-ignored) | None |

## Model picks (from the list only; none has been called)

| Role | Model | Capabilities claimed by the list |
|---|---|---|
| fast, strong, vision | `nvidia/nemotron-3.5-super-vl:free` | Image input, `structured_outputs`, `response_format`, tools, temperature; no forced reasoning |
| judge | `xingchen-agi/xing4.0-29b-a4b:free` | `structured_outputs`, `response_format`, tools, temperature; no forced reasoning |

`dots-studio/dots-3-note-preview:free` also qualifies but is skipped because it has an expiration date of 2026-12-31.

## Sample corpus

| File | Pages | Images passing the filter | Licence |
|---|---|---|---|
| `NIST.AI.100-1.pdf` | 48 | 9 | US government work; NIST grants a royalty-free right to reprint |
| `NIST.CSWP.29.pdf` | 32 | 9 | Same |
| `NIST.SP.800-218.pdf` | 36 | 6 | Same |
| `NIST.SP.800-61r3.pdf` | 48 | 5 | Same |
| `fastapi-first-steps.md` | n/a | 0 | MIT, pinned to a commit |
| `uv-readme.md` | n/a | 0 | MIT or Apache-2.0, pinned to a commit |

**Caption budget:** 29 images, so 29 LLM requests for one full image ingest. No document reaches the cap of 10. The caption cache means this is paid once.

NIST notes that some works may include third-party material under copyright. The corpus is git-ignored and fetched by script, so nothing is redistributed from this repository.

## Download sizes

| Item | Size | Time |
|---|---|---|
| `.venv` | 1.9 GB | 14m30s install |
| uv package cache | 2.1 GB | included above |
| `BAAI/bge-small-en-v1.5` | 128 MB (llama_index cache) | about 3.5 min |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 88 MB (`~/.cache/huggingface`) | 102 s |
| Sample corpus | 5.1 MB | under 1 min |

## Recommendation on OPEN-5

**Buy credits once if you can.** The request budget is the reason:

| Activity | LLM requests | At 40 per day (`MAX_REQUESTS_PER_DAY`) |
|---|---|---|
| Caption the sample corpus | 29 | Most of one day |
| One golden-set run | about 80 to 120 | 2 to 3 days |
| All experiments E1 to E7 | several hundred to about 1,000 | About three to four weeks |

These figures are design estimates from ARCHITECTURE section 13, apart from the caption count, which is measured. Credits are only spent on paid models, so buying them does not change your free-models-only decision. The cap rises to 1,000 per day. The docs state this, but it has not been tested on your account.

Staying at 50 per day works, but it is slow. Use `--offline`, `--max-requests 35` and `--resume`, and spread captioning over the first day.

## Needs your decision

1. **Add `OPENROUTER_API_KEY` to `.env`**, then run `scripts/preflight/s06_key_status.py`, `s10`, `s11`, `s12` and `s13`. They use at most 7 LLM requests. Until then nothing about real OpenRouter behaviour is verified.
2. **OPEN-5:** buy credits or stay at 50 per day (recommendation above).
3. **Docs disagree on two defaults.** `HISTORY_TURNS` is 6 in the PRD and 4 in `.env.example`. `MAX_IMAGES_PER_DOC` is 20 in the PRD and 10 in `.env.example` and ARCHITECTURE. M0 uses the `.env.example` values.
4. **`IngestReport` gap.** F-ING-5 asks for caption requests and tokens, but the schema has only `caption_cost_usd`, which now always reads 0. Fixing it is a public schema change, so it needs your approval before M5.
5. **`SemanticCache` has no `stats()`,** but the `CacheBackend` interface in ARCHITECTURE section 9 requires it. Decide at M6: wrap it or add the method to the reference code.
6. **Reference comments still say "paid reference"** in `schemas.py` and `eval/metrics.py`. They are left as provided under the rule not to change reference code.
7. **`docs/PREFLIGHT_PROMPT.md` still says to push a branch.** It is left as provided, because your instruction to work on `main` applied to this run.
8. **Privacy.** Check the OpenRouter account setting for free models before uploading anything private (ARCHITECTURE section 18).

## Not done, by rule

No golden-set item, cache pair or threshold was created. `MIN_TOP_SIMILARITY` and `CACHE_THRESHOLD` are unchanged.
