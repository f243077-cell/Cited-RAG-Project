# Security checklist

Review of ARCHITECTURE section 18, 2026-11-07. Scope: one person on one machine. "Tested" names a test that passes in the default suite; "checked" means I read the code or ran the command. No external penetration test was done.

| Area | Control | State | Evidence |
|---|---|---|---|
| Uploads | Extension allow-list, magic bytes, empty files | Pass, tested | `validate_file`; `tests/unit/test_loaders.py::test_rejected_files` |
| Uploads | Size cap | Pass, with a gap below | `test_size_limit_uses_mb_and_allows_the_exact_limit` |
| Uploads | Names sanitised, stored under hashed names, no user path reaches the disk | Pass, tested | `test_sanitize_name`; `data/uploads/<doc_id>.<ext>` |
| Uploads | Never executed or rendered | Pass, checked | Text is parsed, and images are handled by pypdf and passed as data |
| Injection | Instruction and data separation, escaping | Pass, tested | `render_passages`; `test_passages_stay_escaped_inside_the_documents_block`; `test_judge_input_escapes_the_answer`; `test_failure_text_stays_escaped` |
| Injection | Canary tests | Provisional | 0 of 5 leaks in standard mode. **Deep mode has no live canary run:** M7 guide, step 8 |
| Injection | No tools, schema-constrained output | Pass, checked | Crews have no tools and no delegation; every task has `output_pydantic` |
| Injection | Output is plain Markdown | Pass, checked | No `unsafe_allow_html`, `st.html` or `components.html` in `src/` |
| Cache | Wrong-answer risk | **Open** | The placeholder threshold let 14 of 43 near misses through in a smoke run: decision 1 in `PROGRESS.md` |
| Secrets | `.env` git-ignored and never committed | Pass, checked | `git log --all -- .env` is empty; `git ls-files` shows no `.env`, `*.sqlite` or `data/` |
| Secrets | No key in tracked files | Pass, checked | `git grep` for OpenRouter and Google key patterns finds nothing |
| Secrets | Keys held as `SecretStr`, never logged | Pass, tested | `test_rejected_key_fails_at_once_without_echoing_it`; `test_the_key_is_not_in_any_logged_row` |
| Secrets | The UI never holds a provider key | Pass, checked | `UISettings` has no such field; the compose frontend has no `env_file` |
| Secrets | Crews get the key as an argument | Pass, with a note | Importing `crewai` runs `load_dotenv()`, so `.env` still lands in the API process environment |
| API | Optional `X-API-Key`, compared in constant time | Pass, tested | `test_api_key_is_enforced_only_when_set`; `test_cache_and_log_routes_need_the_key_when_one_is_set` |
| API | Localhost CORS | Pass, checked | `CORS_ORIGINS` default; only the setting's origins are allowed |
| API | Daily request guard | Pass, tested | `tests/integration/test_request_guard.py` |
| API | Request size limits | Partial | See gap 1 |
| API | In-memory rate limit [SHOULD] | **Not built** | See gap 2 |
| Logs | No key or image bytes stored | Pass, checked | Table columns in `analytics/logs.py` |
| Logs | Retention | Pass, tested | `citerag logs purge`; `test_purge_removes_old_rows_only` |
| Logs | Question text is stored locally | Accepted | Documented in the README |
| Privacy | README statement, `OPENROUTER_DATA_COLLECTION` | Pass, tested | Crews send the same routing body: `test_requests_carry_the_routing_rules_and_the_right_models` |
| Docker | Secrets stay out of the image | Written, untested | `.dockerignore` lists `.env`, `data/`, `eval/` |
| Docker | Ports bound to 127.0.0.1 | Written, untested | `docker-compose.yml` |

## Gaps

1. **An upload is read into memory before the size check when the client sends no size.** `file.size` is checked first, but `validate_file` reads the bytes before rejecting them. Chat bodies are read in full before Pydantic rejects them. On localhost the risk is your own machine's memory. Behind a proxy, set a body limit there.
2. **No API rate limit.** Only the LLM guard limits spending. Anyone who can reach the port can use up your daily quota, so keep it on localhost, and set `CITERAG_API_KEY` if others share the machine.
3. **`/health` needs no key.** It shows package versions, the namespace and document names in the manifest. That is fine on localhost; do not expose it.
4. **The container runs as root.** A non-root user needs an owner for the `./data` bind mount, which could not be tested here.
5. **A timed-out crew keeps running.** The request returns `llm_unavailable` at `CREW_MAX_SECONDS`, but the worker thread runs until the agents' own limit, and can still make a request.

## Before sharing beyond this machine

Set `CITERAG_API_KEY`, add a body limit and a rate limit, set `OPENROUTER_DATA_COLLECTION=deny` if the documents are private, and give the OpenRouter key no more credit than you need.
