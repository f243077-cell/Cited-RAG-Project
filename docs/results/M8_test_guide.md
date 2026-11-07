# M8 test guide: polish

Run from the repo root unless a step says otherwise.

| # | Do this | Expect | LLM requests |
|---|---|---|---|
| 1 | `uv run pytest` and `uv run ruff check .` | All pass, ruff clean | 0 |
| 2 | `uv run pytest tests/integration/test_request_guard.py` | All pass: a full guard answers 429, nothing leaves, hits are still served | 0 |
| 3 | `uv run citerag quota` | `openrouter: N of 40`, `gemini: N of 18` | 0 |
| 4 | Set `MAX_REQUESTS_PER_DAY` to the N from step 3, restart the API, ask a new question | HTTP 429 `budget_exceeded` naming the limit and when a slot frees up. No new row in `llm_calls`. Restore the value | 0 |
| 5 | With the guard still full, ask a question that is already cached, after New chat | The cached answer with its badge | 0 |
| 6 | `uv run citerag logs purge --days 30` | `Deleted N requests, N LLM calls, N messages`. `--days 0` is refused | 0 |
| 7 | Fresh clone: follow the README Quickstart in an empty folder, up to Streamlit | Tests pass with no key, `/health` returns 200, the UI opens within 15 minutes of starting | 0 until you ask a question, then 1 |
| 8 | Read `docs/results/security_checklist.md`, starting with the Gaps | You accept each gap or tell me to close it | 0 |
| 9 | Read the README | The diagram renders on GitHub, and the Results section is marked provisional | 0 |

**Docker.** Untested in M8 (the daemon was not running). Built and passed at the wind-up on 2026-11-07 (WINDUP_SMOKE.md). With Docker Desktop open:

```bash
docker compose config --quiet
docker compose up --build
curl http://localhost:8000/health
docker compose exec frontend env | grep -c OPENROUTER
docker compose exec backend ls -a /app | grep -c '^.env$'
```

| Check | Expect |
|---|---|
| `compose config` | No output, exit 0 |
| Build | Finishes, and may be several GB: the lock file pulls CUDA-enabled PyTorch |
| `/health` | 200 |
| http://localhost:8501 | The UI opens and lists your documents |
| Both `grep -c` commands | `0`: no provider key in the frontend, no `.env` in the image |

**Live cost:** none for Docker unless you ask a question, then 1 request or more. **No-request alternative:** none for the image itself; the compose check and step 7 cover what they can.

**Failure means any of these:**

- Step 1 or 2 fails, or a request reaches the provider while the guard is full.
- `--days 0` deletes anything.
- The quickstart needs a step that the README does not list, or takes over 15 minutes after the downloads.
- A provider key appears in the frontend container or in the image.
- A port answers on anything but 127.0.0.1.
- A number in the README is not marked provisional or differs from `docs/results/`.
