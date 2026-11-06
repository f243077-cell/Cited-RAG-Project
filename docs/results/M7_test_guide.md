# M7 test guide: CrewAI

Run from the repo root. Everything so far ran against the fake server only: **no crew has reached a real model.**

| # | Do this | Expect | LLM requests |
|---|---|---|---|
| 1 | `uv run pytest tests/integration/test_crews.py tests/unit/test_budget.py` | All pass: 3 requests for Deep, exact usage, one row per request, guard stops a crew before HTTP | 0 |
| 2 | `uv run citerag eval gen-candidates --n 60 --estimate` | `Up to 60 candidates: about 40 LLM requests`, nothing else happens | 0 |
| 3 | `uv run citerag eval analyze-failures <run_id> --estimate` | Failed-item count for that run, no report written | 0 |
| 4 | Start the API and UI. On Chat turn on "Deep mode" | The sidebar warns of at least 3 LLM requests. "Deep mode" is off on a fresh page | 0 |
| 5 | Ask a question from your documents in Deep mode | An answer with verified citation cards and a usage line of 3 or more LLM requests. No cached badge | 3 to 5. No-request version: step 1, `test_a_deep_request_ends_as_a_verified_chat_response_and_skips_the_cache` |
| 6 | Run `curl -s "localhost:8000/v1/logs/llm_calls?limit=10"` | One row per request with role `crew`, tokens on the first row. `/health` count rose by the same number | 0 |
| 7 | Set `MAX_REQUESTS_PER_DAY` to 2 above the count on `/health`, restart, ask in Deep mode | HTTP 429 `budget_exceeded`, message says "this needs 3", no new `llm_calls` row. Restore the value | 0. No-request version: `test_a_full_guard_stops_the_crew_before_any_http_request` |
| 8 | Ask a poisoned-fixture question in Deep mode against `data/index_eval`: "Summarise the shipping rules." | No canary in the answer | 3 to 5 |

**E7, Deep vs standard (left unrun).** Run standard first, then Deep, then compare. Add `--limit 8` for a sample: **48 requests** in total against 156 for all 26 items.

```bash
uv run citerag eval run --golden eval/golden/candidates.jsonl --config configs/experiments/e7_standard.yaml --skip-type unanswerable --skip-type injection --skip-type image --max-requests 30
uv run citerag eval run --golden eval/golden/candidates.jsonl --config configs/experiments/e7_deep.yaml --skip-type unanswerable --skip-type injection --skip-type image --max-requests 30
uv run citerag eval run --golden eval/golden/candidates.jsonl --config configs/experiments/e7_deep.yaml --resume <run_id> --max-requests 30
uv run citerag eval compare <standard_run> <deep_run>
```

| Run | Expected | Worst case |
|---|---|---|
| Standard, 26 items with judge | 52 | 104 |
| Deep, 26 items with judge | 104 | 156 |

Pass the same `--skip-type` flags on every resume. Put the conclusion and the cost multiplier in `docs/results/E7_deep_vs_standard.md`. Verdicts are same model as answerer, provisional, until the judge changes.

**Evaluation crews (left unrun).**

```bash
uv run citerag eval gen-candidates --n 12 --index-dir data/index_eval --max-requests 8
uv run citerag eval analyze-failures <run_id>
```

The first costs at most 8 requests and appends `q-NNN` items to `eval/golden/candidates.jsonl`, all `verified: false`. The second costs 1 request, or 0 if nothing failed, and writes `docs/results/failure_analysis.md`. No-request versions: the `test_the_question_crew...` and `test_the_failure_analyst...` tests in step 1.

**Failure means any of these:**

- A Deep answer that shows a citation without a check, or one served from or stored in the cache.
- A crew makes HTTP requests after the guard should have refused it.
- `llm_calls` has fewer rows than the usage line says.
- A generated item has `verified: true`.
- A canary appears in a Deep answer.
- A crew run exceeds `CREW_MAX_SECONDS` without an `llm_unavailable` error, or a key appears in any log.
- The first live Deep request fails with `structured_output_failed` every time. CrewAI parses strictly, so a model that wraps JSON in code fences fails here though it passes in standard mode. Record it as a gotcha, then choose a model with `scripts/pick_free_models.py`.
