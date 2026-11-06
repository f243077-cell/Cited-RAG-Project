# M6 test guide: semantic cache

Run from the repo root. Stop the API before any `citerag ingest` or `citerag eval` command that opens the index.

| # | Do this | Expect | LLM requests |
|---|---|---|---|
| 1 | `uv run pytest tests/unit/test_cache_store.py tests/unit/test_cache_tools.py tests/unit/test_cache_scripts.py tests/unit/test_cache_plots.py tests/integration/test_cache_api.py tests/integration/test_bench_cache.py` | All pass | 0 |
| 2 | `uv run citerag eval cache-sweep` before labelling | `validation_error`: the file is missing and the message names the candidates and the label command. Exit 1 | 0 |
| 3 | `uv run citerag eval cache-sweep --pairs eval/cache_pairs.candidates.jsonl` | `No labelled pairs, so there is nothing to sweep`. Exit 1 | 0 |
| 4 | `uv run python scripts/review_cache_pairs.py --out eval/results/cache_pairs_review.md`, then read the file | 80 pairs, each with both questions, the words that differ and the guard's verdict | 0 |
| 5 | `uv run python scripts/review_cache_pairs.py --label`. Type `e`, `n`, `s` or `q`. Run it again to continue | `eval/cache_pairs.jsonl` holds only what you typed, saved after each answer | 0 |
| 6 | `uv run citerag eval cache-sweep` after labelling | Counts, a recommended threshold, the five closest near misses, `docs/results/e5_threshold.png`. "Nothing was changed": check `.env` is untouched | 0 |
| 7 | Copy the threshold you accept into `CACHE_THRESHOLD`, and restart the API | `curl localhost:8000/v1/cache/stats` shows that threshold | 0 |
| 8 | Start the API and UI. On Chat ask a question, click New chat, ask the same question | Second answer shows "Served from cache", similarity near 1.00, 0 LLM requests | 1. No-request version: step 1 covers it in `test_the_same_question_is_served_from_the_cache_with_no_llm_call` |
| 9 | Ask a paraphrase after New chat | A hit if the similarity reaches your threshold, otherwise a miss | 0 or 1 |
| 10 | Turn off "Use cache" and ask again, or pick a document under "Search only in" | No badge, a normal answer | 1 each. No-request version: `tests/unit/test_pipeline.py` |
| 11 | Open Dashboard | Hit rate, latency, cumulative requests, stages, calls by role, similarity chart with the threshold line, and the sweep chart once step 6 has run | 0 |
| 12 | Press "Clear cache", or run `uv run citerag cache stats`, then `uv run citerag cache clear` | Entries read 0 | 0 |
| 13 | Upload or delete a document, then run `curl localhost:8000/v1/cache/stats` | Entries read 0: the namespace changed | 0, with captions off |

**E6 replay (left unrun).** Build the workload from your labels, check the estimate, then spread it over days:

```bash
uv run python scripts/make_workload.py
uv run citerag eval bench-cache --workload eval/workload.jsonl --index-dir data/index_eval --estimate
uv run citerag eval bench-cache --workload eval/workload.jsonl --index-dir data/index_eval --skip-judge --max-requests 30
uv run citerag eval bench-cache --workload eval/workload.jsonl --index-dir data/index_eval --resume <run_id> --max-requests 30
```

If you accept all 37 equivalent pairs, the workload is 63 questions, 59% repeats. **Cost: about 89 requests without verdicts, 178 with them, and 504 at the very worst.** At 30 a day that is 3 to 6 days. Leave out `--skip-judge` on the last resume to add verdicts. The run writes `docs/results/E6_cache_benchmark.md` and `e6_cache.png` when both passes finish. No-request rehearsal: `uv run pytest tests/integration/test_bench_cache.py`.

**Failure means any of these:**

- A step above gives another result, or a command prints a Python traceback.
- A sweep runs on unlabelled pairs, or changes `.env`.
- A cached answer has an invalid citation, or a badge shows on an image, Deep or filtered request.
- After a corpus change, an old answer comes back from the cache.
- In E6: a false hit, a hit rate under 30%, a reduction under 25%, correctness down by more than 1 point, or hit p95 over 300 ms. Explain any of these in the report.
