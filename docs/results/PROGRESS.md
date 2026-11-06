# Build progress: M6, M7, M8

Run started 2026-11-06. Quota rule for this run: no live LLM requests on either provider. Requests used so far: 0.

| Milestone | State |
|---|---|
| M6 semantic cache | built and pushed; exit criteria need your labels and a live replay |
| M7 CrewAI | not started |
| M8 polish | not started |

## M6

**Done:** `PersistentCache` over the reference cache, guard, namespace invalidation at start-up and on corpus change, `citerag cache stats|clear`, `/v1/cache/stats|clear`, `/v1/logs/*`, `eval cache-sweep`, `eval bench-cache`, Dashboard page and charts 1 to 7, Chat toggle and badge, 80 candidate pairs, `scripts/review_cache_pairs.py`, `scripts/make_workload.py`, `M6_test_guide.md`.

**Not done:** threshold chosen (needs your labels), E5 chart, E6 report (needs a live run), and tokens saved per request (decision 2).

## Decisions for Coder

1. **The cache is on with the placeholder threshold, and that looks unsafe.** `CACHE_ENABLED` defaults to true and `CACHE_THRESHOLD` stays 0.92, as you ordered. A smoke run of the sweep on a throwaway copy, using my proposed labels and not yours, gave a false-hit rate of 33% at 0.92 (14 of 43 near misses) and 0% only at 0.98, where recall is 32%. The set is deliberately adversarial and is not a result. Until you run the sweep, set `CACHE_ENABLED=false` in `.env` if you will rely on answers.
2. **Tokens saved (F-CAC-7) needs a public schema change.** The requests table and `CacheInfo` have no such field, so the log records saved requests and saved shadow dollars only. `bench-cache` still reports tokens with and without the cache. Approve `saved_tokens` on `CacheInfo` and `requests` if you want it per request.
3. **The guard misses meaning the reference code cannot see.** `guard_ok` compares digits and a word list, so these pass it: "first" vs "second", "days" vs "hours", "address" vs "leave out". Near misses of that kind can only be stopped by the threshold, which is why it must be so high. A wider guard means changing reference code, so it is your call.
