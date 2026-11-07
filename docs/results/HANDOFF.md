# Handoff: what only you can do, in order

Written for Coder at the wind-up, 2026-11-07. Each step names its command, rough time, LLM request cost, what it unlocks and
what counts as failure. Steps 1 to 6 cost **no LLM request**. Run every command from the repo root with the API stopped, unless a
step says otherwise (Chroma allows one writer).

**Daily quota:** the app stops at `MAX_REQUESTS_PER_DAY` (40) in a rolling 24 hours, below OpenRouter's 50. `uv run citerag quota`
shows where you are. A capped run pauses cleanly and `--resume <run_id>` carries on the next day without paying twice
(rehearsed in `tests/integration/test_handoff_tools.py`).

| # | Step | Command | Minutes | LLM requests | Unlocks | Failure means |
|---|---|---|---|---|---|---|
| 1 | Build the evaluation index (once) | `uv run citerag ingest data/sample --index-dir data/index_eval`, then `uv run citerag ingest eval/fixtures/poisoned --index-dir data/index_eval` | 10 the first time (model download), 2 after | 0 (captions off) | Every evaluation command | A traceback, or `0 chunk(s)` for a PDF |
| 2 | Review the golden candidates | `uv run python scripts/review_golden.py --out eval/results/golden_review.md`, then read the file; correct any question, answer or page in `eval/golden/candidates.jsonl` | 90 to 150 for 43 items | 0 | Step 3 | The script crashes, or a gold page does not hold the answer and you keep it |
| 3 | Accept what you checked | `uv run python scripts/accept_golden.py --ids a-001,a-002,...` (only the ids you reviewed) | 2 | 0 | All verified runs: `--require-verified` | `uv run python scripts/check_release.py` fails, or an unchecked id is in `golden.jsonl` |
| 4 | Confirm the set loads, and measure retrieval | `uv run citerag eval run --golden eval/golden/golden.jsonl --require-verified --index-dir data/index_eval --offline --rerank` | 5 | 0 | hit@k, MRR, full coverage on the verified set; the gate data for step 5; the rerank decision (OPEN-2) | `not verified by a human` (an item slipped in unverified), or hit@5 under 85% |
| 5 | Choose `MIN_TOP_SIMILARITY` | `uv run citerag eval gate <run_id from step 4>`, then set the value you accept in `.env` | 10 | 0 | A calibrated gate (F-RET-4). The code never sets it | The proposal refuses answerable items you can see should pass |
| 6 | Label the cache pairs, then sweep | `uv run python scripts/review_cache_pairs.py --label` (e, n, s, q; run again to continue), then `uv run citerag eval cache-sweep` | 40 for 80 pairs, 2 for the sweep | 0 | A validated `CACHE_THRESHOLD`. Set it and `CACHE_ENABLED=true` in `.env`; the start-up warning then stops | "No threshold is both safe and useful", or fewer than 30 pairs of either kind (the sweep says PROVISIONAL) |
| 7 | Pick a judge from another model family | `uv run python scripts/pick_free_models.py`, then set `MODEL_JUDGE`. Check with one item: `uv run citerag eval run --golden eval/golden/golden.jsonl --require-verified --index-dir data/index_eval --limit 1` | 15 | about 2 | Verdicts that are not the answerer grading itself | The judge model returns no valid `JudgeVerdict` twice (CLAUDE.md gotchas list the earlier failures) |
| 8 | Baseline run (M4) | `uv run citerag eval run --golden eval/golden/golden.jsonl --require-verified --index-dir data/index_eval --skip-type image --max-requests 35`, then the same with `--resume <run_id>` each day until `complete` | 10 a day | about 71 for 38 items, 140 at worst | `docs/results/baseline.md` without the PROVISIONAL label; PRD section 7 targets | The run keeps going after `llm_unavailable`, or exceeds `--max-requests` |
| 9 | Cache benchmark E6 | `uv run python scripts/make_workload.py`, then `uv run citerag eval bench-cache --workload eval/workload.jsonl --index-dir data/index_eval --estimate`, then the same without `--estimate` and with `--skip-judge --max-requests 35`, resumed daily | 10 a day | about 89 without verdicts, 178 with | `docs/results/E6_cache_benchmark.md`: hit rate, request reduction, correctness change | A false hit, or correctness down more than 1 point |
| 10 | Deep vs standard E7, and the three modes E8 | Print the cost first: `uv run citerag eval run --golden eval/golden/golden.jsonl --require-verified --config configs/experiments/e8_agent.yaml --skip-type image --estimate` (and `e8_standard`, `e8_deep`). Then run each with `--max-requests 35` and `--resume`, then `uv run citerag eval report-e8 <standard> <deep> <agent>` | 10 a day | about 405 for all three on 38 items | `docs/results/E8_modes.md`; E7 is its standard and Deep columns | Any canary in an answer or a tool argument |
| 11 | Multi-agent comparison (M12) | `m12_single.yaml`, `m12_supervisor.yaml` and `e8_deep.yaml` with `--skip-type unanswerable --skip-type injection --skip-type image --limit 15`, then `uv run citerag eval report-e8 <single> <supervisor> <deep> --title "M12: single agent vs supervisor vs Deep" --out docs/results/M12_multi_agent.md` | 10 a day | about 90 if E8 runs are reused, 210 if not | The "when does multi-agent add value" conclusion | More than 11 requests for one supervisor item |
| 12 | Docker, once | `docker compose up --build`, then `curl http://localhost:8000/health` and open http://localhost:8501 | 90 the first time (a 21 GB image), 2 after | 0 until you ask a question | F-OPS-4 fully proven | The backend exits at start (read `docker compose logs backend`), or a provider key shows in `docker compose exec frontend env` |

**Your `.env`.** The code default for `CACHE_ENABLED` is now `false`, but `.env` wins. The wind-up did not edit your `.env`. If it
still says `CACHE_ENABLED=true`, the API logs a warning at start and `/health` lists it until step 6 validates the threshold.

**Money (your call, not done).** Steps 8 to 11 take two to three weeks at 35 requests a day. Buying at least 10 OpenRouter credits
raises the free-model limit to 1,000 a day (PRD OPEN-5). Then set `MAX_REQUESTS_PER_DAY` higher, and the whole programme fits in a
day.

**No-request rehearsals** of steps 8 to 11: `uv run pytest tests/integration/test_handoff_tools.py tests/integration/test_bench_cache.py tests/integration/test_e8.py tests/integration/test_supervisor.py`.
