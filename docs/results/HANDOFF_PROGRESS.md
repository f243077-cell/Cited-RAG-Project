# Handoff progress: steps 1 to 8, done by Claude on 2026-11-08

**Every number here is PROVISIONAL and AI-reviewed, not human-verified.** On 2026-11-08 Coder asked Claude to do the
HANDOFF.md steps and chose "AI review, labelled" (DECISIONS D-15). Each golden item and cache pair carries
`reviewed_by: claude`, and every report, CLI line and UI page says so. A human can re-check any item with
`scripts/review_golden.py` or `scripts/review_cache_pairs.py` and re-accept it with `--reviewed-by human`.

## Step 1: evaluation index

`data/index_eval` already held the sample corpus and the poisoned fixtures (11 documents). It was not rebuilt.

## Steps 2 and 3: golden set reviewed and accepted (0 requests)

All 43 candidates were checked against their gold page text (`scripts/review_golden.py`), and the six image fixtures were
viewed. 41 were correct as written. Two gold pages that do not hold the answer were removed:

| Item | Removed source | Why |
|---|---|---|
| a-001 | NIST.AI.100-1.pdf p39 | It is about cross-sectoral profiles, not the four Core functions |
| a-016 | NIST.CSWP.29.pdf p19 | It describes the AI RMF Core, not the CSF Core |

The near-miss terms of the unanswerable items were checked in the PDFs too ("fine" is "fine-tune", "hours" is a containment
example, no CISA deadline, no AI RMF version 2). `eval/golden/golden.jsonl` now holds 43 items, `verified: true`,
`reviewed_by: claude`.

## Step 4: retrieval on the reviewed set (0 requests)

Run `20261108-143550-golden`, `--offline --rerank --require-verified`, `data/index_eval`. 26 items are scored for
retrieval: 21 answerable and 5 multi-chunk (unanswerable and injection items have no gold page; images are skipped).

| Metric | Hybrid | After rerank |
|---|---|---|
| hit@1 | 69.2% | 69.2% |
| hit@3 | 84.6% | 92.3% |
| hit@5 | 92.3% | 96.2% |
| MRR | 78.2% (@10) | 79.6% (@5) |
| multi_chunk full coverage@5 | 40.0% (2 of 5) | 80.0% (4 of 5) |

hit@5 is above the 85% bar in HANDOFF step 4. With 26 items, one item is 3.8 points.

## Step 5: relevance gate (0 requests)

`uv run citerag eval gate 20261108-143550-golden`:

| MIN_TOP_SIMILARITY | Answerable or multi-chunk stopped (of 26) | Unanswerable stopped (of 7) |
|---|---|---|
| 0.55 | 0 | 1 |
| 0.60 | 0 | 1 |
| 0.65 | 0 | 2 |
| 0.70 | 0 | 3 |
| 0.75 | 4 | 6 |

The tool proposes 0.70, the highest value with no false refusal. The lowest answerable item scores 0.701, so 0.70 has a
margin of 0.001. **Chosen: 0.65**, which still stops 2 of 7 unanswerables with a margin of 0.05. The model must refuse the
rest (CLAUDE.md gotcha: on-topic unanswerables score like answerable items).

## Step 6: cache pairs and threshold sweep (0 requests)

All 80 candidate pairs were labelled pair by pair: 37 equivalent, 43 near misses. The labels agree with the drafter's
proposals; the closest call, cp-065 ("hierarchy of the AI RMF Core" against the CSF Core), is a near miss because a cached
answer would cite the CSF for an AI RMF question.

`uv run citerag eval cache-sweep`: **recommended CACHE_THRESHOLD 0.98**, false-hit rate 0.0%, recall 32.4%, precision
100.0%. The near misses closest to the line (cp-042 0.997, cp-049 0.993, cp-050 0.979, cp-052 0.978, cp-041 0.976) are all
blocked by the guard. The cache stays off (`CACHE_ENABLED=false` in Coder's `.env`); with it on, the start-up warning now
says the threshold comes from AI-labelled pairs.

## Step 7: judge from another model family (2 requests)

`scripts/pick_free_models.py` listed `dots-studio/dots-3-note-preview:free` (structured output and `response_format`,
expires 2026-12-31). One item (`a-001`, run `20261108-144432-golden`) passed: the answer on `nvidia/nemotron-3.5-super-vl:free`,
the verdict on `dots-studio/dots-3-note-preview:free` (8.4 s), both logged in `llm_calls`. One item says the model works,
not that it judges well.

## Step 8: baseline, started (10 requests)

Run `20261108-144525-golden`, `--skip-type image --max-requests 12`, with `MODEL_JUDGE=dots-studio/dots-3-note-preview:free`
and `MIN_TOP_SIMILARITY=0.65` passed as environment variables (Coder's `.env` was not edited). It paused cleanly at the cap
after 5 of 38 items: faithfulness, correctness, citation validity and precision all 5 of 5. Five easy answerable items say
nothing yet; it resumes daily (HANDOFF.md).

## Quota

38 of 40 requests in the 24 hours to 20:08 on 2026-11-08, of which 12 were these steps; the rest were Coder's own use and
the wind-up smoke test.
