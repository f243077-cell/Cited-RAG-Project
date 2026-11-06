# Baseline (provisional)

Date: 2026-11-06. **PROVISIONAL: this is not the M4 baseline.** The exit criterion needs a human-verified golden set, and all 43 items are still drafts with `verified: false`, so no number here may be quoted as a result. Only one item has been judged (same model as answerer, provisional), so faithfulness and correctness have no value yet.

Setup: `nvidia/nemotron-3.5-super-vl:free` for all four roles, prompt version 1, `data/index_eval`, hybrid search, rerank off. Image items were not part of these runs: see `M5_images.md`.

```bash
uv run citerag eval run --golden eval/golden/candidates.jsonl --index-dir data/index_eval --offline --rerank
uv run citerag eval run --golden eval/golden/candidates.jsonl --index-dir data/index_eval --skip-judge --max-requests 16 --skip-type answerable --skip-type multi_chunk
```

## Against the targets

| Goal | Metric | Target | Now | State |
|---|---|---|---|---|
| G1 | Faithfulness (judge) | at least 0.90 | not measured | Needs the answerable run; any value will be same model as answerer, provisional |
| G1 | Citation validity (verifier) | at least 0.95 | not measured on the set | Needs the answerable run |
| G1 | Answer correct (judge) | at least 0.75 | not measured | Needs the answerable run; any value will be same model as answerer, provisional |
| G2 | Correct refusals | at least 0.90 | 7 of 7 (100%) | Provisional, measured at gate 0.30 |
| G2 | Injection canary leaks | 0 | 0 of 5 | Provisional; `i-005` retested with its image captioned, still no leak |
| G3 | hit@5 (hybrid) | at least 0.85 | 24 of 26 (92%) | Provisional |

PRD section 7 is unchanged: it should move only after a verified baseline.

## Multi-chunk coverage

Full coverage@k is the headline for `multi_chunk` items: every distinct gold part must be in the top k, while hit@k needs only one.

| k | Hybrid | After rerank |
|---|---|---|
| 3 | 2 of 5 | 2 of 5 |
| 5 | **2 of 5** | **4 of 5** |
| 10 | 4 of 5 | not applicable |
| 20 | 4 of 5 | not applicable |

The model sees only the top 5, so with rerank off 3 of the 5 items reach it missing a part. The 92% hit@5 hides that. One item never gets full coverage even at 20. Five items is too few to decide on rerank, but this is the strongest evidence for it so far.

Part by part, hybrid, k=5, from `uv run citerag eval parts --index-dir data/index_eval`:

| Item | Part | First rank | At k=5 | Pages |
|---|---|---|---|---|
| `m-001` | 0 | 1 | ok | AI p5, p25, p27 |
| | 1 | 6 | **missed** | CSF p8, p21 |
| `m-002` | 0 | 1 | ok | CSF p29 |
| | 1 | 7 | **missed** | CSF p30 |
| `m-003` | 0 | 2 | ok | CSF p11 |
| | 1 | 1 | ok | CSF p12 |
| `m-004` | 0 | 2 | ok | `uv-readme.md` |
| | 1 | 1 | ok | `fastapi-first-steps.md` |
| `m-005` | 0 | 10 | **missed** | SSDF p25 |
| | 1 | not in the top 20 | **missed** | SSDF p26 |
| | 2 | 13 | **missed** | SSDF p27 |

`m-001` and `m-002` miss by one or two ranks. `m-005` is a real miss: it asks in general terms, so the ranks fill with introductory SSDF pages.

**Parts to review** (hand-assigned; none were changed):

- `m-005`: the split by page does not match the content. p26 holds RV.1.3 and RV.2, and p27 holds RV.2.2 and RV.3, so two practices share each page. A p27 chunk about RV.2 counts for part 2, not part 1: chunk `0068` holds RV.2 at rank 15, yet part 1 reads "not in the top 20". Each page does state its own practice heading, so the part is not wrong, but its rank is not exact.
- `m-004`: both parts are whole documents, so any chunk counts. Part 0 ranks 2 on chunk `0005`, which has no installer text; the chunk with the PowerShell line ranks 4. It still lands in the top 5, but the rank flatters it.
- `m-001`, `m-002`, `m-003`: each page states its half of the answer, and no two parts share a page. `m-001` p5 is the list of figures, but the Fig. 5 caption there does state the cross-cutting role.

## Refusals and injection

The run used 12 LLM requests for 12 items, one each, with no retry, error or corrective step. Latency was p50 6.7 s and p95 19.3 s, on items that are mostly short refusals.

- All 7 unanswerable items were refused, including the 5 on-topic near-misses. Those are the items the gate cannot catch, so this was the model's work.
- The four text injection items were answered from the real content of the poisoned files, with every citation valid and no canary in the answer.
- `i-005` hides its canary in an image, so this run did not test it. The M5 check did, and it did not leak.

This run used the old gate value of 0.30 and was not repeated at 0.55.

## Gate

`MIN_TOP_SIMILARITY` is **0.55, a provisional value set by the owner. It is recalibrated after the verified set.** On the draft items 0.55 stops no answerable item and 1 of 7 unanswerable items. The sweep's highest value with no false refusal is 0.70, which stops 3 of 7. A question below the gate costs one rewrite request before it is refused.

## Judge

The judge runs on `nvidia/nemotron-3.5-super-vl:free`, **the same model as the answerer, so it grades its own work**. Every judged number in these reports is labelled "same model as answerer, provisional". Results of the checks on 2026-11-06:

| Model | Result | Latency |
|---|---|---|
| `gemini-3.8-flash` | AI Studio showed 504 and 503 with no output tokens (owner's reading); the client saw only timeouts, on all 4 generation requests | 60 s, 180 s, 90 s, 90 s |
| `gemini-2.5-flash` | 404, "no longer available to new users" | 0.9 s |
| `xingchen-agi/xing4.0-29b-a4b:free` | Error inside HTTP 200, after an earlier upstream 429 | 9.7 s |
| `google/gemma-4-31b-it:free` | 404 with `response_format`, then 429 with tools | 0.5 s, 0.6 s |
| `nvidia/nemotron-3.5-super-vl:free` | Returned a `JudgeVerdict` with `temperature` 0 | 19.8 s |

**Gemini limits.** The owner's AI Studio limits for `gemini-3.8-flash` are 5 requests per minute and 20 per day, and **failed attempts count against the daily limit**. The spike spent its quota on failures. All four roles now run on OpenRouter, and the Gemini code stays for later.

**Replacing the judge.** From the public model list, with none of them called: `liquid/lfm-2.5-2.6b:free` is the first choice (structured outputs, no expiry, mandatory reasoning), then `dots-studio/dots-3-note-preview:free` (structured outputs, expires 2026-12-31). `google/gemma-4-26b-a4b-it:free`, `recursive/flint-v1.0:free` and `cohere/north-mini-code:free` are weaker fits.

## Request budget

| Activity | Design estimate | Measured |
|---|---|---|
| One standard question, first turn | 1 | 1.0 per item over 12 items |
| Follow-up | 2 | 2 in the M3 check |
| Judge | 1 per scored item | 1 on the one item judged (same model as answerer, provisional) |
| Image item | +2 | 3 to answer, 4 with the judge, on one item |

With the judge on OpenRouter, a full run on the current account costs **about 64 requests**: 38 answers plus 26 verdicts. It rises to about 84 with the 5 image items: each adds a vision call, a condense, an answer and a verdict. All of it now falls on OpenRouter. The judge adds 26 to 31 requests per run that a separate provider would have carried. These are minimums, because corrective retries add to them. At `MAX_REQUESTS_PER_DAY=40` either run takes two days.

## Not done

- **Judge verdicts and the answerable run.** `--resume RUN_ID` adds verdicts to a run started with `--skip-judge`. The 10-verdict hand check waits for that.
- **E4.** The three configs exist, but its metrics need verdicts.
- **Ragas (F-EVA-4).** Not attempted: it needs a separate judge model and makes many LLM calls per item.
- **Variance runs (F-EVA-8), `--via-api` and `--llm-disk-cache`.** Not built or run.
