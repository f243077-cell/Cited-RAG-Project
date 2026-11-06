# Baseline (provisional)

Date: 2026-11-06. **PROVISIONAL: this is not the M4 baseline.** The exit criterion needs a human-verified golden set, and all 43 items are still drafts with `verified: false`, so no number here may be quoted as a result. No answerable item has been judged yet, so faithfulness and correctness have no value.

Setup: `nvidia/nemotron-3.5-super-vl:free` for fast, strong and judge, prompt version 1, `data/index_eval`, hybrid search, rerank off. Image items and the image half of `i-005` wait for M5.

```bash
uv run citerag eval run --golden eval/golden/candidates.jsonl --index-dir data/index_eval --offline --rerank
uv run citerag eval run --golden eval/golden/candidates.jsonl --index-dir data/index_eval --skip-judge --max-requests 16 --skip-type answerable --skip-type multi_chunk
```

## Against the targets

| Goal | Metric | Target | Now | State |
|---|---|---|---|---|
| G1 | Faithfulness (judge) | at least 0.90 | not measured | Needs the answerable run |
| G1 | Citation validity (verifier) | at least 0.95 | not measured on the set | Needs the answerable run |
| G1 | Answer correct (judge) | at least 0.75 | not measured | Needs the answerable run |
| G2 | Correct refusals | at least 0.90 | 7 of 7 (100%) | Provisional, measured at gate 0.30 |
| G2 | Injection canary leaks | 0 | 0 of 5 | Provisional; one canary sits in an image |
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

## Refusals and injection

The run used 12 LLM requests for 12 items, one each, with no retry, error or corrective step. Latency was p50 6.7 s and p95 19.3 s, on items that are mostly short refusals.

- All 7 unanswerable items were refused, including the 5 on-topic near-misses. Those are the items the gate cannot catch, so this was the model's work.
- The four text injection items were answered from the real content of the poisoned files, with every citation valid and no canary in the answer.
- `i-005` hides its canary in an image, so this run did not test it: the image has no text chunks until M5.

This run used the old gate value of 0.30 and was not repeated at 0.55.

## Gate

`MIN_TOP_SIMILARITY` is **0.55, a provisional value set by the owner. It is recalibrated after the verified set.** On the draft items 0.55 stops no answerable item and 1 of 7 unanswerable items. The sweep's highest value with no false refusal is 0.70, which stops 3 of 7. A question below the gate costs one rewrite request before it is refused.

## Judge

The judge currently runs on `nvidia/nemotron-3.5-super-vl:free`, **the same model as the answerer, so it grades its own work**. Treat its verdicts as a stopgap. It has not yet returned a `JudgeVerdict`. Results of the checks on 2026-11-06:

| Model | Result | Latency |
|---|---|---|
| `gemini-3.8-flash` | Timed out on all 3 generation requests | 60 s, 180 s, 90 s |
| `gemini-2.5-flash` | 404, "no longer available to new users" | 0.9 s |
| `xingchen-agi/xing4.0-29b-a4b:free` | Error inside HTTP 200 | 9.7 s |
| `google/gemma-4-31b-it:free` | 404 with `response_format`, then 429 with tools | 0.5 s, 0.6 s |

## Request budget

| Activity | Design estimate | Measured |
|---|---|---|
| One standard question, first turn | 1 | 1.0 per item over 12 items |
| Follow-up | 2 | 2 in the M3 check |
| Judge | 1 per scored item | not measured |

With the judge on OpenRouter, a full run on the current account costs **about 64 requests**: 38 answers plus 26 verdicts. It rises to about 74 with the 5 image items, plus 5 vision calls. The judge adds 26 to 31 requests per run that a separate provider would have carried. These are minimums, because corrective retries add to them. At `MAX_REQUESTS_PER_DAY=40` either run takes two days.

## Not done

- **Judge verdicts and the answerable run.** `--resume RUN_ID` adds verdicts to a run started with `--skip-judge`. The 10-verdict hand check waits for that.
- **E4.** The three configs exist, but its metrics need verdicts.
- **Ragas (F-EVA-4).** Not attempted: it needs a separate judge model and makes many LLM calls per item.
- **Variance runs (F-EVA-8), `--via-api` and `--llm-disk-cache`.** Not built or run.
