# M2 retrieval report

Date: 2026-11-06. Command: `uv run citerag eval retrieval --rerank`. LLM requests: 0.

**These numbers are provisional.** The 12 items in `eval/golden/candidates.jsonl` are drafts with `verified: false`. No human has checked them yet, so they are not a golden set and must not be quoted as results. The questions were also written from the text they point to, which flatters any retriever.

Setup: 6 documents, 290 chunks, `BAAI/bge-small-en-v1.5`, chunk size 512, overlap 64, `TOP_K_RETRIEVE` 20, `TOP_K_CONTEXT` 5.

| Metric (n=10) | Hybrid | After rerank (20 to 5) |
|---|---|---|
| hit@1 | 6/10 | 8/10 |
| hit@3 | 7/10 | 9/10 |
| hit@5 | 9/10 | 10/10 |
| hit@10 | 10/10 | not applicable |
| hit@20 | 10/10 | not applicable |
| MRR | 0.688 (@10) | 0.853 (@5) |

With about 10 items, one item is 10 points, so this shows nothing firm about rerank. OPEN-2 waits for experiment E3 on the verified set.

## Relevance gate

| Item type | Top cosine |
|---|---|
| Answerable (10) | 0.720 to 0.883 |
| Unanswerable (2) | 0.510 and 0.540 |

`MIN_TOP_SIMILARITY` stays at the 0.30 placeholder, so neither unanswerable item would be refused by the gate today. M4 calibrates it on the verified unanswerable items.

## For the human review

- **a-004** ranks 5 because the top hit, page 31 of `NIST.CSWP.29.pdf`, is the glossary, which also mentions Tiers. Check whether page 31 belongs in its gold sources.
- **a-007** ranks 7: page 1 of `NIST.SP.800-61r3.pdf` matches the title but does not answer the question.
- Move an item to `eval/golden/golden.jsonl` with `verified: true` only after checking its question, answer and sources.
