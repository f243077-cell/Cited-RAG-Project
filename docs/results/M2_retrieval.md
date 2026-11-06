# M2 retrieval report

Date: 2026-11-06. LLM requests: 0.

```bash
uv run citerag ingest data/sample --index-dir data/index_eval
uv run citerag ingest eval/fixtures/poisoned --index-dir data/index_eval
uv run citerag eval retrieval --rerank --skip-type image --index-dir data/index_eval
```

**These numbers are provisional.** The 43 items in `eval/golden/candidates.jsonl` are drafts with `verified: false`. No human has checked them, so they are not a golden set and must not be quoted as results. The questions were also written from the text they point to, which flatters any retriever.

Setup: 10 documents, 294 chunks, `BAAI/bge-small-en-v1.5`, chunk size 512, overlap 64, `TOP_K_RETRIEVE` 20, `TOP_K_CONTEXT` 5.

## Draft set

| Type | Items | Note |
|---|---|---|
| answerable | 21 | |
| multi_chunk | 5 | |
| unanswerable | 7 | 5 on-topic near-misses, 2 easy controls |
| injection | 5 | One canary is inside an image, so it needs M5 |
| image | 5 | Skipped here: they need the M5 image flow |

Every gold page was checked against the `page` and `doc_name` stored after ingestion, which is the PDF position from 1, not the printed number.

## Retrieval (answerable and multi_chunk, n=26)

| Metric | Hybrid | After rerank (20 to 5) |
|---|---|---|
| hit@1 | 18/26 (69%) | 18/26 (69%) |
| hit@3 | 21/26 (81%) | 24/26 (92%) |
| hit@5 | 23/26 (88%) | 25/26 (96%) |
| hit@10 | 24/26 (92%) | not applicable |
| hit@20 | 25/26 (96%) | not applicable |
| MRR | 0.763 (@10) | 0.796 (@5) |

Rerank looks useful at hit@3 and hit@5, but OPEN-2 waits for experiment E3 on the verified set.

Weakest items: `a-017` is missed even at hit@20, and `a-007` ranks 14, behind pages that only mention the title.

For `multi_chunk` items, hit@k counts any one gold source, so a hit does not prove every part was retrieved. M4 should score those items for full coverage.

## Relevance gate

| Item type | Top cosine |
|---|---|
| Answerable and multi_chunk (26) | 0.701 to 0.883 |
| Unanswerable near-misses (5) | 0.651 to 0.766 |
| Unanswerable easy controls (2) | 0.510 and 0.635 |

**The gate cannot separate near-misses from answerable items:** the ranges overlap. A threshold that refuses them all would also refuse real questions. Refusing them is the model's job (F-ANS-3), so M4 should calibrate the gate low and measure correct refusals on these items. `MIN_TOP_SIMILARITY` stays at the 0.30 placeholder.

`u-001` scores 0.635 here, up from 0.540 on the demo index, because it matches the poisoned shipping note.

## Review notes

- Move an item to `eval/golden/golden.jsonl` with `verified: true` only after checking its question, answer and sources.
- `a-012`, `a-016` and `v-001` list every page that states the answer, including glossary and list-of-figures pages.
- `a-005`, `v-005`: pages that merely name the same terms were left out.
