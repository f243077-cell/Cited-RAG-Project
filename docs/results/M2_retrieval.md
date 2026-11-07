# M2 retrieval report (PROVISIONAL)

**PROVISIONAL:** every number below comes from golden items no human has verified.

Date: 2026-11-06. LLM requests: 0.

```bash
uv run citerag ingest data/sample --index-dir data/index_eval
uv run citerag ingest eval/fixtures/poisoned --index-dir data/index_eval
uv run citerag eval retrieval --rerank --skip-type image --index-dir data/index_eval
```

**These numbers are provisional.** The 43 items in `eval/golden/candidates.jsonl` are drafts with `verified: false`. No human has checked them, so they are not a golden set and must not be quoted as results. The questions were also written from the text they point to, which flatters any retriever.

Setup: 11 files (10 with text, plus the whiteboard PNG with no chunks until M5), 294 chunks, `BAAI/bge-small-en-v1.5`, chunk size 512, overlap 64, `TOP_K_RETRIEVE` 20, `TOP_K_CONTEXT` 5.

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
| hit@3 | 22/26 (85%) | 24/26 (92%) |
| hit@5 | 24/26 (92%) | 25/26 (96%) |
| hit@10 | 25/26 (96%) | not applicable |
| hit@20 | 25/26 (96%) | not applicable |
| MRR | 0.782 (@10) | 0.796 (@5) |

Rerank looks useful at hit@3 and hit@5, but OPEN-2 waits for experiment E3 on the verified set.

Weakest items: `a-017` is missed even at hit@20, and the chunk holding the `a-007` answer ranks 4. See the diagnosis below.

For `multi_chunk` items, hit@k counts any one gold source, so a hit does not prove every part was retrieved. M4 should score those items for full coverage.

## Diagnosis of a-007 and a-017

```bash
uv run citerag eval explain a-007 --index-dir data/index_eval
uv run citerag eval explain a-017 --index-dir data/index_eval
```

Vector and BM25 rank the whole index here, so a rank past 20 is visible. Hybrid is the real fused list, cut to `TOP_K_RETRIEVE` (20). BM25 lists only chunks that match a query term. No retrieval setting was changed.

| Item | Chunk | Holds the answer | Vector | BM25 | Hybrid | After rerank |
|---|---|---|---|---|---|---|
| a-007 | p11 `0012` (gold) | yes | 25 of 294 | 4 of 214 | 14 of 20 | 3 of 5 |
| a-007 | p3 `0003` (gold) | yes | 12 of 294 | 1 of 214 | 4 of 20 | not recorded |
| a-017 | p18 `0035` (gold) | no | 68 of 294 | 62 of 155 | not found | not found |
| a-017 | p18 `0036` (gold) | yes | 137 of 294 | 19 of 155 | not found | not found |

Extraction is not the cause for either item. Both answers come through intact. On a-017, pypdf flattens the four-column table, so chunk `0036` runs from the end of the PO.5 row into PS.1, but the PS.1 text is whole.

**a-007 is mostly a gold-label error.**

- Page 3 also states the answer: "Supersedes NIST SP 800-61r2 (August 2012)". The earlier audit searched for one exact phrase and missed it. Page 3 was added to the gold sources after this diagnosis.
- Adding page 3 moved hybrid from hit@3 21/26, hit@5 23/26, hit@10 24/26 and MRR@10 0.763 to the scores above. Rerank scores did not change.
- The new page-level rank of 2 is still too kind. It comes from chunk `0002`, which only says "superseding the existing authorities". The chunk with the answer ranks 4.
- Vector search prefers front matter because the question contains the document number. The top cosine is 0.840, against 0.727 for `0012`. The number `61r3` appears in 49 of the 72 chunks of this document.
- `0012` misses the vector top 20 by a little (rank 20 scores 0.736), so fusion credits it from BM25 only. That is why BM25 rank 4 becomes hybrid rank 14.

**a-017 is a real retrieval miss, caused by the words in the question.**

- The BM25 tokenizer (`(?u)\b\w\w+\b`) drops the "1" from `PS.1`, leaving `ps`. It cannot tell PS.1 from PS.2 or PS.3.
- Of the query tokens `what`, `purpos`, `ssdf`, `practic` and `ps`, chunk `0036` contains only `ps`.
- `SSDF` is in the running header, and `0036` starts mid-page, so it has no header. The term helps 41 of the other 84 chunks in this document instead.
- Vector cosine is 0.571. Rank 20 needs 0.658.
- BM25 rank 19 is inside its top 20, but one list's credit is not enough to stay in the fused top 20. Rerank only sees those 20, so it cannot recover the chunk.
- The chunk is findable. Its title as the query, "Protect All Forms of Code from Unauthorized Access and Tampering", ranks it vector 2, BM25 1 and hybrid 1. "What is PS.1?" gets hybrid 15.

**Smaller chunks alone would not fix either item.** Each answer sentence embedded alone scores 0.740 (a-007) and 0.609 (a-017), up only slightly from 0.727 and 0.571. That would barely reach rank 20 on a-007 and not reach it on a-017. This compares one sentence against today's index, so it is an estimate.

**Page-level scoring can overstate a hit.** A chunk counts as a hit if it shares a page with the answer, as chunks `0002` and `0035` show. M4 should keep this in mind when reading hit@k.

Ideas to test at M4 on the verified set, none tested yet: a BM25 token pattern that keeps identifiers such as `PS.1`, and removing running headers or adding section context to chunks.

## Gold page audit

```bash
uv run python scripts/audit_gold_pages.py --index-dir data/index_eval
```

The script lists pages in a gold document that share the gold answer's key terms but are not in `gold_sources`. It first listed 51 pages across 13 items for human review. After that review, 9 of those pages were added to the gold sources: `a-001` p5, p8, p39; `a-015` p31; `a-016` p7, p8, p10, p19, p20. The script now lists 42 pages across 13 items. A listed page may only mention the terms.

The added pages changed no retrieval score, because all three items already ranked 1 on a gold page.

Tested by dropping each gold page in turn, on items with two or more gold pages: it listed 34 of 38 again. The 4 misses are all `multi_chunk` items (`m-001` three times, `m-003` once), because a whole-answer term match is weak when each page holds only part of the answer. A short list for those items does not prove their pages are complete.

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
