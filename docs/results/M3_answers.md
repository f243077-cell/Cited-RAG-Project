# M3 answer check (PROVISIONAL)

**PROVISIONAL:** four live questions that show the path works, not an evaluation. The relevance gate was the uncalibrated placeholder 0.30.

Date: 2026-11-06. Model for `fast` and `strong`: `nvidia/nemotron-3.5-super-vl:free` through OpenRouter. Prompt version 1. Index: `data/index`, 6 documents, 290 chunks. Rerank off.

These four requests show the path works. They are not an evaluation, and `MIN_TOP_SIMILARITY` is still the placeholder 0.30.

| Question | Result | Valid citations | LLM requests | Output tokens | Total |
|---|---|---|---|---|---|
| What are the four functions of the AI RMF Core? | answered | 1 of 1 | 1 | 418 | 10.9 s |
| Which of them is cross-cutting? (follow-up) | answered | 2 of 2 | 2 | 1,042 | 17.5 s |
| What is the capital of France? (same session) | refused | none | 2 | 346 | 5.4 s |
| what is fastapi? (typed in the browser) | answered | 2 of 2 | 1 | 704 | 21.0 s |

The first three ran through `POST /v1/chat`; the fourth ran through the Streamlit Chat page. No call failed or retried, and no answer needed the corrective regeneration.

- **Follow-up:** the condense call turned the message into "Which of the four functions of the AI RMF Core (GOVERN, MAP, MEASURE, and MANAGE) is cross-cutting?", and page 25 and page 27 were cited.
- **Refusal:** the off-topic question scored cosine 0.411, above the placeholder gate, so the refusal came from the model, not from the gate.
- **Latency:** nearly all time is the LLM. Retrieval took 44 to 168 ms and generation 3.9 to 19.8 s. NF-1 asks for a p50 of at most 15 s on a cache miss. Four requests are too few to change that target, so M4 should set it from the full run.
- **Not measured yet:** `lat_rerank` stays 0 because `retrieve()` includes the rerank; with rerank on, its time lands in `lat_retrieve`.
