# Preflight prompt for Claude Code

Paste everything below the line into Claude Code, started in the repository root, after `cp .env.example .env` and adding `OPENROUTER_API_KEY`.

---

You are setting up and hardening the CiteRAG project. Read CLAUDE.md, docs/PRD.md and docs/ARCHITECTURE.md completely first. Follow CLAUDE.md throughout (framework boundaries, free-tier rules, no co-author or "generated with" attribution anywhere, targeted edits, ask before adding dependencies or changing boundaries).

REPOSITORY
This folder is a clone of https://github.com/f243077-cell/Cited-RAG-Project.git (remote "origin"). Run `git status`, `git log --oneline -5` and `git remote -v` first. If the repository already contains files that are not part of this starter kit (a README, a LICENSE, anything else), do not overwrite them: keep them, merge sensibly (for example append a short project section to an existing README), and list what you kept. Confirm that .env, data/ and *.sqlite are git-ignored before any commit.

GIT (explicit permission from me for this run only)
Work on a new branch named setup/preflight-m0, never on main. Make small Conventional Commits, in this order: (1) the starter kit as provided, (2) ruff-only fixes if any, (3) pricing, model and doc updates from the preflight, (4) preflight scripts and docs/PREFLIGHT_REPORT.md, (5) the sample-corpus script, (6) M0. Never add Co-Authored-By, "Generated with" or session links to any commit message, PR text, file or code comment. After M0 passes, push the branch with `git push -u origin setup/preflight-m0` and tell me the branch name. Do not merge, do not open a pull request, do not push to main, do not force-push. If the push fails (authentication or permissions), stop, report the exact error, and leave the commits local.

GOAL
Run a preflight that verifies everything the docs mark "not verified" or "confirm", fix what you can yourself, fetch what needs downloading, then build Milestone M0 only and stop. Work autonomously. Stop and ask me only when a fix would need a new dependency, a change to a framework boundary, a change to a public schema, or a change to the meaning of the provided reference code. Use a task list. Do the phases in order.

QUOTA RULES (this project uses OpenRouter free models)
The free tier allows 20 requests per minute and 50 per day (1,000 per day once the account has bought at least 10 credits). Every HTTP attempt counts, retries included. During this whole preflight, send at most 20 LLM requests in total, count them yourself, and print the running count in the report. Never retry in a loop. If a call returns 429, 402 or a daily-limit error, stop all live checks, mark the remaining live items "skipped: quota", and carry on with the local ones. Never print or log the API key. Never use the random `openrouter/free` model.

PHASE 0: ENVIRONMENT
1. Detect the OS and shell. Check that Python 3.11+ and uv are installed; if uv is missing, install it with the official method for this OS and tell me what you ran.
2. Run `uv sync --extra dev`. Fix problems you can fix. Install no llama-index LLM package, no `anthropic` and no `litellm`. If a dependency cannot be resolved, stop and ask.
3. Run `uv run pytest` (expect 36 passed), `uv run ruff check .` and `uv run ruff format --check .` (both expected clean). If any fails, make the smallest edit that fixes it without changing behaviour, then confirm all tests still pass. List every edit.
4. Check that .env exists and OPENROUTER_API_KEY is set (never print it). If missing, skip every live step below, mark them "skipped: no key", and continue with the local ones.

PHASE 1: ACCOUNT, LIMITS, MODELS, PRICING
5. Fetch https://openrouter.ai/docs/api-reference/limits and confirm the free-tier numbers and the credit threshold quoted in docs/ARCHITECTURE.md section 11. If they differ, fix the docs and .env.example (LLM_MAX_RPM, MAX_REQUESTS_PER_DAY) and tell me.
6. If OpenRouter offers an endpoint that reports key status or credits (check the API docs; for example a key-info endpoint), call it once and tell me my tier and remaining limits without printing the key. Count it as a request if it is one.
7. Write scripts/pick_free_models.py (PRD F-OPS-7): download the public model list (no key needed), keep models whose prompt and completion prices are both zero, read the architecture input modalities and supported parameters, print a table and suggested .env lines for the four roles (fast, strong, vision, judge). The judge should be a different model from the answerer when possible. Check the field names against one real response first and correct ARCHITECTURE section 11 if they differ. Add a unit test for the filtering with a small synthetic list.
8. Run the script and put its picks into .env (not .env.example). Compare them with the placeholder IDs in .env.example; if the placeholders no longer exist or no longer support what the roles need, update .env.example and docs/ARCHITECTURE.md section 11.
9. Shadow pricing: choose one paid reference model on OpenRouter, copy its input and output price per million tokens from OpenRouter's official model page into configs/pricing.yaml, set retrieved_on, source_url and reference_model. Add the loader that exposes it as pricing["_reference"]. Never use numbers from memory.

PHASE 2: SPIKES (throwaway scripts in scripts/preflight/, one per item; each prints PASS or FAIL plus evidence; the live ones count against the 20-request cap)
10. LIVE, about 2 requests: with the chosen strong model, run one structured call through the json_schema path (`llm.bind(response_format=schema_response_format(S))`, `parse_model`) and, if that fails or is unsupported, one through the tool-calling path (`bind_tools`). Record which methods work for each chosen model, the real token usage fields, and the exact key that holds the model that answered. Confirm that `extra_body` provider preferences (require_parameters) and the optional headers are accepted. If `require_parameters: true` yields a 503 for a model, record it and say what you recommend.
11. LIVE, 1 request: image payload with the chosen vision model: send a small generated PNG using the standard LangChain image block and get a structured ImageAnalysis back. Record whether image tokens appear in the usage.
12. LIVE, 1 request: test whether the chosen models accept `temperature` (send 0 once on the judge model). Record the result; the rule stays "send no sampling parameters unless configured".
13. LIVE, 1 to 3 requests: CrewAI with the real service: one agent and one task with `LLM(model="openrouter/<chosen id>")` and `output_pydantic`, telemetry disabled through the env vars. Confirm `CrewOutput.token_usage` against the number of requests you actually sent. Do not run the 3-task crew live (it was verified against a fake server and costs 3 requests).
14. LOCAL: download BAAI/bge-small-en-v1.5 through llama-index-embeddings-huggingface, embed two sentences, confirm 384 dimensions and unit norm, and that identical text gives cosine about 1.0. Report download size and time.
15. LOCAL: Chroma and LlamaIndex: tiny index with the cosine collection, store_nodes_override=True and a persisted docstore; reload; BM25 after reload; hybrid QueryFusionRetriever(num_queries=1, use_async=False); a doc_ids filter with FilterOperator.IN (fallback: OR of equality filters); the true cosine gate via a direct Chroma query, confirming it differs from the LlamaIndex score as documented (score = exp(-distance)).
16. LOCAL: SentenceTransformerRerank with cross-encoder/ms-marco-MiniLM-L-6-v2 on 5 passages; report latency on this machine.
17. LOCAL: Streamlit: check what st.chat_input(accept_file=True, file_type=[...]) returns (type and attribute names) using streamlit.testing or a short headless run, and that st.pyplot works. Record the exact return shape.
18. LOCAL: if Docker is installed, run `docker --version` and note whether a build looks feasible; do not build the image yet.

PHASE 3: SAMPLE CORPUS (OPEN-1)
19. Create scripts/fetch_sample.py that downloads 5 to 8 public documents, about 100 to 200 pages in total, into data/sample/. Licences must allow redistribution (prefer US-government public-domain PDFs such as NIST publications, and MIT or CC-BY documentation); at least two PDFs with figures or screenshots; at least one Markdown file. Record each file's URL, licence and sha256 in data/sample/SOURCES.md. Skip any document whose licence is unclear. Run it and report page counts and the number of embedded images per PDF (the caption budget depends on it: every captioned image is one LLM request).

PHASE 4: FIX THE DOCS (targeted edits only, never rewrite whole files)
20. In docs/ARCHITECTURE.md section 3, move every item you verified from "not verified" to "verified by running" with the date; correct anything the spikes disproved. Update section 21 and the CLAUDE.md Gotchas list with every new finding (exact error messages, parameter names, return shapes, which free models passed which check). Make small PRD wording fixes only if a finding requires it.
21. Write docs/PREFLIGHT_REPORT.md: a table with columns item, result (PASS, FAIL, SKIPPED), evidence, what you changed, what still needs my decision. Add the number of LLM requests used, the OS and Python versions, the download sizes, and my free-tier status. Add a short recommendation on OPEN-5 (buy credits or stay at 50 requests per day) based on the request budget table in ARCHITECTURE section 13 and the image counts from step 19.

PHASE 5: BUILD M0 (PRD section 8)
22. Build Milestone M0 exactly as specified: repo layout from ARCHITECTURE section 5, config.py (pydantic-settings, every key in .env.example), errors.py, FakeGateway, a NoOpCache behind a CacheBackend interface, the /health endpoint (versions, manifest placeholder, namespace, cache stats, LLM requests in the last 24 hours and the limit), the prompts directory with versioned files for the prompts in ARCHITECTURE section 8, and a test that builds each role's client without sampling parameters. Keep the reference code and tests exactly as provided. Make `uv run pytest`, `uv run ruff check .` and `uv run ruff format --check .` pass. Start the API and confirm /health returns 200, then stop it. Do not implement the real gateway yet (that is M3).

RULES FOR THE WHOLE RUN
- Do not create or mark "verified" any golden-set item, cache pair or threshold; those are mine to do later.
- Do not change MIN_TOP_SIMILARITY or CACHE_THRESHOLD; they are calibrated in M4 and M6.
- Do not add dependencies without asking; if a spike needs one, report it instead.
- Never commit .env, data/ or *.sqlite.
- If something fails twice with the same approach, stop retrying, record it in the report, and move on.

FINAL MESSAGE
Give me: a summary table of the preflight results, the files you created or edited, the LLM requests used, the items that need my decision, the PRD requirement IDs covered by M0, and the pushed branch name. Then stop and wait for me to say "Build M1".
