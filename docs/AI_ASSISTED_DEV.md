# AI-assisted development log

Roadmap item 3.4. How CiteRAG was built with Claude Code as the implementer and the owner as reviewer and decision-maker, written
from the git history (86 commits through M11, on 2026-11-06 and 2026-11-07), `docs/results/PROGRESS.md` and the gotchas in
`CLAUDE.md`. It records what was delegated, what the assistant got wrong, and the prompts that worked.

## How the work was split

| Owner | Claude Code |
|---|---|
| Wrote the PRD and architecture with tested reference code, and set the rules in `CLAUDE.md` | Read the docs first, then built one milestone at a time |
| Approved dependencies, schema changes and the provider; chose models after spikes | Checked package APIs, licences and model lists on the installed packages or the live list, never from memory |
| Owns the golden set and the cache labels: verified items and pairs come only from them | Drafted candidates with `verified: false` or `labelled: false`, plus review scripts to label them |
| Rationed live LLM requests (40 a day) | Spent requests only when a decision depended on one, with a cap and the count reported |
| From Phase 3, delegated design decisions for a whole phase | Recorded each decision and its reason in the phase PRD |

## Per milestone: what was delegated, and what went wrong

| Milestone | Delegated | What the assistant got wrong, and how it was caught |
|---|---|---|
| Preflight | Spikes against LlamaIndex, Chroma, CrewAI and OpenRouter | Nothing shipped wrong; the spikes produced most of the early gotchas, such as Chroma scores being `exp(-distance)` and CrewAI over-counting a shared `LLM` |
| M0 to M2 | Scaffold, ingestion, hybrid retrieval, gate, hit@k | BM25 padding its list with score-0 chunks was found only by an `eval explain` on a bad rank |
| M3 | Cited answers, verifier, chat UI | `_storable` cached answers whose citations were valid but whose text had an unbacked `[n]`. Caught by a failing test, then fixed |
| M4 | Evaluation harness, golden candidates | Printed `hit@5: 0.0%` when nothing was scored. Wrote tests with wrong expectations (`or True`, a cap off by one) |
| M5 | Images and a second provider | Chose `gemini-3.8-flash` from the listing, and it never answered; 5 requests spent. Logged a missing key as an attempt. Claimed 21 captions and 8 saved where the logs said 20 and 9 |
| M6 | Semantic cache, sweep, benchmark, dashboard | `cache or NoOpCache()` replaced an **empty** cache, because `SemanticCache` defines `__len__`, so uploads stopped invalidating it. Caught by its own integration test. One commit message left out half of its content |
| M7 | CrewAI Deep mode and evaluation crews | A loop over exception causes hung forever on CrewAI's circular chains. Missed that importing `crewai` loads `.env` into the process, which broke an unrelated test. Missed two hidden CrewAI retries |
| M8 | Docker, README, security checklist | Wrote Docker files while the daemon was down; run later, an `env_file` quirk turned a comment into a setting. A fresh clone found an NLTK crash on hard-linked files that the repo on D: never showed. Overstated in `PROGRESS.md` what the clone test covered, then corrected it |
| M9 | Read-only tools and the MCP server | Nothing shipped wrong; a stdio subprocess test was written first to catch a stray line on stdout |
| M10 | LangGraph agent | A failed native tool call also switched the model's structured-output method for every schema, caught by a test. The first design trusted the model's first tool choice; a 3-request spike showed it called `list_documents` instead of searching |
| M11 to M12 | E8, supervisor, this log | Wrong arithmetic in the expected values of its own estimate tests |

**Patterns in the mistakes.**

- **Python truthiness and hidden framework behaviour** caused the costliest bugs: the empty cache, CrewAI's retries and `.env` loading, and the circular exception chains. Integration tests with the real class found them; mocks would not have.
- **The environment differed from the developer's.** The NLTK hard-link crash, the Docker `env_file` parsing and the shell mangling `\\n` in patch scripts all hid until something ran in a fresh place.
- **The assistant's own tests were sometimes wrong.** Each failing test was read before the code was changed: about one in five failures was the test.
- **Claims drifted from evidence** in summaries (counts, coverage of a check). The fix was to re-read the logs or files before stating a number.

## Prompts that worked

1. **A hard quota rule at the top of the request:** "No live LLM requests in this run", or a stated cap. Every expensive step then came back as a command, an estimate and a `--max-requests` cap instead of being run.
2. **A do-not list with an escape hatch:** "Do not change a public schema or add a dependency. If one is needed, write it under Decisions for Coder and carry on." Work kept moving, and nothing risky happened silently.
3. **"Show a failing test and ask"** for reference code. The assistant never "improved" the provided cache and verifier code; it wrapped them (`PersistentCache`) instead.
4. **A test guide per milestone:** exact commands, the expected result, the request cost with a no-request alternative, and what counts as failure. It forced each feature to be checkable by someone else.
5. **A progress file kept up to date after each milestone**, so work resumed cleanly after the context ran out.
6. **"If something fails twice with the same approach, stop and report."** It stopped retry loops on free models, which cost quota.
7. **Owner-only judgements stated as rules:** "Do not create or mark verified any golden item or cache pair." The assistant drafted 80 cache pairs and golden candidates but never labelled one.
8. **Delegating a whole phase** once the rules were in place: "take the decisions you think best and tell me when the phase is completed".

## Prompts that needed a second pass

- Requests that bundled a design change mid-task, such as adding Gemini during M3, cost a re-plan. A separate message works better.
- "Plan only" requests should still allow reading installed packages. Doing so in Phase 3 showed that `langgraph` and `mcp` were already transitive dependencies, which changed the plan.
