# Results and guides: index

Every file in `docs/results/`, plus the design documents they rest on. **PROVISIONAL** marks a number that rests on the
unverified golden set, the same-model judge or an unvalidated threshold.

## Start here

| File | What it is |
|---|---|
| [HANDOFF.md](HANDOFF.md) | The steps only the owner can do, in order, with cost and failure criteria |
| [WINDUP_AUDIT.md](WINDUP_AUDIT.md) | Every requirement ID, M0 to M12: done, partial or missing, with a check command |
| [WINDUP_SMOKE.md](WINDUP_SMOKE.md) | The live smoke test at the wind-up, and the clean-clone proof |
| [DECISIONS.md](DECISIONS.md) | Decisions taken at the wind-up, with how to reverse each |
| [PROGRESS.md](PROGRESS.md) | Milestone status from M6 on, and the decisions taken along the way |
| [security_checklist.md](security_checklist.md) | ARCHITECTURE section 18, item by item, with open gaps |

## Results

| File | Milestone | State |
|---|---|---|
| [M2_retrieval.md](M2_retrieval.md) | M2 hit@k on the first draft set | PROVISIONAL: unverified items |
| [M3_answers.md](M3_answers.md) | M3 four live answers | PROVISIONAL: a path check, not an evaluation |
| [baseline.md](baseline.md) | M4 baseline | PROVISIONAL: unverified items, same-model judge |
| [experiments_retrieval.md](experiments_retrieval.md) | E1 to E3 retrieval experiments | PROVISIONAL: unverified items |
| [M5_images.md](M5_images.md) | M5 image check | PROVISIONAL: two draft items, same-model judge |
| Charts: [e1_chunking.png](e1_chunking.png), [e2_hybrid.png](e2_hybrid.png), [e3_rerank.png](e3_rerank.png) | E1 to E3 | PROVISIONAL |

Not yet written, because they need the verified set or labelled pairs: `E5` (threshold sweep chart), `E6_cache_benchmark.md`,
`E8_modes.md`, `M12_multi_agent.md`, `failure_analysis.md`.

## Test guides, one per milestone from M6

| Guide | Milestone |
|---|---|
| [M6_test_guide.md](M6_test_guide.md) | Semantic cache |
| [M7_test_guide.md](M7_test_guide.md) | CrewAI Deep mode and evaluation crews |
| [M8_test_guide.md](M8_test_guide.md) | Docker, request guard, README |
| [M9_test_guide.md](M9_test_guide.md) | Read-only tools and the MCP server |
| [M10_test_guide.md](M10_test_guide.md) | LangGraph agent and Agent mode |
| [M11_test_guide.md](M11_test_guide.md) | Evaluation E8 |
| [M12_test_guide.md](M12_test_guide.md) | Supervisor, comparison, AI-assisted development |

## Design and history

| File | What it is |
|---|---|
| [../PRD.md](../PRD.md), [../ARCHITECTURE.md](../ARCHITECTURE.md) | M0 to M8: requirements and design, with the tested reference code |
| [../PHASE3_PRD.md](../PHASE3_PRD.md), [../PHASE3_ARCHITECTURE.md](../PHASE3_ARCHITECTURE.md) | M9 to M12: agent mode, tools, MCP |
| [../AI_ASSISTED_DEV.md](../AI_ASSISTED_DEV.md) | How the project was built with Claude Code: what went wrong, the prompts that worked |
| [../PREFLIGHT_REPORT.md](../PREFLIGHT_REPORT.md) | The checks before M0 |
| [../../CLAUDE.md](../../CLAUDE.md) | Working rules, commands and every gotcha found |
