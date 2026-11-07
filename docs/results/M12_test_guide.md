# M12 test guide: supervisor, comparison and AI-assisted development

Run from the repo root.

| # | Do this | Expect | LLM requests |
|---|---|---|---|
| 1 | `uv run pytest tests/integration/test_supervisor.py` | All pass: one handoff costs 5 requests, the cap is 11, the guard needs room for 11, and the factory picks the architecture | 0 |
| 2 | Set `AGENT_ARCHITECTURE=supervisor`, restart the API, ask in Agent mode: "What are the CSF Tiers, and what does GOVERN do in the AI RMF?" | A cited answer. The trace starts with `supervise` ("research: ..."), shows the researcher's plan, tools and reflect, then `supervise` ("hand off to the writer") and synthesis | 5 usually, at most 11 |
| 3 | Set `AGENT_ARCHITECTURE=single` again and restart | Agent mode is back to the single agent | 0 |
| 4 | For each config, `uv run citerag eval run --golden eval/golden/golden.jsonl --require-verified --config configs/experiments/m12_supervisor.yaml --skip-type image --estimate`, also with `m12_single.yaml` and `e8_deep.yaml` | The estimate for each, with nothing run | 0 |
| 5 | Run the three, capped and resumed as in the M11 guide, then `uv run citerag eval report-e8 <single_run> <supervisor_run> <deep_run> --title "M12: single agent vs supervisor vs Deep" --out docs/results/M12_multi_agent.md` | A three-column table and chart, provisional while items are unverified | see the table below |
| 6 | Add a short conclusion under the table: on these items, does the supervisor buy coverage or correctness worth its requests and seconds? | Your answer to "when does multi-agent add value" | 0 |
| 7 | Read `docs/AI_ASSISTED_DEV.md` | Each milestone's delegated work, what went wrong, and the prompts that worked match your memory of the project | 0 |

**Cost of the comparison** on 15 items (the 5 `multi_chunk` and 10 answerable), with the judge:

| Run | Expected | Worst case |
|---|---|---|
| Single agent | about 60 | 135 |
| Supervisor | about 90 | 180 |
| Deep | about 60 | 90 |

The 15 items are reached with `--skip-type unanswerable --skip-type injection --skip-type image --limit 15`. If the E8 single-agent and
Deep runs used the same items and settings, only the supervisor run is new: about 90.

**No-request alternative:** step 1 runs the supervisor end to end on `FakeGateway`, and `tests/integration/test_e8.py` rehearses the
report on fakes.

**Failure means any of these:**

- More than 11 requests for one supervisor question, or a third research handoff.
- The supervisor's routing step shows passage text, or a canary appears in a task.
- A supervisor answer with an unverified citation, or one served from the cache.
- `AGENT_ARCHITECTURE=single` still routes through the supervisor.
