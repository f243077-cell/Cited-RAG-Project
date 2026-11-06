# Evaluation fixtures

`poisoned/` holds made-up documents with prompt-injection text and canary strings (PRD F-SAF-1). Index them only into the evaluation index, never the demo index:

```bash
uv run citerag ingest data/sample --index-dir data/index_eval
uv run citerag ingest eval/fixtures/poisoned --index-dir data/index_eval
```

`images/` holds figures from NIST publications, attached to `image` items as if a user had uploaded them. Republished courtesy of the National Institute of Standards and Technology.

`uv run python scripts/make_eval_fixtures.py` rebuilds the PNG files. It needs `data/sample`.
