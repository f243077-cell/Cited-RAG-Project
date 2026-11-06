"""One real cited answer through OpenRouter: uv run pytest -m live. Uses one to three free-tier requests."""

import asyncio

import pytest

from citerag.analytics.logs import LogStore
from citerag.config import get_settings
from citerag.llm.gateway import OpenRouterGateway
from citerag.rag.generate import generate
from citerag.rag.verify import Passage, needs_retry, verify_citations
from citerag.schemas import LLMCitedAnswer

TEXT = "Items can be returned within thirty days of delivery if they are unused."


@pytest.mark.live
def test_one_real_cited_answer(tmp_path):
    cfg = get_settings().model_copy(update={"data_dir": tmp_path, "llm_max_retries": 1})
    gateway = OpenRouterGateway(cfg, LogStore(cfg.logs_db_path))
    passage = Passage(marker=1, chunk_id="c", doc_id="d", doc_name="policy.md", page=None, text=TEXT)
    result = asyncio.run(generate(gateway, [passage], "How long do I have to return an item?", "live"))
    assert isinstance(result.parsed, LLMCitedAnswer) and result.parsed.answerable
    citations, _ = verify_citations(result.parsed.citations, {1: passage})
    assert not needs_retry(True, result.parsed.answer, citations)
