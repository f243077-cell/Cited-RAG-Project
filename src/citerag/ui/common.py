"""Shared Streamlit pieces. Output is plain Markdown only, never HTML (F-SAF-2)."""

from __future__ import annotations

import streamlit as st

from citerag.config import UISettings
from citerag.schemas import ChatResponse, Citation
from citerag.ui.api_client import ApiClient


@st.cache_resource
def get_client() -> ApiClient:
    cfg = UISettings()
    key = cfg.citerag_api_key.get_secret_value() if cfg.citerag_api_key else None
    return ApiClient(cfg.api_base_url, key)


def citation_label(c: Citation) -> str:
    state = ":material/check_circle:" if c.valid else ":material/warning:"
    page = "" if c.page is None else f", p{c.page}"
    return f"{state} [{c.marker}] {c.doc_name}{page}"


def usage_line(response: ChatResponse) -> str:
    u = response.usage
    seconds = u.latency_ms.total / 1000
    return (
        f"{u.input_tokens} tokens in, {u.output_tokens} out · {u.llm_requests} LLM requests · "
        f"\\${u.est_cost_usd:.4f} · {seconds:.1f} s"
    )


def render_answer(response: ChatResponse) -> None:
    st.markdown(response.answer.answer)
    for warning in response.warnings:
        st.warning(warning)
    for citation in response.answer.citations:
        with st.expander(citation_label(citation)):
            st.text(citation.quote)
            if not citation.valid:
                st.caption(f"Quote not found in the passage (match {citation.match_score:.0f} of 100)")
    if response.answer.answerable:
        st.caption(f"Confidence: {response.answer.confidence}")
    st.caption(usage_line(response))
