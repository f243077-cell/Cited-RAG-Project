"""Shared Streamlit pieces. Model output is plain Markdown or plain text, never HTML (F-SAF-2): answers, quotes and
trace steps never pass through `style`, which builds only the page chrome from fixed strings and escaped values."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from citerag.config import UISettings
from citerag.schemas import AgentStep, AgentTrace, ChatResponse, Citation, DocumentInfo
from citerag.ui import style
from citerag.ui.api_client import ApiClient, ApiError

_MARKER = re.compile(r"\[(\d{1,2})\]")


@st.cache_resource
def get_client() -> ApiClient:
    cfg = UISettings()
    key = cfg.citerag_api_key.get_secret_value() if cfg.citerag_api_key else None
    return ApiClient(cfg.api_base_url, key)


@st.cache_data(ttl=10, show_spinner=False)
def cached_health() -> dict | None:
    """/health for the sidebar card and the home page; None when the backend does not answer."""
    try:
        return get_client().health()
    except ApiError:
        return None


def count_kinds(documents: list[DocumentInfo]) -> str:
    """'5 PDF · 2 Markdown' from the file extensions."""
    names = {"pdf": "PDF", "md": "Markdown", "markdown": "Markdown", "txt": "Text", "png": "Image", "jpg": "Image",
             "jpeg": "Image"}  # fmt: skip
    counts: dict[str, int] = {}
    for d in documents:
        kind = names.get(d.doc_name.rsplit(".", 1)[-1].lower(), "Other")
        counts[kind] = counts.get(kind, 0) + 1
    return " · ".join(f"{n} {kind}" for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def marker_chips(answer: str) -> str:
    """[1] becomes inline code, which the stylesheet draws as a small chip. Still Markdown, never HTML."""
    return _MARKER.sub(lambda m: f" `{m.group(1)}`", answer)


def citation_label(c: Citation) -> str:
    state = ":material/check_circle:" if c.valid else ":material/warning:"
    page = "" if c.page is None else f", p{c.page}"
    return f"{state} [{c.marker}] {c.doc_name}{page}"


def source_label(c: Citation) -> str:
    """One row of the Sources card: number, document, page, and whether the quote was found."""
    page = "" if c.page is None else f" · Page {c.page}"
    state = ":green[:material/check: Checked]" if c.valid else ":orange[:material/warning: Not found]"
    return f"**{c.marker}**  {c.doc_name}{page}   {state}"


def usage_line(response: ChatResponse) -> str:
    u = response.usage
    seconds = u.latency_ms.total / 1000
    return (
        f"{u.input_tokens} tokens in, {u.output_tokens} out · {u.llm_requests} LLM requests · "
        f"\\${u.est_cost_usd:.4f} · {seconds:.1f} s"
    )


def cache_line(response: ChatResponse) -> str:
    info = response.cache
    return (
        f":material/bolt: Served from cache · similarity {info.similarity:.2f} · "
        f"{info.saved_requests or 0} LLM requests saved"
    )


def step_line(step: AgentStep) -> str:
    """Plain text: arguments and notes may hold model words, so they are never rendered as Markdown."""
    parts = [step.node]
    if step.tool:
        args = ", ".join(f"{key}={value!r}" for key, value in (step.args or {}).items())
        parts.append(f"{step.tool}({args})")
    if step.note:
        parts.append(f"- {step.note}")
    if step.hits:
        parts.append(f"[{', '.join(step.hits[:5])}{', ...' if len(step.hits) > 5 else ''}]")
    parts.append(f"({step.llm_requests} LLM requests, {step.latency_ms} ms)")
    return " ".join(parts)


def render_trace(trace: AgentTrace) -> None:
    with st.expander(f"Agent trace · {len(trace.steps)} steps · {trace.llm_requests} LLM requests"):
        st.caption("Plan")
        for number, question in enumerate(trace.sub_questions, start=1):
            st.text(f"{number}. {question}")
        st.caption("Steps")
        for step in trace.steps:
            st.text(step_line(step))
        st.caption(f"Stopped because: {trace.stop_reason or 'finished'} · tool calls: {trace.method}")


def answer_chips(response: ChatResponse, mode: str) -> str:
    citations = response.answer.citations
    chips = []
    if citations:
        valid = sum(c.valid for c in citations)
        kind = "good" if valid == len(citations) else "warn"
        chips.append(
            style.pill(f"{valid} of {len(citations)} citations checked", kind=kind, icon_name="check")
        )
    chips.append(style.pill(f"{mode.capitalize()} mode"))
    passages = len({c.chunk_id for c in citations})
    if passages:
        chips.append(style.pill(f"Cites {passages} passage{'s' if passages != 1 else ''}"))
    return f'<div class="cr-chips">{"".join(chips)}</div>'


def render_answer(response: ChatResponse, mode: str = "standard", key: int | str = 0) -> None:
    st.markdown(marker_chips(response.answer.answer))
    st.html(answer_chips(response, mode))
    for warning in response.warnings:
        st.warning(warning)
    if response.answer.citations:
        with st.container(key=f"sources_{key}"):
            st.html('<div class="cr-src-title">SOURCES</div>')
            for citation in response.answer.citations:
                with st.expander(source_label(citation)):
                    if citation.image_path and Path(citation.image_path).is_file():  # F-MM-4
                        st.image(citation.image_path, width=320)
                    st.text(citation.quote)
                    if not citation.valid:
                        st.caption(
                            f"Quote not found in the passage (match {citation.match_score:.0f} of 100)"
                        )
    if response.answer.answerable:
        st.caption(f"Confidence: {response.answer.confidence}")
    if response.cache.hit:
        st.caption(cache_line(response))
    if response.trace is not None:
        render_trace(response.trace)
    st.caption(usage_line(response))
