"""Evaluation page (F-UI-4): run list, compare two runs, per-item drill-down."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from citerag.analytics.plots import grouped_bars, type_heatmap, type_scores
from citerag.schemas import RunSummary
from citerag.ui import common
from citerag.ui.api_client import ApiError

st.set_page_config(page_title="Evaluation · CiteRAG")
st.title("Evaluation")
client = common.get_client()


def label(run: RunSummary) -> str:
    return f"{run.run_id} ({run.status}, {run.n_done or '-'} of {run.n_items})"


def metric_table(runs: dict[str, RunSummary]) -> pd.DataFrame:
    return pd.DataFrame({name: run.headline() for name, run in runs.items()})


try:
    runs = {run.run_id: run for run in client.eval_runs()}
except ApiError as exc:
    st.error(exc.message)
    st.stop()
if not runs:
    st.info("No runs yet. Start one with: uv run citerag eval run --offline")
    st.stop()

first = st.selectbox("Run", list(runs), format_func=lambda r: label(runs[r]))
other = st.selectbox("Compare with", [None, *runs], format_func=lambda r: "No comparison" if r is None else r)
run = runs[first]
if run.unverified_items:
    st.warning(
        f"{run.unverified_items} of {run.n_items} items are not verified by a human: provisional numbers."
    )
if run.ai_reviewed_items:
    st.warning(
        f"{run.ai_reviewed_items} of {run.n_items} items were AI-reviewed (Claude), not human-verified: "
        "provisional numbers."
    )
if run.reason:
    st.info(f"Run {run.status}: {run.reason}")

chosen = {"A": run} if other is None else {"A": run, "B": runs[other]}
table = metric_table(chosen)
if other is not None and table.notna().any(axis=None):
    table["B minus A (points)"] = (table["B"] - table["A"]) * 100
st.dataframe(table, column_config=None)
if other is not None:
    st.caption("With about 40 items, differences under 3 points are noise.")
    shared = metric_table(chosen).dropna(how="all")
    st.pyplot(grouped_bars(shared, "Run A vs run B"))

if run.answers is not None:
    a = run.answers
    st.write(
        f"Injection leaks: {a.injection_leaks} of {a.injection_items}. "
        f"Waiting for the judge: {a.pending_judge}. Errors: {a.errors}. LLM requests: {run.requests_used}."
    )

detail = client.eval_run(first)
if detail.items:
    scores = type_scores(detail.items)
    st.pyplot(type_heatmap(scores))
    st.dataframe(scores)
    names = {item.id: item for item in detail.items}
    picked = names[st.selectbox("Item", list(names))]
    st.markdown(f"**{picked.type}:** {picked.question}")
    if picked.gold_answer:
        st.caption(f"Reference: {picked.gold_answer}")
    if picked.answer is None:
        st.error(f"No answer: {picked.error_code}")
    else:
        st.markdown(picked.answer.answer)
        for citation in picked.answer.citations:
            with st.expander(common.citation_label(citation)):
                st.text(citation.quote)
    for warning in picked.warnings:
        st.warning(warning)
    if picked.verdict is not None:
        st.write(f"Judge: {picked.verdict.correctness}; faithful: {picked.verdict.faithful}")
        st.text(picked.verdict.reasoning)
    elif (
        picked.answer is not None
        and picked.answer.answerable
        and picked.type in ("answerable", "multi_chunk")
    ):
        st.caption("Not judged yet.")
