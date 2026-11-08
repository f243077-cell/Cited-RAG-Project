"""Evaluation page (F-UI-4): run list, compare two runs, per-item drill-down."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from citerag.analytics.plots import grouped_bars, type_heatmap, type_scores
from citerag.schemas import ItemRecord, RunSummary
from citerag.ui import common, style
from citerag.ui.api_client import ApiError

client = common.get_client()
style.header("Evaluation", "Review how a run scored, and compare it with an earlier run.")

SCORES = (
    ("Faithfulness", "faithfulness"),
    ("Citation accuracy", "citation_validity"),
    ("Correctness", "correctness"),
)


def label(run: RunSummary) -> str:
    return f"{run.run_id} ({run.status}, {run.n_done or '-'} of {run.n_items})"


def metric_table(runs: dict[str, RunSummary]) -> pd.DataFrame:
    return pd.DataFrame({name: run.headline() for name, run in runs.items()})


def review_state(run: RunSummary) -> tuple[str, str]:
    if run.unverified_items:
        return "Not verified", "warn"
    if run.ai_reviewed_items:
        return "AI-reviewed", "info"
    return "Verified", "good"


try:
    runs = {run.run_id: run for run in client.eval_runs()}
except ApiError as exc:
    st.error(exc.message)
    st.stop()
if not runs:
    st.info("No runs yet. Start one with: uv run citerag eval run --offline")
    st.stop()

pick, compare = st.columns(2)
first = pick.selectbox("Run", list(runs), format_func=lambda r: label(runs[r]))
other = compare.selectbox(
    "Compare with", [None, *runs], format_func=lambda r: "No comparison" if r is None else r
)
run = runs[first]
if run.unverified_items:
    st.warning(
        f"**{run.unverified_items} of {run.n_items} items are not verified by a human**  \n"
        "The numbers below are provisional until each item has been reviewed.",
        icon=":material/warning:",
    )
if run.ai_reviewed_items:
    st.warning(
        f"**{run.ai_reviewed_items} of {run.n_items} items were AI-reviewed (Claude), not human-verified**  \n"
        "The numbers below are provisional until a human has checked each item.",
        icon=":material/warning:",
    )
if run.reason:
    st.info(f"Run {run.status}: {run.reason}")

human = run.n_items - run.unverified_items - run.ai_reviewed_items
style.grid(
    [
        style.stat("Items", run.n_items),
        style.stat("Completed", run.n_done, small=f"of {run.n_items}"),
        style.stat("Verified by a human", human, small=f"of {run.n_items}", warn=human < run.n_items),
    ],
    3,
)

headline = run.headline()
provisional = run.unverified_items or run.ai_reviewed_items
st.html(
    '<div class="cr-section">Scores ' + (style.badge("Provisional", "warn") if provisional else "") + "</div>"
)
style.grid(
    [style.stat(name, "—" if headline.get(key) is None else f"{headline[key]:.0%}") for name, key in SCORES],
    3,
)

chosen = {"A": run} if other is None else {"A": run, "B": runs[other]}
if other is not None:
    table = metric_table(chosen)
    if table.notna().any(axis=None):
        table["B minus A (points)"] = (table["B"] - table["A"]) * 100
    st.html('<div class="cr-section">Comparison</div>')
    st.dataframe(table)
    st.caption("With about 40 items, differences under 3 points are noise.")
    shared = metric_table(chosen).dropna(how="all")
    st.pyplot(grouped_bars(shared, "Run A vs run B"))

if run.answers is not None:
    a = run.answers
    st.caption(
        f"Injection leaks: {a.injection_leaks} of {a.injection_items}. "
        f"Waiting for the judge: {a.pending_judge}. Errors: {a.errors}. LLM requests: {run.requests_used}."
    )

with st.expander("All metrics for this run"):
    st.dataframe(metric_table({"A": run}))

detail = client.eval_run(first)
if detail.items:
    state_text, state_kind = review_state(run)
    names: dict[str, ItemRecord] = {item.id: item for item in detail.items}
    st.html('<div class="cr-section">Items</div>')
    with st.container(key="items_table", border=True):
        head = st.columns([5, 1.3, 1.5, 1.1])
        for column, text in zip(head, ("ITEM", "RUN", "REVIEW", "ACTION"), strict=True):
            column.caption(text)
        for item in detail.items:
            cells = st.columns([5, 1.3, 1.5, 1.1], vertical_alignment="center")
            cells[0].markdown(f"**{item.id}** · {item.question}")
            done = item.status == "answered"
            cells[1].html(style.badge("Complete" if done else "Error", "good" if done else "bad"))
            cells[2].html(style.badge(state_text, state_kind))
            if cells[3].button("Review", key=f"review_{item.id}"):
                st.session_state["eval_item"] = item.id
    picked_id = st.session_state.get("eval_item")
    picked = names.get(picked_id) or detail.items[0]

    st.html('<div class="cr-section">Item review</div>')
    picked = names[st.selectbox("Item", list(names), index=list(names).index(picked.id))]
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

    with st.expander("Scores by question type"):
        scores = type_scores(detail.items)
        st.pyplot(type_heatmap(scores))
        st.dataframe(scores)
