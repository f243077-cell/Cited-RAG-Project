"""Library page (F-UI-2): upload, list, delete, ingest report."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from citerag.ui import common
from citerag.ui.api_client import ApiError

st.set_page_config(page_title="Library · CiteRAG")
st.title("Library")
client = common.get_client()

uploaded = st.file_uploader(
    "Add documents", type=["pdf", "md", "markdown", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True
)
captions = st.toggle(
    "Caption images inside documents",
    value=False,
    help="Makes figures searchable. Each new image is one free-tier LLM request, up to 10 per document.",
)
if captions:
    st.caption("Each new image uses one LLM request and can take up to 60 s. Stored captions are free.")
if uploaded and st.button("Ingest"):
    try:
        with st.spinner("Ingesting"):
            files = [(f.name, f.getvalue()) for f in uploaded]
            st.session_state["last_report"] = client.upload(files, captions)
    except ApiError as exc:
        st.error(exc.message)

if report := st.session_state.get("last_report"):
    st.subheader("Ingest report")
    columns = [
        "doc_name",
        "status",
        "pages",
        "chunks",
        "images_found",
        "images_captioned",
        "images_skipped",
        "caption_requests",
        "duration_ms",
    ]
    st.dataframe(pd.DataFrame([r.model_dump() for r in report])[columns], hide_index=True)

st.subheader("Documents")
try:
    documents = client.documents()
except ApiError as exc:
    st.error(exc.message)
    documents = []
if not documents:
    st.info("No documents yet.")
else:
    rows = [d.model_dump(exclude={"doc_id"}) for d in documents]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    names = {d.doc_id: d.doc_name for d in documents}
    target = st.selectbox("Delete a document", list(names), format_func=names.get)
    if st.button("Delete"):
        try:
            client.delete_document(target)
        except ApiError as exc:
            st.error(exc.message)
        else:
            st.session_state.pop("last_report", None)
            st.rerun()
