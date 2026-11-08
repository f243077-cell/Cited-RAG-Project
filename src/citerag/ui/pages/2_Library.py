"""Library page (F-UI-2): upload, list, delete, ingest report."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from citerag.schemas import DocumentInfo
from citerag.ui import common, style
from citerag.ui.api_client import ApiError

client = common.get_client()
style.header("Library", "Add the documents CiteRAG searches, and manage what is indexed.")


def size(n: int) -> str:
    return f"{n / 1_048_576:.2f} MB" if n >= 1_048_576 else f"{n / 1024:.1f} KB"


def kind(name: str) -> str:
    return {"pdf": "PDF", "md": "MD", "markdown": "MD", "txt": "TXT"}.get(
        name.rsplit(".", 1)[-1].lower(), "IMG"
    )


def row(d: DocumentInfo) -> str:
    pages = "—" if d.pages is None else d.pages
    when = f"{d.ingested_at:%b} {d.ingested_at.day} · {d.ingested_at:%H:%M}"
    return (
        f'<tr><td><span class="cr-type">{kind(d.doc_name)}</span><span class="cr-mono">{style.esc(d.doc_name)}</span>'
        f'</td><td class="num dim">{pages}</td><td class="num">{d.chunks}</td><td class="num dim">{d.images}</td>'
        f'<td class="num dim">{size(d.size_bytes)}</td><td class="dim">{when}</td></tr>'
    )


with st.container(key="card_add"):
    st.html('<p class="cr-card-title">Add documents</p>')
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "md", "markdown", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    captions = st.toggle(
        "Caption images inside documents",
        value=False,
        help="Makes figures searchable. Each new image is one free-tier LLM request, up to 10 per document.",
    )
    if captions:
        st.caption("Each new image uses one LLM request and can take up to 60 s. Stored captions are free.")
    if uploaded and st.button("Ingest", type="primary"):
        try:
            with st.spinner("Ingesting"):
                files = [(f.name, f.getvalue()) for f in uploaded]
                st.session_state["last_report"] = client.upload(files, captions)
        except ApiError as exc:
            st.error(exc.message)

if report := st.session_state.get("last_report"):
    st.html('<div class="cr-section">Ingest report</div>')
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

try:
    documents = client.documents()
except ApiError as exc:
    st.error(exc.message)
    documents = []
if not documents:
    st.info("No documents yet.")
else:
    st.html(
        f'<div class="cr-table" style="margin-top:1.2rem"><div class="head">Documents '
        f'<span class="cr-count">{len(documents)}</span></div><table><tr><th>Document</th><th class="num">Pages</th>'
        '<th class="num">Chunks</th><th class="num">Images</th><th class="num">Size</th><th>Ingested (UTC)</th></tr>'
        + "".join(row(d) for d in documents)
        + "</table></div>"
    )
    names = {d.doc_id: d.doc_name for d in documents}
    with st.container(key="card_delete"):
        st.html(
            '<p class="cr-card-title">Delete a document</p><p class="cr-card-sub">Removes the document and its chunks '
            "from the library. This cannot be undone.</p>"
        )
        pick, act = st.columns([6, 1], vertical_alignment="bottom")
        target = pick.selectbox("Document", list(names), format_func=names.get)
        with act.container(key="delete_btn"):
            delete = st.button("Delete", icon=":material/delete:", width="stretch")
        if delete:
            try:
                client.delete_document(target)
            except ApiError as exc:
                st.error(exc.message)
            else:
                st.session_state.pop("last_report", None)
                st.rerun()
