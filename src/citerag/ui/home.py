"""Home page: what CiteRAG does, the three main pages, and the library at a glance."""

from __future__ import annotations

import streamlit as st

from citerag.ui import common, style
from citerag.ui.api_client import ApiError

st.html(
    '<h1 style="font-size:3.1rem;margin:0 0 0.6rem;padding:0">CiteRAG</h1>'
    '<p class="cr-sub" style="max-width:560px">Ask questions about your own documents and get answers with checked '
    "citations.</p>"
)
first, second, _ = st.columns([1.35, 1.55, 5], gap="small")
if first.button(
    "Open chat", type="primary", icon=":material/arrow_forward:", icon_position="right", key="open_chat"
):
    st.switch_page("pages/1_Chat.py")
if second.button("Add documents", key="add_documents"):
    st.switch_page("pages/2_Library.py")

cards = [
    ("chat", "chat", "Chat", "Ask a question and get an answer with checked citations."),
    ("book", "library", "Library", "Upload and manage the documents CiteRAG searches."),
    ("chart", "dashboard", "Dashboard", "Follow cache hits, similarity and latency over time."),
]
st.html(
    '<div class="cr-grid c3" style="margin-top:1.6rem">'
    + "".join(
        f'<a class="cr-card" href="{path}">{style.icon(name)}<h4>{title}</h4><p>{text}</p></a>'
        for name, path, title, text in cards
    )
    + "</div>"
)

for warning in (common.cached_health() or {}).get("warnings", []):
    st.warning(warning)

st.html('<div class="cr-section">Library at a glance</div>')
try:
    documents = common.get_client().documents()
except ApiError as exc:
    st.error(f"Backend: {exc.message}")
else:
    pdfs = [d for d in documents if d.doc_name.lower().endswith(".pdf")]
    kinds = common.count_kinds(documents)
    style.grid(
        [
            style.stat("Documents", len(documents), kinds or "None yet"),
            style.stat("Chunks indexed", sum(d.chunks for d in documents), "Searchable passages"),
            style.stat(
                "PDF pages",
                sum(d.pages or 0 for d in pdfs),
                f"Across {len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''}",
            ),
            style.stat("Images", sum(d.images for d in documents), "Found inside documents"),
        ],
        4,
    )
