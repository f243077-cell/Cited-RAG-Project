"""Streamlit entry point: uv run streamlit run src/citerag/ui/app.py"""

from __future__ import annotations

import streamlit as st

from citerag.ui import common
from citerag.ui.api_client import ApiError

st.set_page_config(page_title="CiteRAG")
st.title("CiteRAG")
st.write("Ask questions about your own documents and get answers with checked citations.")
st.page_link("pages/1_Chat.py", label="Chat")
st.page_link("pages/2_Library.py", label="Library")
st.page_link("pages/3_Dashboard.py", label="Dashboard")

try:
    health = common.get_client().health()
except ApiError as exc:
    st.error(f"Backend: {exc.message}")
else:
    used, limit = health["llm_requests_24h"], health["max_requests_per_day"]
    st.caption(f"Backend is running · {used} of {limit} LLM requests used in the last 24 hours")
