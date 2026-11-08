"""Streamlit entry point: uv run streamlit run src/citerag/ui/app.py

It routes the five pages, applies the stylesheet and draws the sidebar status card once for every page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from citerag.ui import common, style

HERE = Path(__file__).parent

st.set_page_config(page_title="CiteRAG", page_icon=str(style.MARK), layout="wide")
style.apply()
st.logo(str(style.LOGO), size="large", icon_image=str(style.MARK))

pages = [
    st.Page(str(HERE / "home.py"), title="Home", icon=":material/home:", default=True),
    st.Page(str(HERE / "pages" / "1_Chat.py"), title="Chat", icon=":material/chat_bubble:", url_path="chat"),
    st.Page(str(HERE / "pages" / "2_Library.py"), title="Library", icon=":material/menu_book:", url_path="library"),
    st.Page(str(HERE / "pages" / "3_Dashboard.py"), title="Dashboard", icon=":material/bar_chart:", url_path="dashboard"),
    st.Page(
        str(HERE / "pages" / "4_Evaluation.py"), title="Evaluation", icon=":material/assignment_turned_in:",
        url_path="evaluation",
    ),
]  # fmt: skip
page = st.navigation(pages)
style.sidebar_status(common.cached_health())
page.run()
