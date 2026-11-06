"""Chat page (F-UI-1). Image input, Deep mode and the cache toggle arrive with M5, M7 and M6."""

from __future__ import annotations

import uuid

import streamlit as st

from citerag.schemas import ChatRequest
from citerag.ui import common
from citerag.ui.api_client import ApiError

st.set_page_config(page_title="Chat · CiteRAG")
st.title("Chat")
client = common.get_client()
state = st.session_state
state.setdefault("session_id", uuid.uuid4().hex)
state.setdefault("history", [])

selected: list[str] = []
with st.sidebar:
    try:
        documents = client.documents()
    except ApiError as exc:
        st.error(exc.message)
        documents = []
    names = {d.doc_id: d.doc_name for d in documents}
    selected = st.multiselect("Search only in", list(names), format_func=names.get)
    if st.button("New chat"):
        try:
            client.delete_session(state.session_id)
        except ApiError as exc:
            st.error(exc.message)
        state.session_id, state.history = uuid.uuid4().hex, []
        st.rerun()

for turn in state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(turn["text"])
        else:
            common.render_answer(turn["response"])

if question := st.chat_input("Ask about your documents"):
    state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    request = ChatRequest(session_id=state.session_id, message=question, doc_ids=selected or None)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and checking citations"):
                response = client.chat(request)
        except ApiError as exc:
            state.history.pop()
            st.error(exc.message)
        else:
            state.history.append({"role": "assistant", "response": response})
            common.render_answer(response)
