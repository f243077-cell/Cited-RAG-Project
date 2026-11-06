"""Chat page (F-UI-1)."""

from __future__ import annotations

import base64
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
    use_cache = st.toggle(
        "Use cache", value=True, help="Off skips the cache for this question: no lookup and nothing stored."
    )
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
            if turn.get("image"):
                st.image(turn["image"], caption="Your image", width=240)
            st.markdown(turn["text"])
        else:
            common.render_answer(turn["response"])

submitted = st.chat_input(
    "Ask about your documents, or attach an image", accept_file=True, file_type=["png", "jpg", "jpeg", "webp"]
)
if submitted:
    upload = submitted.files[0] if submitted.files else None
    picture = upload.getvalue() if upload else None
    question = submitted.text or "What does this image show?"
    state.history.append({"role": "user", "text": question, "image": picture})
    with st.chat_message("user"):
        if picture:
            st.image(picture, caption="Your image", width=240)
        st.markdown(question)
    image_fields = {}
    if upload:
        image_fields = {
            "image_b64": base64.b64encode(picture).decode("ascii"),
            "image_media_type": "image/jpeg" if upload.type == "image/jpg" else upload.type,
        }
    request = ChatRequest(
        session_id=state.session_id,
        message=question,
        doc_ids=selected or None,
        use_cache=use_cache,
        **image_fields,
    )
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
