"""Chat page (F-UI-1)."""

from __future__ import annotations

import base64
import uuid

import streamlit as st

from citerag.schemas import ChatRequest
from citerag.ui import common, style
from citerag.ui.api_client import ApiError

SUGGESTIONS = (
    "What are the core functions of the NIST AI Risk Management Framework?",
    "How does depth-first search decide which node to expand next?",
    "How do I declare a path parameter in FastAPI?",
)

client = common.get_client()
state = st.session_state
state.setdefault("session_id", uuid.uuid4().hex)
state.setdefault("history", [])

selected: list[str] = []
with st.sidebar:
    st.divider()
    try:
        documents = client.documents()
    except ApiError as exc:
        st.error(exc.message)
        documents = []
    names = {d.doc_id: d.doc_name for d in documents}
    selected = st.multiselect(
        "Search only in", list(names), format_func=names.get, placeholder="All documents"
    )
    label_side, toggle_side = st.columns([3, 1], vertical_alignment="center")
    label_side.markdown("**Use cache**")
    with toggle_side.container(key="use_cache_row"):
        use_cache = st.toggle(
            "Use cache",
            value=True,
            help="Off skips the cache for this question: no lookup and nothing stored.",
        )
    st.caption("Reuse answers to near-identical questions.")
    with st.container(key="mode"):
        mode = st.radio(
            "Mode",
            ["Standard", "Deep", "Agent"],
            horizontal=True,
            help="Deep: three agents check the answer. Agent: plans, searches the documents with tools, then answers.",
        )
    if mode == "Deep":
        st.caption("Deep mode costs at least 3 LLM requests per question, against 1 for standard.")
    if mode == "Agent":
        st.caption(
            "Agent mode costs 4 LLM requests usually and at most 8. It skips the cache and takes no images."
        )
    with st.container(key="new_chat"):
        if st.button("New chat", icon=":material/add:", width="stretch"):
            try:
                client.delete_session(state.session_id)
            except ApiError as exc:
                st.error(exc.message)
            state.session_id, state.history = uuid.uuid4().hex, []
            st.rerun()

title, scope = st.columns([3, 2], vertical_alignment="bottom")
title.title("Chat")
in_scope = (
    f"Searching {len(selected)} of {len(names)} documents"
    if selected
    else f"Searching all {len(names)} documents"
)
scope.html(f'<div class="cr-right">{style.pill(in_scope)}</div>')

picked = None
if not state.history:
    st.html(
        f'<div class="cr-empty">{style.icon("mark")}<h3>Ask about your documents</h3><p>Every answer points back to '
        "the passages it came from, and each citation is checked against its source.</p></div>"
        '<div class="cr-try">TRY ASKING</div>'
    )
    with st.container(key="suggestions"):
        for number, suggestion in enumerate(SUGGESTIONS):
            if st.button(suggestion, key=f"suggestion_{number}", width="stretch"):
                picked = suggestion

for number, turn in enumerate(state.history):
    avatar = None if turn["role"] == "user" else str(style.MARK)
    with st.chat_message(turn["role"], avatar=avatar):
        if turn["role"] == "user":
            if turn.get("image"):
                st.image(turn["image"], caption="Your image", width=240)
            st.markdown(turn["text"])
        else:
            common.render_answer(turn["response"], turn.get("mode", "standard"), number)

note = f"{mode} mode · cache {'on' if use_cache else 'off'}"
with st.bottom:  # pinned under the page: the input, then the one-line note the design shows below it
    submitted = st.chat_input(
        "Ask about your documents, or attach an image",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    st.html(f'<div class="cr-input-note">{style.esc(note)}</div>')

if submitted or picked:
    upload = submitted.files[0] if submitted and submitted.files else None
    picture = upload.getvalue() if upload else None
    question = picked or submitted.text or "What does this image show?"
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
        mode=mode.lower(),
        **image_fields,
    )
    with st.chat_message("assistant", avatar=str(style.MARK)):
        try:
            with st.spinner("Searching and checking citations"):
                response = client.chat(request)
        except ApiError as exc:
            state.history.pop()
            st.error(exc.message)
        else:
            state.history.append({"role": "assistant", "response": response, "mode": mode.lower()})
            common.render_answer(response, mode.lower(), len(state.history) - 1)
            if picked:
                st.rerun()
