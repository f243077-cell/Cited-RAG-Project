"""Step 17 (LOCAL): return shape of st.chat_input(accept_file=True), and st.pyplot, through streamlit.testing."""

import inspect
import json

from common import result

import streamlit
from streamlit.elements.widgets import chat
from streamlit.testing.v1 import AppTest


def app() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st

    value = st.chat_input("Ask", accept_file=True, file_type=["png", "jpg", "jpeg", "webp"])
    st.write({"type": type(value).__name__, "text": getattr(value, "text", None), "files": repr(getattr(value, "files", None))})
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    st.pyplot(fig)
    plt.close(fig)


print("streamlit", streamlit.__version__)
print("ChatInputValue fields:", inspect.signature(chat.ChatInputValue))
at = AppTest.from_function(app).run(timeout=30)
print("empty input returns:", at.json[0].value if at.json else at.markdown)
ok = not at.exception
if ok:
    at.chat_input[0].set_value("hello").run(timeout=30)
    print("text-only submit:", at.json[0].value if at.json else at.markdown)
    ok = not at.exception
payload = at.json[0].value if at.json else "{}"
data = json.loads(payload) if isinstance(payload, str) else payload
result(ok and data.get("text") == "hello", f"no app exceptions; pyplot rendered; last value {data}")
