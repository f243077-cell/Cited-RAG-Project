"""Dashboard page (F-UI-3): charts from the request log, through the HTTP API only."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from citerag.analytics.plots import (
    cache_hit_rate,
    calls_by_role,
    cumulative_requests,
    latency_hit_miss,
    similarity_hist,
    stage_latency,
    threshold_chart_rows,
)
from citerag.ui import common
from citerag.ui.api_client import ApiError

st.set_page_config(page_title="Dashboard · CiteRAG")
st.title("Dashboard")
client = common.get_client()


@st.cache_data(ttl=30)
def load_logs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None]:
    source = common.get_client()
    frames = pd.DataFrame(source.request_log()), pd.DataFrame(source.llm_call_log())
    return *frames, source.cache_status(), source.cache_sweep()


try:
    requests, calls, status, sweep = load_logs()
except ApiError as exc:
    st.error(exc.message)
    st.stop()

first, second = st.columns([3, 1])
first.caption(f"Cache: {status['backend']}, {status['entries']} entries, threshold {status['threshold']:.2f}")
if second.button("Clear cache"):
    try:
        client.clear_cache()
    except ApiError as exc:
        st.error(exc.message)
    else:
        st.cache_data.clear()
        st.rerun()

if requests.empty:
    st.info("No requests yet. Ask something on the Chat page.")
    st.stop()

used = requests[requests.cache_enabled.astype(bool)]
hits = int(used.cache_hit.astype(bool).sum())
saved = int(requests.saved_requests.fillna(0).sum())
st.write(f"{len(requests)} requests · {hits} cache hits of {len(used)} lookups · {saved} LLM requests saved")
st.caption(
    "Tokens saved are not in the log yet; `citerag eval bench-cache` reports tokens with and without the cache."
)

if not used.empty:
    st.pyplot(cache_hit_rate(used))
    st.pyplot(similarity_hist(used, status["threshold"]))
st.pyplot(latency_hit_miss(requests))
st.pyplot(cumulative_requests(requests))
st.pyplot(stage_latency(requests))
if not calls.empty:
    st.pyplot(calls_by_role(calls))
if sweep is None:
    st.info("No threshold sweep yet. Label pairs, then run: uv run citerag eval cache-sweep")
else:
    st.pyplot(threshold_chart_rows(sweep["rows"], sweep["recommended"]))
    st.caption(f"{sweep['n_equivalent']} equivalent and {sweep['n_near_miss']} near-miss pairs.")

st.subheader("Requests")
columns = ["message", "cache_hit", "cache_similarity", "llm_requests", "lat_total", "error_code"]
st.dataframe(requests[columns])
