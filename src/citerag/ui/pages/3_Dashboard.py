"""Dashboard page (F-UI-3): charts from the request log, through the HTTP API only."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from citerag.analytics.plots import (
    calls_by_role,
    cumulative_requests,
    hit_rate_trend,
    latency_hit_miss,
    similarity_zones,
    stage_latency,
    threshold_chart_rows,
)
from citerag.ui import common, style
from citerag.ui.api_client import ApiError

client = common.get_client()


@st.cache_data(ttl=30)
def load_logs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None]:
    source = common.get_client()
    frames = pd.DataFrame(source.request_log()), pd.DataFrame(source.llm_call_log())
    return *frames, source.cache_status(), source.cache_sweep()


try:
    requests, calls, status, sweep = load_logs()
except ApiError as exc:
    st.title("Dashboard")
    st.error(exc.message)
    st.stop()

head, action = st.columns([4, 1], vertical_alignment="top")
with head:
    st.title("Dashboard")
    st.html(
        '<div style="margin-top:-0.6rem">'
        + style.pill("Cache", status["backend"])
        + style.pill("Entries", status["entries"])
        + style.pill("Threshold", f"{status['threshold']:.2f}")
        + "</div>"
    )
with action:
    st.html('<div style="height:0.9rem"></div>')
    if st.button("Clear cache", icon=":material/delete:", width="stretch"):
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
rate = f"{hits / len(used):.0%}" if len(used) else "—"
style.grid(
    [
        style.stat("Requests", len(requests)),
        style.stat("Cache hits", hits, small=f"of {len(used)} lookups"),
        style.stat("Hit rate", rate),
        style.stat("LLM requests saved", saved),
    ],
    4,
)
st.info(
    "Tokens saved are not in the log yet. `citerag eval bench-cache` reports tokens with and without the cache.",
    icon=":material/info:",
)


def card_head(title: str, subtitle: str = "", right: str = "") -> None:
    sub = f'<p class="cr-card-sub">{style.esc(subtitle)}</p>' if subtitle else ""
    side = f'<span class="cr-card-sub">{style.esc(right)}</span>' if right else ""
    st.html(f'<div class="cr-card-head"><p class="cr-card-title">{style.esc(title)}</p>{side}</div>{sub}')


def mean_ms(values: pd.Series) -> str:
    return f"{values.mean():,.0f} ms" if len(values) else "— ms"


if not used.empty:
    with st.container(key="chart_hits"):
        card_head("Cache hit rate", right=f"Last {min(20, len(used))} requests")
        st.pyplot(hit_rate_trend(used), width="stretch")

left, right = st.columns(2)
with left, st.container(key="chart_similarity"):
    card_head("Similarity of hits and misses", "Lookups at or above the threshold are served from cache.")
    if used.cache_similarity.notna().any():
        st.pyplot(similarity_zones(used, status["threshold"]), width="stretch")
        st.html(
            '<div class="cr-card-sub"><span style="color:#E5B454">■</span> Best similarity of a miss &nbsp; '
            '<span style="color:#6E8BFF">■</span> Similarity of a hit</div>'
        )
    else:
        st.caption("No cache lookups yet.")
with right, st.container(key="chart_latency"):
    card_head("Latency, hit vs miss", "Response time by lookup outcome. Shorter is better.")
    hit_mask = requests.cache_hit.astype(bool)
    hit_ms, miss_ms = requests.lat_total[hit_mask], requests.lat_total[~hit_mask]
    top = max(hit_ms.mean() if len(hit_ms) else 0, miss_ms.mean() if len(miss_ms) else 0) or 1
    bars = "".join(
        f'<div class="row"><span>{name}</span><b>{mean_ms(values)}</b></div>'
        f'<div class="cr-bar"><i style="width:{(values.mean() / top * 100) if len(values) else 0:.0f}%"></i></div>'
        for name, values in (("Cache hit", hit_ms), ("LLM call (miss)", miss_ms))
    )
    st.html(f'<div class="cr-lat">{bars}</div>')
    st.caption(
        f"{int(hit_mask.sum())} served from cache, {int((~hit_mask).sum())} answered by the LLM; mean of each."
    )

with st.expander("More charts and the request log"):
    st.pyplot(latency_hit_miss(requests))
    st.pyplot(cumulative_requests(requests))
    st.pyplot(stage_latency(requests))
    if not calls.empty:
        st.pyplot(calls_by_role(calls))
    if sweep is None:
        st.info("No threshold sweep yet. Label pairs, then run: uv run citerag eval cache-sweep")
    else:
        st.pyplot(threshold_chart_rows(sweep["rows"], sweep["recommended"]))
        labelled_by = " (labelled by Claude, not a human)" if sweep.get("ai_labelled") else ""
        st.caption(
            f"{sweep['n_equivalent']} equivalent and {sweep['n_near_miss']} near-miss pairs{labelled_by}."
        )
    columns = ["message", "cache_hit", "cache_similarity", "llm_requests", "lat_total", "error_code"]
    st.dataframe(requests[columns])
