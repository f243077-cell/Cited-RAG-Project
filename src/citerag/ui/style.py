"""The look of the UI: one stylesheet and small HTML pieces for the page chrome (cards, chips, tables).

F-SAF-2 still holds: model output never goes through here. Only fixed strings and values passed through `esc` reach
`st.html`; answers, quotes and traces stay plain Markdown or plain text in `common.py`.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "logo.svg"
MARK = ASSETS / "mark.svg"

_STROKE = (
    'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
)
ICONS = {
    "chat": f'<svg viewBox="0 0 24 24" {_STROKE}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "book": f'<svg viewBox="0 0 24 24" {_STROKE}><path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/></svg>',
    "chart": f'<svg viewBox="0 0 24 24" {_STROKE}><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
    "check": f'<svg viewBox="0 0 24 24" {_STROKE}><path d="M20 6 9 17l-5-5"/></svg>',
    "info": f'<svg viewBox="0 0 24 24" {_STROKE}><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "mark": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M9 5H6v14h3"/><path d="M15 5h3v14h-3"/><circle cx="12" cy="12" r="1.3" '
    'fill="currentColor" stroke="none"/></svg>',
}

CSS = """
<style>
:root {
  --bg: #0D1015; --side: #11141B; --card: #151922; --card-2: #1A1F2A; --line: #232833; --line-2: #2C3240;
  --ink: #E6E8EE; --muted: #9AA3B2; --faint: #6B7383; --accent: #6E8BFF; --accent-soft: #1E2640;
  --accent-ink: #8EA2FF; --good: #4ADE80; --good-soft: #12261C; --warn: #F2C46D; --warn-soft: #251F12;
  --warn-line: #5A4A26; --bad: #F08A8A; --bad-line: #5C2B2B; --mono: "JetBrains Mono", ui-monospace, monospace;
}
.block-container { max-width: 1000px; padding-top: 3.2rem; }
h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
[data-testid="stHeader"] { background: transparent; }

/* Sidebar: logo, navigation, the status card pinned to the bottom */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); width: 240px !important; min-width: 240px !important; }
[data-testid="stSidebarNavSeparator"] { display: none; }
[data-testid="stSidebarNav"] a { border-radius: 10px; padding: 0.45rem 0.75rem; margin: 2px 0; }
[data-testid="stSidebarNav"] a span { color: var(--muted); font-weight: 500; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: #1C2232; }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--ink); font-weight: 600; }
[data-testid="stSidebarNav"] a[aria-current="page"] [data-testid="stIconMaterial"] { color: var(--accent); }
[data-testid="stSidebarUserContent"] { padding-bottom: 8.5rem; }
[data-testid="stSidebarUserContent"] hr { margin: 0.4rem 0 1rem; border-color: var(--line); }
.st-key-status_card { position: fixed; bottom: 16px; left: 16px; width: 208px; z-index: 100; }
[data-testid="stSidebar"][aria-expanded="false"] .st-key-status_card { display: none; }
.cr-status { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 14px 12px; }
.cr-status .row { display: flex; align-items: center; gap: 8px; font-size: 0.86rem; color: var(--ink); }
.cr-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); display: inline-block; }
.cr-dot.down { background: var(--bad); }
.cr-status .quota { display: flex; justify-content: space-between; margin-top: 14px; font-size: 0.76rem;
  color: var(--muted); }
.cr-status .quota b { font-family: var(--mono); font-weight: 400; color: var(--ink); }
.cr-bar { height: 5px; background: #232836; border-radius: 3px; margin-top: 8px; overflow: hidden; }
.cr-bar > i { display: block; height: 100%; background: var(--accent); border-radius: 3px; }

/* Shared pieces */
.cr-sub { color: var(--muted); font-size: 1.05rem; margin: -0.4rem 0 1.4rem; line-height: 1.55; }
.cr-section { font-weight: 600; font-size: 0.98rem; margin: 2rem 0 0.8rem; color: var(--ink); }
.cr-grid { display: grid; gap: 14px; }
.cr-grid.c3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cr-grid.c4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 760px) { .cr-grid.c3, .cr-grid.c4 { grid-template-columns: 1fr 1fr; } }
.cr-card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 20px 22px; }
a.cr-card { display: block; text-decoration: none !important; color: inherit !important; transition: border-color .15s; }
a.cr-card:hover { border-color: var(--line-2); background: var(--card-2); }
.cr-icon { width: 38px; height: 38px; border-radius: 10px; background: var(--accent-soft); color: var(--accent);
  display: flex; align-items: center; justify-content: center; margin-bottom: 18px; }
.cr-icon .cr-glyph { width: 19px; height: 19px; display: block; }
.cr-card h4 { margin: 0 0 6px; padding: 0; font-size: 1.08rem; font-weight: 600; color: var(--ink); }
.cr-card p { margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.5; }
.cr-stat .label { color: var(--muted); font-size: 0.86rem; }
.cr-stat .value { font-size: 1.85rem; font-weight: 600; color: var(--ink); margin: 6px 0 4px; line-height: 1.15; }
.cr-stat .value small { font-size: 0.86rem; color: var(--muted); font-weight: 400; margin-left: 4px; }
.cr-stat .note { color: var(--muted); font-size: 0.78rem; }
.cr-stat.warn { border-color: var(--warn-line); }
.cr-stat.warn .value { color: var(--warn); }
.cr-pill { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line-2); background: var(--card);
  border-radius: 999px; padding: 4px 12px; font-size: 0.8rem; color: var(--muted); margin-right: 6px; }
.cr-pill code, .cr-pill b { font-family: var(--mono); font-weight: 400; color: var(--ink); background: none; padding: 0; }
.cr-pill.good { color: var(--good); border-color: #1F4A30; background: var(--good-soft); }
.cr-pill.warn { color: var(--warn); border-color: var(--warn-line); background: var(--warn-soft); }
.cr-pill .cr-glyph { width: 13px; height: 13px; display: inline-block; }
.cr-right { display: flex; justify-content: flex-end; padding-top: 0.9rem; }
.cr-mono { font-family: var(--mono); font-size: 0.86rem; }

/* Tables */
.cr-table { background: var(--card); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.cr-table .head { padding: 16px 20px; font-weight: 600; border-bottom: 1px solid var(--line); display: flex;
  align-items: center; gap: 10px; }
.cr-count { font-size: 0.75rem; font-family: var(--mono); color: var(--muted); border: 1px solid var(--line-2);
  border-radius: 999px; padding: 1px 8px; font-weight: 400; }
.cr-table table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 0; }
.cr-table th { text-align: left; font-weight: 500; font-size: 0.72rem; letter-spacing: 0.06em; color: var(--faint);
  padding: 12px 20px; border-bottom: 1px solid var(--line); text-transform: uppercase; background: transparent; }
.cr-table td { padding: 14px 20px; border-bottom: 1px solid var(--line); color: var(--ink); border-left: none;
  border-right: none; }
.cr-table tr:last-child td { border-bottom: none; }
.cr-table th, .cr-table td { border-left: none !important; border-right: none !important; }
.cr-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.cr-table .dim { color: var(--muted); }
.cr-type { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); background: #1D2230;
  border: 1px solid var(--line-2); border-radius: 5px; padding: 1px 6px; margin-right: 10px; }
.cr-badge { font-size: 0.78rem; border-radius: 999px; padding: 3px 10px; border: 1px solid; white-space: nowrap; }
.cr-badge.good { color: var(--good); border-color: #1F4A30; background: var(--good-soft); }
.cr-badge.warn { color: var(--warn); border-color: var(--warn-line); background: var(--warn-soft); }
.cr-badge.bad { color: var(--bad); border-color: var(--bad-line); background: #2A1616; }
.cr-badge.info { color: var(--accent-ink); border-color: #2C3760; background: var(--accent-soft); }

/* Bordered containers act as cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 14px; }
.st-key-card_add, .st-key-card_delete, .st-key-chart_hits, .st-key-chart_similarity, .st-key-chart_latency, [class*="st-key-sources_"],
[class*="st-key-sources_"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; }
.cr-card-title { font-weight: 600; font-size: 0.98rem; color: var(--ink); margin: 0; }
.cr-card-sub { color: var(--muted); font-size: 0.84rem; margin: 2px 0 0; }
.cr-card-head { display: flex; justify-content: space-between; align-items: baseline; }

/* Alerts as the design's banners */
[data-testid="stAlertContainer"] { border-radius: 12px; border: 1px solid var(--line); background: var(--card); }
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentInfo"] { font-size: 0.9rem; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) [data-testid="stAlertContainer"] {
  background: var(--warn-soft); border-color: var(--warn-line); color: var(--warn); }
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) p { color: #E9DCC0; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) strong { color: var(--warn); }
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertContainer"] {
  background: var(--card); border-color: var(--line); }
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) p,
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stIconMaterial"] {
  color: var(--muted); }
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) code { color: var(--ink); }

/* Buttons */
.stButton button, .stDownloadButton button { font-weight: 500; padding: 0.5rem 1.1rem; }
.stButton button[kind="primary"] { color: #0B0E14; font-weight: 600; }
.stButton button[kind="secondary"] { background: var(--card); border-color: var(--line-2); }
.st-key-delete_btn button { color: var(--bad); border-color: var(--bad-line); background: transparent; }
.st-key-delete_btn button:hover { border-color: var(--bad); color: var(--bad); }
.st-key-suggestions .stButton button > div { justify-content: flex-start; width: 100%; }
.st-key-suggestions .stButton button p { text-align: left; }
.st-key-suggestions .stButton button { justify-content: flex-start; text-align: left; background: var(--card);
  border-color: var(--line); color: var(--ink); font-weight: 400; padding: 0.7rem 1rem; }
.st-key-suggestions .stButton button:hover { border-color: var(--line-2); background: var(--card-2); }
.st-key-items_table .stButton button { padding: 0.3rem 0.9rem; font-size: 0.85rem; }

/* Chat */
.cr-empty { text-align: center; margin: 7vh auto 1.6rem; max-width: 470px; }
.cr-empty .cr-icon { width: 54px; height: 54px; border-radius: 14px; margin: 0 auto 26px; }
.cr-empty .cr-icon .cr-glyph { width: 24px; height: 24px; }
.cr-empty h3 { font-size: 1.4rem; font-weight: 600; margin: 0 0 8px; padding: 0; }
.cr-empty p { color: var(--muted); font-size: 0.93rem; line-height: 1.55; }
.cr-try { max-width: 540px; margin: 0 auto 6px; font-size: 0.72rem; letter-spacing: 0.08em; color: var(--faint); }
.st-key-suggestions { max-width: 540px; margin: 0 auto; }
[data-testid="stChatMessage"] { background: transparent; padding: 0.4rem 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
  display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 75%; background: var(--card-2); border: 1px solid var(--line); border-radius: 12px;
  padding: 0.55rem 1rem; }
[data-testid="stChatMessageContent"] p { line-height: 1.75; }
[data-testid="stChatMessageContent"] p code { background: var(--accent-soft); color: var(--accent-ink);
  border-radius: 5px; padding: 1px 6px; font-size: 0.72rem; }
.cr-chips { display: flex; flex-wrap: wrap; gap: 0; margin: 6px 0 4px; }
.cr-src-title { font-size: 0.72rem; letter-spacing: 0.08em; color: var(--faint); margin: 0 0 4px; }
[class*="st-key-sources_"] [data-testid="stExpander"] details { border: none; border-top: 1px solid var(--line); border-radius: 0; }
[class*="st-key-sources_"] [data-testid="stExpander"] summary { padding: 0.75rem 0.2rem; }
[class*="st-key-sources_"] [data-testid="stExpander"] summary p { font-family: var(--mono); font-size: 0.85rem; }
[data-testid="stChatInput"] { border-radius: 14px; border-color: var(--line-2); background: var(--card); }
[data-testid="stBottomBlockContainer"] { max-width: 1000px; padding-bottom: 1.4rem; }
.cr-input-note { text-align: center; color: var(--faint); font-size: 0.78rem; margin-top: -4px; }

/* Chat sidebar controls */
.st-key-use_cache_row [data-testid="stWidgetLabel"] { display: none; }
.st-key-use_cache_row .stCheckbox { display: flex; justify-content: flex-end; }
.st-key-mode [data-testid="stElementContainer"], .st-key-mode .stRadio,
.st-key-mode [data-testid="stRadio"] > div { width: 100% !important; }
.st-key-mode [data-testid="stRadioGroup"] { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 3px; gap: 2px; display: flex; flex-wrap: nowrap; width: 100%; }
.st-key-mode [data-testid="stRadioGroup"] > div { flex: 1 1 0; min-width: 0; margin: 0; border-radius: 8px;
  display: flex; }
.st-key-mode [data-testid="stRadioOption"] { width: 100%; justify-content: center; padding: 0.4rem 0; margin: 0;
  cursor: pointer; }
.st-key-mode [data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-mode [data-testid="stRadioOption"] p { font-size: 0.86rem; color: var(--muted); white-space: nowrap; }
.st-key-mode [data-testid="stRadioGroup"] > div[data-selected="true"] { background: #222A3C; }
.st-key-mode [data-testid="stRadioGroup"] > div[data-selected="true"] p { color: var(--ink); font-weight: 600; }
.st-key-new_chat button { width: 100%; }

/* Library uploader as the design's drop zone: icon tile, two lines of text, the button on the right */
[data-testid="stFileUploaderDropzone"] { background: transparent; border: 1px dashed var(--line-2); border-radius: 12px;
  padding: 16px 18px; display: flex; flex-direction: row-reverse; justify-content: space-between; align-items: center; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: flex; align-items: center; gap: 14px; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content: ""; flex: 0 0 38px; height: 38px; border-radius: 10px;
  background: #1E2640 var(--upload-icon) center / 18px no-repeat; }
[data-testid="stFileUploaderDropzoneInstructions"] > div { display: flex; flex-direction: column; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::before { content: "Drop files here, or choose them from your computer";
  font-size: 0.9rem; color: var(--ink); }
[data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0.78rem; color: var(--muted); }
[data-testid="stFileUploaderDropzone"] button { background: var(--accent); color: #0B0E14; border: none; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] { display: none; }
[data-testid="stFileUploaderDropzone"] button p { font-size: 0; }
[data-testid="stFileUploaderDropzone"] button p::after { content: "Upload files"; font-size: 0.88rem; font-weight: 600; }

/* Charts */
.cr-lat { margin: 18px 0 4px; }
.cr-lat .row { display: flex; justify-content: space-between; font-size: 0.88rem; color: var(--ink); }
.cr-lat .row b { font-family: var(--mono); font-weight: 400; color: var(--muted); font-size: 0.8rem; }
.cr-lat .cr-bar { height: 8px; margin: 10px 0 22px; background: #202533; }
</style>
"""


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def _icon_css() -> str:
    """st.html sanitises inline SVG away, so each icon is a CSS mask over a data URI, coloured by `currentColor`."""
    rules = []
    for name, svg in ICONS.items():
        data = quote(
            svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ').replace("currentColor", "black")
        )
        rules.append(
            f".i-{name}::before {{ content: ''; display: block; width: 100%; height: 100%; background: currentColor; "
            f'-webkit-mask: url("data:image/svg+xml,{data}") center / contain no-repeat; '
            f'mask: url("data:image/svg+xml,{data}") center / contain no-repeat; }}'
        )
    upload = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#6E8BFF" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><path d="m17 8-5-5-5 5"/>'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/></svg>'
    )
    rules.append(f':root {{ --upload-icon: url("data:image/svg+xml,{quote(upload)}"); }}')
    return "<style>" + " ".join(rules) + "</style>"


def apply() -> None:
    """The stylesheet: fixed text, no value from a user or a model."""
    st.html(CSS + _icon_css())


def icon(name: str) -> str:
    return f'<span class="cr-icon"><i class="cr-glyph i-{name}"></i></span>'


def glyph(name: str) -> str:
    return f'<i class="cr-glyph i-{name}"></i>'


def header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.html(f'<p class="cr-sub">{esc(subtitle)}</p>')


def stat(label: str, value: object, note: str = "", small: str = "", warn: bool = False) -> str:
    extra = f"<small>{esc(small)}</small>" if small else ""
    return (
        f'<div class="cr-card cr-stat{" warn" if warn else ""}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}{extra}</div><div class="note">{esc(note)}</div></div>'
    )


def grid(cells: list[str], columns: int) -> None:
    st.html(f'<div class="cr-grid c{columns}">{"".join(cells)}</div>')


def pill(text: str, value: object | None = None, kind: str = "", icon_name: str | None = None) -> str:
    mark = glyph(icon_name) if icon_name else ""
    shown = f" <b>{esc(value)}</b>" if value is not None else ""
    return f'<span class="cr-pill {kind}">{mark}{esc(text)}{shown}</span>'


def badge(text: str, kind: str) -> str:
    return f'<span class="cr-badge {kind}">{esc(text)}</span>'


def sidebar_status(health: dict | None) -> None:
    """The card at the bottom of the sidebar: backend state and the 24-hour request count against the guard."""
    if health is None:
        body = '<div class="row"><span class="cr-dot down"></span>Backend not reachable</div>'
    else:
        used, limit = int(health.get("llm_requests_24h", 0)), int(health.get("max_requests_per_day", 0)) or 1
        width = min(100, round(100 * used / limit))
        body = (
            '<div class="row"><span class="cr-dot"></span>Backend running</div>'
            f'<div class="quota"><span>LLM requests · 24 h</span><b>{used} / {limit}</b></div>'
            f'<div class="cr-bar"><i style="width:{width}%"></i></div>'
        )
    with st.sidebar.container(key="status_card"):
        st.html(f'<div class="cr-status">{body}</div>')
