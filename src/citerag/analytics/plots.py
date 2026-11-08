"""Charts as pure functions: data in, Matplotlib Figure out. No pyplot, so nothing global is touched.
Bars carry direct labels and every page shows the same numbers as a table, so no reading rests on color alone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from citerag.schemas import ItemRecord

if TYPE_CHECKING:
    from citerag.eval.bench import BenchSummary
    from citerag.eval.cache_tools import SweepResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # fixed order, checked with the palette validator
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)
_BAR_HEIGHT = 0.26


def _canvas(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE)
    fig.add_subplot(1, 1, 1, facecolor=SURFACE)
    return fig


def _quiet(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(frame: pd.DataFrame, title: str) -> Figure:
    """Horizontal bars, shares from 0 to 1. Rows are metrics, columns are the runs (at most three).
    Used for run A vs B (chart 8) and for experiments (chart 10)."""
    rows, series = list(frame.index), list(frame.columns)
    fig = _canvas(7.5, 0.5 + 0.34 * len(rows) * max(len(series), 2))
    ax = fig.axes[0]
    for i, name in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (_BAR_HEIGHT + 0.02)
        places = [r + offset for r in range(len(rows))]
        values = frame[name].astype(float)
        ax.barh(places, values.fillna(0), height=_BAR_HEIGHT, color=SERIES[i], label=str(name))
        for y, value in zip(places, values, strict=True):
            text = "not measured" if pd.isna(value) else f"{value:.0%}"
            ax.text(0.0 if pd.isna(value) else value + 0.01, y, text, va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), rows, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    if len(series) > 1:
        fig.legend(frameon=False, loc="outside lower left", ncols=len(series), fontsize=8, labelcolor=MUTED)
    _quiet(ax)
    return fig


def type_scores(items: list[ItemRecord]) -> pd.DataFrame:
    """Per-type shares for the heatmap (chart 9). NaN marks a share with nothing to measure."""
    rows = []
    for record in items:
        answer = record.answer
        citations = answer.citations if answer else []
        rows.append(
            {
                "type": record.type,
                "answered": float(bool(answer and answer.answerable)),
                "citations valid": sum(c.valid for c in citations) / len(citations) if citations else None,
                "faithful": float(record.verdict.faithful) if record.verdict else None,
                "correct": float(record.verdict.correctness == "correct") if record.verdict else None,
            }
        )
    frame = pd.DataFrame(rows, columns=["type", "answered", "citations valid", "faithful", "correct"])
    return frame.astype({c: float for c in frame.columns if c != "type"}).groupby("type").mean()


def type_heatmap(frame: pd.DataFrame, title: str = "Scores by question type") -> Figure:
    fig = _canvas(6.5, 0.9 + 0.5 * len(frame))
    ax = fig.axes[0]
    ax.imshow(frame.to_numpy(dtype=float), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for y, row in enumerate(frame.to_numpy(dtype=float)):
        for x, value in enumerate(row):
            dark = not pd.isna(value) and value >= 0.55
            text = "not measured" if pd.isna(value) else f"{value:.0%}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color=SURFACE if dark else MUTED)
    ax.set_xticks(range(len(frame.columns)), frame.columns, color=INK, fontsize=9)
    ax.set_yticks(range(len(frame)), frame.index, color=INK, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    return fig


def _quiet_lines(ax) -> None:
    """For line charts the grid runs across, so values read against the y axis."""
    _quiet(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def cache_hit_rate(requests: pd.DataFrame, window: int = 20) -> Figure:
    """Chart 1: rolling hit rate over requests where the cache was on."""
    rate = requests.cache_hit.astype(float).rolling(window, min_periods=1).mean().to_numpy()
    fig = _canvas(7.5, 2.8)
    ax = fig.axes[0]
    ax.plot(range(1, len(rate) + 1), rate, color=SERIES[0], linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Request", color=MUTED, fontsize=9)
    ax.set_title(f"Cache hit rate, last {window} requests", loc="left", color=INK, fontsize=11)
    _quiet_lines(ax)
    return fig


def latency_hit_miss(requests: pd.DataFrame) -> Figure:
    """Chart 2: latency with a log axis, because a hit is often a thousand times faster."""
    hit = requests.cache_hit.astype(bool)
    groups = [("Miss", requests.lat_total[~hit]), ("Hit", requests.lat_total[hit])]
    shown = [(name, values.clip(lower=1)) for name, values in groups if len(values)]
    fig = _canvas(7.5, 2.8)
    ax = fig.axes[0]
    ax.boxplot(
        [values.to_numpy() for _, values in shown],
        orientation="horizontal",
        tick_labels=[f"{name} (n={len(values)})" for name, values in shown],
        widths=0.5,
        medianprops={"color": SERIES[1], "linewidth": 2},
        boxprops={"color": SERIES[0]},
        whiskerprops={"color": SERIES[0]},
        capprops={"color": SERIES[0]},
        flierprops={"markeredgecolor": MUTED, "markersize": 4},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Milliseconds, log scale", color=MUTED, fontsize=9)
    ax.set_title("Latency, hit vs miss", loc="left", color=INK, fontsize=11)
    _quiet(ax)
    return fig


def cumulative_requests(requests: pd.DataFrame) -> Figure:
    """Chart 3: one axis. Shadow dollars read 0 on free models, so the spec's second axis would be empty."""
    actual = requests.llm_requests.cumsum().to_numpy()
    without = (requests.llm_requests + requests.saved_requests.fillna(0)).cumsum().to_numpy()
    places = range(1, len(actual) + 1)
    fig = _canvas(7.5, 2.8)
    ax = fig.axes[0]
    ax.plot(places, without, color=SERIES[1], linewidth=2, label="Without the cache")
    ax.plot(places, actual, color=SERIES[0], linewidth=2, label="Actual")
    for values in (without, actual):
        ax.annotate(f"{values[-1]:.0f}", (places[-1], values[-1]), xytext=(4, 0), textcoords="offset points",
                    color=MUTED, fontsize=8, va="center")  # fmt: skip
    ax.set_xlabel("Request", color=MUTED, fontsize=9)
    ax.set_title("Cumulative LLM requests", loc="left", color=INK, fontsize=11)
    fig.legend(frameon=False, loc="outside lower left", ncols=2, fontsize=8, labelcolor=MUTED)
    _quiet_lines(ax)
    return fig


def panel_bars(panels: dict[str, pd.Series], title: str, height: float = 2.8) -> Figure:
    """Small multiples of labelled bars, one x scale per panel: never two scales on one axis."""
    fig = Figure(figsize=(7.5, height), layout="constrained", facecolor=SURFACE)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (name, values) in zip(axes, panels.items(), strict=True):
        ax.set_facecolor(SURFACE)
        rows = [str(label) for label in values.index]
        numbers = values.astype(float).to_numpy()
        ax.barh(rows, numbers, height=0.55, color=SERIES[0])
        top = float(numbers.max()) if len(numbers) and numbers.max() > 0 else 1.0
        for row, value in zip(rows, numbers, strict=True):
            ax.text(value + top * 0.02, row, f"{value:,.0f}", va="center", fontsize=8, color=MUTED)
        ax.set_xlim(0, top * 1.2)
        ax.invert_yaxis()
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
        _quiet(ax)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    return fig


def calls_by_role(calls: pd.DataFrame) -> Figure:
    """Chart 4: tokens and requests are different measures, so each gets its own panel."""
    frame = calls.assign(tokens=calls.input_tokens + calls.output_tokens)
    grouped = frame.groupby("role").agg(tokens=("tokens", "sum"), requests=("role", "count"))
    grouped = grouped.sort_values("requests", ascending=False)
    return panel_bars(
        {"LLM requests": grouped.requests, "Tokens": grouped.tokens},
        "LLM use by call type",
        1.2 + 0.32 * len(grouped),
    )


_STAGES = {
    "lat_condense": "Condense",
    "lat_embed": "Embed",
    "lat_cache": "Cache",
    "lat_retrieve": "Retrieve",
    "lat_rerank": "Rerank",
    "lat_generate": "Generate",
    "lat_verify": "Verify",
}


def stage_latency(requests: pd.DataFrame) -> Figure:
    """Chart 5: mean milliseconds per stage on misses. One hue: the length is the message."""
    misses = requests[~requests.cache_hit.astype(bool)]
    means = misses[list(_STAGES)].mean().rename(_STAGES).fillna(0)
    return panel_bars({"Mean milliseconds": means}, "Latency by stage, misses only", 3.0)


def similarity_hist(requests: pd.DataFrame, threshold: float) -> Figure:
    """Chart 6: how close the misses came. Misses that crowd the line are where the threshold matters."""
    seen = requests[requests.cache_similarity.notna()]
    hit = seen.cache_hit.astype(bool)
    bins = np.linspace(0, 1, 41)
    fig = _canvas(7.5, 2.8)
    ax = fig.axes[0]
    ax.hist(
        seen.cache_similarity[~hit], bins=bins, color=SERIES[0], alpha=0.75, label="Best similarity of a miss"
    )
    ax.hist(seen.cache_similarity[hit], bins=bins, color=SERIES[1], alpha=0.75, label="Similarity of a hit")
    ax.axvline(threshold, color=INK, linewidth=1.5, linestyle="--")
    ax.text(threshold, ax.get_ylim()[1], f" threshold {threshold:.2f}", color=INK, fontsize=8, va="top")
    ax.set_xlabel("Cosine similarity", color=MUTED, fontsize=9)
    ax.set_title("Similarity of hits and misses", loc="left", color=INK, fontsize=11)
    fig.legend(frameon=False, loc="outside lower left", ncols=2, fontsize=8, labelcolor=MUTED)
    _quiet_lines(ax)
    return fig


def threshold_chart_rows(rows: list[dict], chosen: float | None) -> Figure:
    """Chart 7: three lines on one share axis, with the recommended threshold marked."""
    frame = pd.DataFrame(rows)
    fig = _canvas(7.5, 3.2)
    ax = fig.axes[0]
    for color, column, name in zip(
        SERIES,
        ("precision", "recall", "false_hit_rate"),
        ("Precision", "Recall", "False-hit rate"),
        strict=True,
    ):
        ax.plot(frame.tau, frame[column], color=color, linewidth=2, label=name)
    if chosen is not None:
        ax.axvline(chosen, color=INK, linewidth=1.5, linestyle="--")
        ax.text(chosen + 0.004, 1.04, f"{chosen:.2f}", color=INK, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Threshold", color=MUTED, fontsize=9)
    ax.set_title("Cache threshold sweep, guard applied", loc="left", color=INK, fontsize=11)
    fig.legend(frameon=False, loc="outside lower left", ncols=3, fontsize=8, labelcolor=MUTED)
    _quiet_lines(ax)
    return fig


def threshold_chart(result: SweepResult) -> Figure:
    return threshold_chart_rows([row.model_dump() for row in result.rows], result.recommended)


def bench_chart(summary: BenchSummary) -> Figure:
    """E6: cache off vs on, one panel per measure."""
    off, on = summary.passes["off"], summary.passes["on"]
    names = ["Cache off", "Cache on"]
    return panel_bars(
        {
            "LLM requests": pd.Series([off.llm_requests, on.llm_requests], index=names),
            "Latency p95, ms": pd.Series([off.latency_p95_ms or 0, on.latency_p95_ms or 0], index=names),
            "Judged correct": pd.Series([off.correct.n, on.correct.n], index=names),
        },
        f"Cache off vs on, {off.n_items} questions",
        2.0,
    )


# Dark charts for the redesigned dashboard (2026-11-08): the card colour is the figure colour, so a chart sits flat
# inside its card. Same data rules as above: one series per chart, direct labels, the table carries the numbers.
DARK_SURFACE = "#151922"
DARK_INK = "#E6E8EE"
DARK_MUTED = "#9AA3B2"
DARK_GRID = "#232833"
DARK_ACCENT = "#6E8BFF"
DARK_WARN = "#E5B454"


def _dark_canvas(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), layout="constrained", facecolor=DARK_SURFACE)
    ax = fig.add_subplot(1, 1, 1, facecolor=DARK_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(DARK_GRID)
    ax.tick_params(colors=DARK_MUTED, length=0, labelsize=8)
    ax.grid(axis="y", color=DARK_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig


def hit_rate_trend(requests: pd.DataFrame, last: int = 20) -> Figure:
    """Cumulative hit rate over the last `last` cache lookups, with the latest value labelled."""
    recent = requests.tail(last)
    rate = recent.cache_hit.astype(float).expanding().mean().to_numpy()
    x = np.arange(1, len(rate) + 1)
    fig = _dark_canvas(7.6, 2.6)
    ax = fig.axes[0]
    ax.fill_between(x, rate, color=DARK_ACCENT, alpha=0.14, linewidth=0)
    ax.plot(x, rate, color=DARK_ACCENT, linewidth=2)
    ax.scatter(x[:1], rate[:1], s=36, facecolor=DARK_SURFACE, edgecolor=DARK_ACCENT, linewidth=1.6, zorder=3)
    ax.scatter(x[-1:], rate[-1:], s=36, color=DARK_ACCENT, zorder=3)
    ax.annotate(f"{rate[-1]:.0%}", (x[-1], rate[-1]), xytext=(0, 9), textcoords="offset points", ha="center",
                color=DARK_INK, fontsize=9)  # fmt: skip
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0.7, max(len(rate), 2) + 0.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(x if len(x) <= 20 else x[::5])
    ax.set_xlabel("Request", color=DARK_MUTED, fontsize=8)
    return fig


def similarity_zones(requests: pd.DataFrame, threshold: float) -> Figure:
    """Where the lookups landed: misses below the threshold, hits at or above it."""
    seen = requests[requests.cache_similarity.notna()]
    hit = seen.cache_hit.astype(bool)
    fig = _dark_canvas(3.9, 2.4)
    ax = fig.axes[0]
    ax.grid(False)
    ax.axvspan(0, threshold, color=DARK_WARN, alpha=0.10, linewidth=0)
    ax.axvspan(threshold, 1, color=DARK_ACCENT, alpha=0.22, linewidth=0)
    ax.axvline(threshold, color=DARK_INK, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(threshold, 1.04, f"threshold {threshold:.2f}", color=DARK_MUTED, fontsize=7.5, ha="right",
            transform=ax.get_xaxis_transform())  # fmt: skip
    ax.text(threshold / 2, 0.5, "Miss zone", color=DARK_WARN, fontsize=8, ha="center", va="center",
            transform=ax.get_xaxis_transform())  # fmt: skip
    rng = np.random.default_rng(0)
    for values, color in (
        (seen.cache_similarity[~hit], DARK_WARN),
        (seen.cache_similarity[hit], DARK_ACCENT),
    ):
        if len(values):
            ax.scatter(values, rng.uniform(0.25, 0.75, len(values)), s=22, color=color, edgecolor=DARK_SURFACE,
                       linewidth=1, zorder=3, transform=ax.get_xaxis_transform())  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Cosine similarity", color=DARK_MUTED, fontsize=8)
    return fig
