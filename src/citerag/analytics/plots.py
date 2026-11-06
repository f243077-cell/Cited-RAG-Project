"""Charts as pure functions: data in, Matplotlib Figure out. No pyplot, so nothing global is touched.
Bars carry direct labels and every page shows the same numbers as a table, so no reading rests on color alone."""

from __future__ import annotations

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from citerag.schemas import ItemRecord

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
