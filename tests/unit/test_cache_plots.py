"""Smoke tests for the cache and dashboard charts (Agg, synthetic data)."""

import pandas as pd
from matplotlib.figure import Figure

from citerag.analytics import plots
from citerag.eval.bench import BenchSummary, PassSummary
from citerag.eval.cache_tools import SweepResult, SweepRow
from citerag.schemas import Rate

STAGES = (
    "lat_condense",
    "lat_embed",
    "lat_cache",
    "lat_retrieve",
    "lat_rerank",
    "lat_generate",
    "lat_verify",
)


def requests():
    return pd.DataFrame(
        {
            "cache_hit": [False, True, False, True],
            "cache_enabled": [True, True, True, True],
            "cache_similarity": [0.5, 0.97, None, 0.95],
            "llm_requests": [1, 0, 2, 0],
            "saved_requests": [None, 1, None, 2],
            "lat_total": [9000, 20, 12000, 25],
            **{stage: [100, 1, 200, 1] for stage in STAGES},
        }
    )


def is_chart(figure):
    return isinstance(figure, Figure) and len(figure.axes) >= 1


def test_each_dashboard_chart_returns_a_figure():
    frame = requests()
    calls = pd.DataFrame(
        {"role": ["generate", "judge", "generate"], "input_tokens": [5, 6, 7], "output_tokens": [1, 2, 3]}
    )
    assert is_chart(plots.cache_hit_rate(frame))
    assert is_chart(plots.latency_hit_miss(frame))
    assert is_chart(plots.cumulative_requests(frame))
    assert is_chart(plots.calls_by_role(calls))
    assert is_chart(plots.stage_latency(frame))
    assert is_chart(plots.similarity_hist(frame, 0.92))


def test_charts_cope_with_only_misses_or_only_hits():
    frame = requests()
    for part in (frame[~frame.cache_hit], frame[frame.cache_hit]):
        assert is_chart(plots.latency_hit_miss(part))
        assert is_chart(plots.similarity_hist(part, 0.92))
        assert is_chart(plots.cumulative_requests(part.reset_index(drop=True)))


def test_without_the_cache_line_adds_back_what_hits_saved():
    fig = plots.cumulative_requests(requests())
    without, actual = fig.axes[0].get_lines()
    assert without.get_ydata()[-1] == 6 and actual.get_ydata()[-1] == 3


def test_the_stage_chart_uses_misses_only():
    fig = plots.stage_latency(requests())
    bars = [patch.get_width() for patch in fig.axes[0].patches]
    assert max(bars) == 150  # the mean of 100 and 200: the 1 ms of the hits is left out


def sweep(recommended):
    rows = [SweepRow(tau=t, precision=1.0, recall=1 - t, false_hit_rate=0.0) for t in (0.7, 0.8, 0.9)]
    return SweepResult(
        n_equivalent=30,
        n_near_miss=30,
        enough_pairs=True,
        max_false_hit_rate=0.02,
        recommended=recommended,
        rows=rows,
        rows_without_guard=rows,
        worst_near_misses=[],
    )


def test_the_sweep_chart_marks_the_threshold_only_when_there_is_one():
    assert len(plots.threshold_chart(sweep(0.8)).axes[0].get_lines()) == 4
    assert len(plots.threshold_chart(sweep(None)).axes[0].get_lines()) == 3


def test_the_bench_chart_has_one_panel_per_measure():
    def one(name, requests_used):
        return PassSummary(
            name=name,
            n_items=2,
            n_done=2,
            hits=0,
            hit_rate=Rate(n=0, of=2, rate=0),
            errors=0,
            latency_p95_ms=900.0,
            llm_requests=requests_used,
            judge_requests=0,
            input_tokens=0,
            output_tokens=0,
            shadow_usd=0,
            correct=Rate(n=2, of=2, rate=1),
        )

    summary = BenchSummary(
        run_id="r",
        status="complete",
        workload="w",
        threshold=0.92,
        repeat_share=0.5,
        passes={"off": one("off", 2), "on": one("on", 1)},
        requests_used=3,
    )
    assert len(plots.bench_chart(summary).axes) == 3


def test_the_dark_dashboard_charts_return_figures_and_cope_with_one_kind_only():
    frame = requests()
    assert is_chart(plots.hit_rate_trend(frame)) and is_chart(plots.similarity_zones(frame, 0.92))
    for part in (frame[~frame.cache_hit], frame[frame.cache_hit], frame.head(1)):
        assert is_chart(plots.hit_rate_trend(part.reset_index(drop=True)))
        assert is_chart(plots.similarity_zones(part, 0.92))


def test_the_hit_rate_trend_labels_the_latest_rate():
    figure = plots.hit_rate_trend(requests())
    assert "50%" in [t.get_text() for t in figure.axes[0].texts]
