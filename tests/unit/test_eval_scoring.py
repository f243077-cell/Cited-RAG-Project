import pytest

from citerag.errors import ValidationFailedError
from citerag.eval.experiments import Experiment, apply_overrides, load_experiment
from citerag.eval.scoring import gate_sweep, leaked, needs_judge, propose_gate, summarize
from citerag.schemas import (
    Citation,
    CitedAnswer,
    GoldenSource,
    ItemRecord,
    JudgeVerdict,
    LatencyMs,
    UsageInfo,
)

GOLD = [GoldenSource(doc="policy.pdf", page=3)]


def citation(valid=True, page=3):
    return Citation(
        marker=1,
        chunk_id="c",
        doc_id="d",
        doc_name="policy.pdf",
        page=page,
        quote="q",
        valid=valid,
        match_score=99,
    )


def record(kind="answerable", answerable=True, citations=(), verdict=None, **kw):
    answer = CitedAnswer(answer="text", answerable=answerable, confidence="high", citations=list(citations))
    usage = UsageInfo(input_tokens=10, output_tokens=4, llm_requests=1, latency_ms=LatencyMs(total=1000))
    fields = {
        "id": "x",
        "type": kw.pop("type", kind),
        "question": "q",
        "status": "answered",
        "answer": answer,
    }
    return ItemRecord(**fields, gold_sources=GOLD, usage=usage, verdict=verdict, **kw)


def verdict(correctness="correct", faithful=True):
    return JudgeVerdict(faithful=faithful, correctness=correctness, reasoning="r")


def test_rates_keep_their_counts():
    summary = summarize(
        [
            record(citations=[citation()], verdict=verdict()),
            record(citations=[citation(page=9)], verdict=verdict("partial", faithful=False)),
            record(citations=[citation(valid=False)], verdict=verdict("incorrect")),
            record(answerable=False),  # a refused answerable item
        ]
    )
    assert (summary.faithfulness.n, summary.faithfulness.of) == (2, 3)
    assert (summary.correctness.n, summary.correctness.of) == (1, 4)  # the refusal counts as not correct
    assert (summary.correct_or_partial.n, summary.correct_or_partial.of) == (2, 4)
    assert (summary.citation_validity.n, summary.citation_validity.of) == (2, 3)
    assert (summary.citation_precision.n, summary.citation_precision.of) == (1, 2)
    assert (summary.false_refusal.n, summary.false_refusal.of) == (1, 4)


def test_refusals_and_leaks_need_no_judge():
    summary = summarize(
        [
            record("unanswerable", answerable=False),
            record("unanswerable", answerable=True),
            record("injection", canary="CANARY-1"),
        ]
    )
    assert (summary.correct_refusal.n, summary.correct_refusal.of) == (1, 2)
    assert summary.injection_leaks == 0 and summary.injection_items == 1
    assert summary.pending_judge == 0


def test_unmeasured_rates_are_none_not_zero():
    summary = summarize([record("unanswerable", answerable=False)])
    assert summary.faithfulness.rate is None and summary.false_refusal.rate is None


def test_canary_match_ignores_case():
    item = record("injection", canary="CANARY-1")
    item.answer.answer = "here is canary-1"
    assert leaked(item)
    assert summarize([item]).injection_leaks == 1


def test_only_answered_scored_items_wait_for_the_judge():
    assert needs_judge(record())
    assert not needs_judge(record(verdict=verdict()))
    assert not needs_judge(record(answerable=False))
    assert not needs_judge(record("unanswerable"))


def test_failed_items_count_as_errors():
    failed = ItemRecord(
        id="e", type="answerable", question="q", status="error", error_code="structured_output_failed"
    )
    summary = summarize([failed, record()])
    assert summary.errors == 1 and summary.false_refusal.of == 1


def test_operations_use_every_answered_item():
    summary = summarize([record(), record()])
    assert summary.latency_p50_ms == 1000 and summary.mean_llm_requests == 1


def test_gate_proposal_is_the_highest_value_with_no_false_refusal():
    rows = gate_sweep([0.70, 0.88], [0.51, 0.77])
    best = propose_gate(rows)
    assert best.tau == 0.70 and best.false_refusals == 0 and best.gate_refusals == 1
    assert next(r for r in rows if r.tau == 0.71).false_refusals == 1


def test_gate_proposal_is_none_when_nothing_is_safe():
    assert propose_gate(gate_sweep([0.1], [0.1])) is None


def test_experiment_overrides_are_validated(settings):
    changed = apply_overrides(settings, {"chunk_size": "256", "top_k_context": 3})
    assert changed.chunk_size == 256 and settings.chunk_size == 512
    with pytest.raises(ValidationFailedError):
        apply_overrides(settings, {"chunk_size": "many"})


@pytest.mark.parametrize("key", ["no_such_setting", "openrouter_api_key", "gemini_api_key"])
def test_experiments_cannot_set_unknown_or_secret_keys(settings, key):
    with pytest.raises(ValidationFailedError):
        apply_overrides(settings, {key: "x"})


def test_every_experiment_file_loads(settings):
    from pathlib import Path

    files = sorted((Path(__file__).resolve().parents[2] / "configs" / "experiments").glob("*.yaml"))
    assert len(files) == 18
    for path in files:
        experiment = load_experiment(path)
        assert isinstance(experiment, Experiment) and experiment.name == path.stem
        apply_overrides(settings, experiment.overrides)
