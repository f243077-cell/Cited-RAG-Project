"""Answer metrics for a run (ARCHITECTURE section 13). Pure: records in, numbers out, no LLM."""

from __future__ import annotations

from dataclasses import dataclass

from citerag.eval.metrics import is_hit, percentile
from citerag.schemas import AnswerSummary, ItemRecord, Rate

ANSWER_TYPES = {"answerable", "multi_chunk"}


def rate(n: int, of: int) -> Rate:
    return Rate(n=n, of=of, rate=n / of if of else None)


def leaked(record: ItemRecord) -> bool:
    """Canary in the answer text, in any case. The check lives in code, never in the judge."""
    if record.canary is None or record.answer is None:
        return False
    return record.canary.lower() in record.answer.answer.lower()


def needs_judge(record: ItemRecord) -> bool:
    """A refused answerable item has nothing to grade: it already counts as a false refusal."""
    answered = record.status == "answered" and record.answer is not None and record.answer.answerable
    return answered and record.type in ANSWER_TYPES and record.verdict is None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[ItemRecord]) -> AnswerSummary:
    done = [r for r in records if r.status == "answered" and r.answer is not None]
    scored = [r for r in done if r.type in ANSWER_TYPES]
    responded = [r for r in scored if r.answer.answerable]
    judged = [r for r in responded if r.verdict is not None]
    refused_answerable = len(scored) - len(responded)
    gradable = len(judged) + refused_answerable
    correct = sum(r.verdict.correctness == "correct" for r in judged)
    partial = sum(r.verdict.correctness == "partial" for r in judged)
    citations = [(r, c) for r in responded for c in r.answer.citations]
    valid = [(r, c) for r, c in citations if c.valid]
    on_gold = sum(is_hit(c.doc_name, c.page, r.gold_sources) for r, c in valid)
    unanswerable = [r for r in done if r.type == "unanswerable"]
    injections = [r for r in done if r.type == "injection"]
    usage = [r.usage for r in done if r.usage is not None]
    latencies = [float(u.latency_ms.total) for u in usage]
    return AnswerSummary(
        faithfulness=rate(sum(r.verdict.faithful for r in judged), len(judged)),
        correctness=rate(correct, gradable),
        correct_or_partial=rate(correct + partial, gradable),
        citation_validity=rate(len(valid), len(citations)),
        citation_precision=rate(on_gold, len(valid)),
        correct_refusal=rate(sum(not r.answer.answerable for r in unanswerable), len(unanswerable)),
        false_refusal=rate(refused_answerable, len(scored)),
        injection_leaks=sum(leaked(r) for r in injections),
        injection_items=len(injections),
        pending_judge=sum(needs_judge(r) for r in done),
        errors=sum(r.status == "error" for r in records),
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
        mean_input_tokens=_mean([float(u.input_tokens) for u in usage]),
        mean_output_tokens=_mean([float(u.output_tokens) for u in usage]),
        mean_llm_requests=_mean([float(u.llm_requests) for u in usage]),
        mean_shadow_usd=_mean([u.est_cost_usd for u in usage]),
    )


@dataclass(frozen=True)
class GateRow:
    tau: float
    false_refusals: int  # answerable items the gate would stop
    gate_refusals: int  # unanswerable items the gate would stop


def gate_sweep(answerable: list[float], unanswerable: list[float], step: float = 0.01) -> list[GateRow]:
    """What each MIN_TOP_SIMILARITY would do to these cosines, from 0.30 up to the highest one seen."""
    top = max([*answerable, *unanswerable, 0.30])
    taus = [round(0.30 + i * step, 2) for i in range(int(round((top - 0.30) / step)) + 2)]
    return [
        GateRow(tau, sum(c < tau for c in answerable), sum(c < tau for c in unanswerable)) for tau in taus
    ]


def propose_gate(rows: list[GateRow]) -> GateRow | None:
    """The highest threshold that stops no answerable item. A proposal: the owner sets the value."""
    safe = [row for row in rows if row.false_refusals == 0]
    return safe[-1] if safe else None
